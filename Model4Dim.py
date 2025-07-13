import numpy as np
import jax.numpy as jnp
from jax import random,vmap,grad,jit,lax,debug,value_and_grad
from scipy.stats import linregress
from scipy.optimize import minimize
import time
from Common import *
from CartPole import *

def collect_data(N, data_range=[5, 5, jnp.pi, 20], seed=None):
    if seed is None:
        seed = int(time.time())
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, 5)
    X = jnp.stack([
        random.uniform(subkeys[0], shape=(N,), minval=-data_range[0], maxval=data_range[0]),
        random.uniform(subkeys[1], shape=(N,), minval=-data_range[1], maxval=data_range[1]),
        random.uniform(subkeys[2], shape=(N,), minval=-data_range[2], maxval=data_range[2]),
        random.uniform(subkeys[3], shape=(N,), minval=-data_range[3], maxval=data_range[3]),
    ], axis=1)

    cartpole = CartPole(visual=False)
    Y_list = []
    for i in range(N):
        cartpole.setState(X[i])
        cartpole.performAction(0.0)
        y = cartpole.getState() - X[i]
        Y_list.append(y)
    Y = jnp.stack(Y_list)

    axis_names = ['Cart Location (m)', 'Cart Velocity (m/s)', 'Pole Angle (rad)', 'Pole Velocity (rad/s)']
    return X, Y, axis_names

def fit_linear(X, Y):
    N = X.shape[0]
    X_aug = jnp.concatenate([X, jnp.ones((N, 1))], axis=1)
    result = jnp.linalg.solve(X_aug.T @ X_aug, X_aug.T @ Y)
    A = result[:-1]  # coefficients
    b = result[-1]      # intercept
    return A, b

def linear_MSE(X, Y, A, b):
    Y_pred = (X @ A.T) + b
    return jnp.mean((Y - Y_pred) ** 2, axis=0)

def linear_residuals(X, Y, A, b):
    Y_pred = (X @ A.T) + b
    return Y - Y_pred

def wrap_angle(theta):
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
def linear_rollout(initial_state, A, b, end_time=10.0, dt=0.1):
    steps = int(end_time / dt)
    initial_state = jnp.asarray(initial_state)
    
    def step_fn(carry, _):
        x, time = carry
        d_state = (x @ A.T) + b
        new_state = x + d_state
        new_state = new_state.at[2].set(wrap_angle(new_state[2]))
        new_time = time + dt
        state = jnp.concatenate([new_state, jnp.array([new_time])])
        return (new_state, new_time), state

    _, states = lax.scan(step_fn, (initial_state, 0.0), xs=None, length=steps)

    return states    

def select_basis_centres(X, M, seed=None):
    if seed is None:
        seed = int(time.time())
    key = random.PRNGKey(seed)
    indices = random.permutation(key, X.shape[0])[:M]
    return X[indices]

@jit
def K_func_single(x_1, x_2, sigma):
    dX = x_1-x_2
    dX.at[2].set(jnp.sin(dX[2] / 2))
    return jnp.exp(-jnp.sum((dX ** 2) / (2 * sigma)))

@jit
def K_func_array(X_1, X_2, sigma):
    K = vmap(lambda x_1: vmap(lambda x_2: K_func_single(x_1, x_2, sigma))(X_2))(X_1)
    return K

@jit
def fit_alpha(X, X_basis, Y, sigma, lam):
    K_MN = K_func_array(X_basis, X, sigma)
    K_MM = K_func_array(X_basis, X_basis, sigma)
    A = K_MN @ K_MN.T + lam * K_MM
    B = K_MN @ Y
    alpha = jnp.linalg.lstsq(A, B, rcond=None)[0]
    return alpha
    
@jit
def non_linear_prediction(x, X_basis, alpha, sigma):
    K = vmap(lambda x_basis: K_func_single(x, x_basis, sigma))(X_basis)
    return K @ alpha

def non_linear_rollout(initial_state, X_basis, alpha, sigma, end_time=10.0, dt=0.1):
    steps = int(end_time / dt)
    initial_state = jnp.asarray(initial_state)

    @jit
    def step_fn(carry, _):
        state, time = carry  
        d_state = vmap(lambda a, s: non_linear_prediction(state, X_basis, a, s), in_axes=(1, 0))(alpha, sigma)
        new_state = state + d_state
        new_state = new_state.at[2].set(wrap_angle(new_state[2]))
        new_time = time + dt
        state = jnp.concatenate([new_state, jnp.array([new_time])])
        return (new_state, new_time), state

    init_carry = (initial_state, 0.0)
    _, states = lax.scan(step_fn, init_carry, xs=None, length=steps)

    return states

@jit
def non_linear_MSE(X, Y, X_basis, alpha, sigma):
    def predict(x):
        K = vmap(lambda x_b: K_func_single(x, x_b, sigma))(X_basis)
        return K @ alpha
    Y_pred = vmap(predict)(X)
    return jnp.mean((Y - Y_pred) ** 2)

def fit_hyperparameters(X_train, Y_train, X_basis, X_test, Y_test):
    @jit
    def obj_func(params_jax):
        sigma = params_jax[:-1]
        lam = params_jax[-1]
        alpha = fit_alpha(X_train, X_basis, Y_train, sigma, lam)
        return non_linear_MSE(X_test, Y_test, X_basis, alpha, sigma)

    value_and_grad_func = jit(value_and_grad(obj_func))

    def format_func(params_np):
        params_jax = jnp.array(params_np)
        loss, grad_val = value_and_grad_func(params_jax)
        return float(loss), np.array(grad_val)

    result = minimize(
        fun=lambda p: format_func(p),
        x0=np.concatenate([np.std(X_train, axis=0), [0.01]]),
        method='L-BFGS-B',
        jac=True,
        bounds=[(0, None)] * (1 + np.shape(X_train)[1])
    )
    sigma = jnp.array(result.x[:-1])
    lam = jnp.array(result.x[-1])
    alpha = fit_alpha(X_train, X_basis, Y_train, sigma, lam)
    return sigma, lam, alpha, result.fun

def joint_rollout(initial_state, A, b, X_basis, alpha, sigma, end_time=10.0, dt=0.1):
    steps = int(end_time / dt)
    initial_state = jnp.asarray(initial_state)

    @jit
    def step_fn(carry, _):
        curr_state, time = carry
        x = curr_state
        d_state_linear = (x @ A.T) + b
        d_state_non_linear = vmap(lambda a, s: non_linear_prediction(x, X_basis, a, s), in_axes=(1, 0))(alpha, sigma)
        new_state = curr_state + d_state_linear + d_state_non_linear
        new_state = new_state.at[2].set(wrap_angle(new_state[2]))
        new_time = time + dt
        state = jnp.concatenate([new_state, jnp.array([new_time])])
        return (new_state, new_time), state

    init_carry = (initial_state, 0.0)
    _, states = lax.scan(step_fn, init_carry, xs=None, length=steps)

    return states

def joint_MSE(X, Y, X_basis, A, b, alpha, sigma):
    def predict(x):
        K = vmap(lambda x_b: K_func_single(x, x_b, sigma))(X_basis)
        return K @ alpha
    R_pred = vmap(predict)(X)
    Y_pred = R_pred + (X @ A.T) + b
    return jnp.mean((Y - Y_pred) ** 2)