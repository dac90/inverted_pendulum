import numpy as np
import jax.numpy as jnp
from jax import random,vmap,grad,jit,lax,debug,value_and_grad
from scipy.stats import linregress
from scipy.optimize import minimize
import time
from Common import *
from CartPole import *

def collect_data(N, data_range=[12,8,jnp.pi,16,20], seed=None):
    if seed is None:
        seed = int(time.time())
    print(seed)
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, 6)

    X_4dim = jnp.stack([
        random.uniform(subkeys[0], shape=(N,), minval=-data_range[0], maxval=data_range[0]),
        random.uniform(subkeys[1], shape=(N,), minval=-data_range[1], maxval=data_range[1]),
        random.uniform(subkeys[2], shape=(N,), minval=-data_range[2], maxval=data_range[2]),
        random.uniform(subkeys[3], shape=(N,), minval=-data_range[3], maxval=data_range[3]),
        random.uniform(subkeys[4], shape=(N,), minval=-data_range[4], maxval=data_range[4]),
    ], axis=1)
    X_5dim = conv_4_to_5_array(X_4dim)
    
    cartpole = CartPole(visual=False)
    Y_list = []
    for i in range(N):
        cartpole.setState(X_4dim[i,:-1])
        cartpole.performAction(X_4dim[i,-1])
        y = cartpole.getState() - X_4dim[i,:-1]
        Y_list.append(y)
    Y = jnp.stack(Y_list)

    axis_names = ['Cart Location (m)', 'Cart Velocity (m/s)', 'Sin Pole Angle', 'Cos Pole Angle', 'Pole Velocity (rad/s)', 'Force (N)']

    return X_5dim, Y, axis_names

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
    
@jit
def wrap_angle(theta):
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

def linear_rollout(initial_state, A, b, end_time=10.0, dt=0.1):
    steps = int(end_time / dt)
    initial_state = jnp.asarray(initial_state)

    @jit
    def step_fn(carry, _):
        curr_state, time = carry
        x = conv_4_to_5_single(curr_state)
        d_state = (A @ x) + b
        new_state = curr_state + d_state
        new_state = new_state.at[2].set(wrap_angle(new_state[2]))
        new_time = time + dt
        state = jnp.concatenate([new_state, jnp.array([new_time])])
        return (new_state, new_time), state

    init_carry = (initial_state, 0.0)
    _, states = lax.scan(step_fn, init_carry, xs=None, length=steps)

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
    return jnp.exp(-jnp.sum((dX ** 2) / (2 * sigma)))

@jit
def K_func_array(X_1, X_2, sigma):
    K = vmap(lambda x_1: vmap(lambda x_2: K_func_single(x_1, x_2, sigma))(X_2))(X_1)
    return K

@jit
def fit_alpha(X, X_basis, Y, sigma, lam):
    K_MN = K_func_array(X_basis, X, sigma)
    K_MM = K_func_array(X_basis, X_basis, sigma)
    A = (K_MN @ K_MN.T) + (lam * K_MM)
    B = K_MN @ Y 
    alpha = jnp.linalg.lstsq(A, B, rcond=None)[0]
    return alpha

@jit
def non_linear_prediction(x, X_basis, alpha, sigma):
    K = vmap(lambda x_basis: K_func_single(x, x_basis, sigma))(X_basis)
    return K @ alpha

def non_linear_rollout(initial_state, action, X_basis, alpha, sigma, end_time=10.0, dt=0.1):
    steps = int(end_time / dt)
    initial_state = jnp.asarray(initial_state)

    @jit
    def step_fn(carry, _):
        curr_state, time = carry
        x = jnp.append(conv_4_to_5_single(curr_state),action)
        d_state = vmap(lambda a, s: non_linear_prediction(x, X_basis, a, s), in_axes=(1, 0))(alpha, sigma)
        new_state = curr_state + d_state
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

def joint_rollout(initial_state, action, A, b, X_basis, alpha, sigma, end_time=10.0, dt=0.1):
    steps = int(end_time / dt)
    initial_state = jnp.asarray(initial_state)

    @jit
    def step_fn(carry, _):
        curr_state, time = carry
        x = jnp.append(conv_4_to_5_single(curr_state),action)
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

@jit
def find_loss(x,sigma_l=jnp.array([0.5,0.5,0.5,0.5])):
    return 1-jnp.exp(-jnp.sum((x**2)/(2.0 * sigma_l**2)))

@jit
def apply_policy(x, p):
    return jnp.clip(x @ p, -20.0, 20.0)

def linear_policy_rollout(initial_state, p, sigma_l, A, b, end_time=10.0, dt=0.1):
    steps = int(end_time / dt)
    initial_state = jnp.asarray(initial_state)

    @jit
    def step_fn(carry, _):
        curr_state, time, cum_loss = carry
        action = apply_policy(conv_4_to_5_single(curr_state), p)
        x = jnp.append(conv_4_to_5_single(curr_state), action)
        d_state = (x @ A.T) + b
        new_state = curr_state + d_state
        new_state = new_state.at[2].set(wrap_angle(new_state[2]))
        new_time = time + dt
        new_cum_loss = cum_loss + find_loss(new_state)
        state = jnp.concatenate([new_state, jnp.array([action, new_time])])
        return (new_state, new_time, new_cum_loss), state

    init_carry = (initial_state, 0.0, 0.0)
    (_, _, total_loss), states = lax.scan(step_fn, init_carry, xs=None, length=steps)

    return states, total_loss

def non_linear_policy_rollout(initial_state, p, sigma_l, X_basis, alpha, sigma, end_time=10.0, dt=0.1):
    steps = int(end_time / dt)
    initial_state = jnp.asarray(initial_state)

    @jit
    def step_fn(carry, _):
        curr_state, time, cum_loss = carry
        action = apply_policy(conv_4_to_5_single(curr_state), p)
        x = jnp.append(conv_4_to_5_single(curr_state), action)
        d_state = vmap(lambda a, s: non_linear_prediction(x, X_basis, a, s), in_axes=(1, 0))(alpha, sigma)
        new_state = curr_state + d_state
        new_state = new_state.at[2].set(wrap_angle(new_state[2]))
        new_time = time + dt
        new_cum_loss = cum_loss + find_loss(new_state)
        state_with_time = jnp.concatenate([new_state, jnp.array([action, new_time])])
        return (new_state, new_time, new_cum_loss), state_with_time

    init_carry = (initial_state, 0.0, 0.0)
    (_, _, total_loss), states = lax.scan(step_fn, init_carry, xs=None, length=steps)

    # Prepend initial state (with time = 0)
    initial_with_time = jnp.concatenate([initial_state, jnp.array([0.0, 0.0])])[None, :]
    states = jnp.concatenate([initial_with_time, states],axis=0)
    return states, total_loss

def joint_policy_rollout(initial_state, p, sigma_l, A, b, X_basis, alpha, sigma, end_time=10.0, dt=0.1):
    steps = int(end_time / dt)
    initial_state = jnp.asarray(initial_state)

    @jit
    def step_fn(carry, _):
        curr_state, time, cum_loss = carry
        action = apply_policy(conv_4_to_5_single(curr_state), p)
        x = jnp.append(conv_4_to_5_single(curr_state), action)
        d_state_linear = (x @ A.T) + b
        d_state_non_linear = vmap(lambda a, s: non_linear_prediction(x, X_basis, a, s), in_axes=(1, 0))(alpha, sigma)
        new_state = curr_state + d_state_linear + d_state_non_linear
        new_state = new_state.at[2].set(wrap_angle(new_state[2]))
        new_time = time + dt
        new_cum_loss = cum_loss + find_loss(new_state)
        state_with_time = jnp.concatenate([new_state, jnp.array([action, new_time])])
        return (new_state, new_time, new_cum_loss), state_with_time

    init_carry = (initial_state, 0.0, 0.0)
    (_, _, total_loss), states = lax.scan(step_fn, init_carry, xs=None, length=steps)

    # Prepend initial state (with time = 0)
    initial_with_time = jnp.concatenate([initial_state, jnp.array([0.0, 0.0])])[None, :]
    states = jnp.concatenate([initial_with_time, states],axis=0)
    return states, total_loss

def policy_loss(initial_state, p, sigma_l, A, b, X_basis, alpha, sigma, end_time, dt=0.1):
    steps = int(end_time / dt)
    initial_state = jnp.asarray(initial_state)
      
    @jit
    def step_fn(carry, _):
        curr_state, cum_loss = carry
        x = conv_4_to_5_single(curr_state)
        x = jnp.append(x,apply_policy(x, p))
        d_state_linear = (x @ A.T) + b
        d_state_non_linear = vmap(lambda a, s: non_linear_prediction(x, X_basis, a, s), in_axes=(1, 0))(alpha, sigma)
        new_state = curr_state + d_state_linear + d_state_non_linear
        new_state = new_state.at[2].set(wrap_angle(new_state[2]))
        new_cum_loss = cum_loss + find_loss(new_state)
        return (new_state, new_cum_loss), None  # returning dummy output

    (_, total_loss), _ = lax.scan(step_fn, (initial_state, 0.0), xs=None, length=steps)
    return total_loss

def fit_policy(initial_state, initial_guess, sigma_l, A, b, X_basis, alpha, sigma, end_time=5.0, dt=0.1):
    initial_guess = np.asarray(initial_guess, dtype=np.float64)
    @jit
    def obj_func(p):
        return policy_loss(initial_state, p, sigma_l, A, b, X_basis, alpha, sigma, end_time, dt)

    grad_func = jit(grad(obj_func))

    def format_func(p_np):
        p_jax = jnp.array(p_np)
        loss = obj_func(p_jax)
        grad_val = grad_func(p_jax)
        return float(loss), np.array(grad_val)

    result = minimize(
        fun=lambda p: format_func(p),
        x0=initial_guess,
        method='SLSQP',
        jac=True,
    )
    
    p = jnp.array(result.x)
    return p, result.fun

def repeated_fit_policy(initial_state, sigma_l, samples, A, b, X_basis, alpha, sigma, end_time=5.0):
    guesses = np.random.randn(samples, 5) * np.array([25.0, 10.0, 50.0, 25.0, 10.0])

    best_loss = float('inf')
    best_p = None

    for i in range(samples):
        p_rand, loss_rand = fit_policy(initial_state, guesses[i], sigma_l, A, b, X_basis, alpha, sigma, end_time)
        if loss_rand < best_loss:
            best_loss = loss_rand
            best_p = p_rand

    return best_p, best_loss

def time_fit_policy(initial_state, initial_guess, sigma_l, A, b, X_basis, alpha, sigma, horison=10.0, dt=0.1):
    p_list = []
    curr_p = initial_guess
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    for end_time in np.arange(1, horison+0.01, 0.5):
        candidates = []
        losses = []

        # Candidate 1: current_p
        p1, loss1 = fit_policy(initial_state, curr_p, sigma_l, A, b, X_basis, alpha, sigma, end_time)
        candidates.append(p1)
        losses.append(loss1)

        # Candidate 2: null vector
        p2, loss2 = fit_policy(initial_state, jnp.zeros_like(initial_guess), sigma_l, A, b, X_basis, alpha, sigma, end_time)
        candidates.append(p2)
        losses.append(loss2)

        # Candidates 3–5: random Gaussian initial guesses
        for _ in range(3):
            random_guess = jnp.array(rng.normal(scale=10, size=initial_guess.shape), dtype=initial_guess.dtype)
            p_rand, loss_rand = fit_policy(initial_state, random_guess, sigma_l, A, b, X_basis, alpha, sigma, end_time)
            candidates.append(p_rand)
            losses.append(loss_rand)

        # Select the best candidate
        best_idx = jnp.argmin(jnp.array(losses))
        curr_p = candidates[best_idx]
        curr_loss = losses[best_idx]

        print(f"end_time: {end_time}")
        for i, (p_i, l_i) in enumerate(zip(candidates, losses), 1):
            print(f"  Candidate {i}: loss = {l_i}")
        print(f"  -> Selected Candidate {best_idx + 1}\n")

        p_list.append(curr_p)

    p_array = jnp.stack(p_list)
    return curr_p, curr_loss, p_array