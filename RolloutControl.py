import numpy as np
import jax.numpy as jnp
from jax import random,vmap,debug,grad,jit,lax
from scipy.stats import linregress
from scipy.optimize import minimize
import time
from Common import *
from CartPole import *

def wrap_angle(theta):
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi


def find_loss(x,sigma_l):
    return 1 - jnp.exp(-jnp.sum((x**2)/(2.0 * (sigma_l**2))))


def apply_policy(x, p):
    return jnp.clip(x @ p, -20.0, 20.0)


def perform_action_step(state, action):
    """
    Performs one full environment step (all sim_steps) on the CartPole state.
    This function is fully functional and JAX-compatible.
    
    Args:
        state: jnp.array([x, x_dot, theta, theta_dot])
        action: scalar force value

    Returns:
        new_state: jnp.array([x, x_dot, theta, theta_dot])
    """
    # --- Config (hardcoded for self-containedness) ---
    pole_length = 0.5
    pole_mass = 0.5
    cart_mass = 0.5
    mu_c = 0.001
    mu_p = 0.001
    gravity = 9.8
    max_force = 20.0
    delta_time = 0.1
    sim_steps = 50

    # Clip force via tanh
    force = max_force * jnp.tanh(action / max_force)

    def euler_step(state, _):
        x, x_dot, theta, theta_dot = state
        s = jnp.sin(theta)
        c = jnp.cos(theta)
        m = 4.0 * (cart_mass + pole_mass) - 3.0 * pole_mass * (c**2)

        cart_accel = (2.0 * (pole_length * pole_mass * (theta_dot**2) * s +
                             2.0 * (force - mu_c * x_dot)) -
                      3.0 * pole_mass * gravity * c * s +
                      6.0 * mu_p * theta_dot * c / pole_length) / m

        pole_accel = (-3.0 * c * (2.0 / pole_length) *
                      (pole_length / 2.0 * pole_mass * (theta_dot**2) * s +
                       force - mu_c * x_dot) +
                      6.0 * (cart_mass + pole_mass) /
                      (pole_mass * pole_length) *
                      (pole_mass * gravity * s -
                       2.0 / pole_length * mu_p * theta_dot)) / m

        dt = delta_time / sim_steps
        x_dot_new = x_dot + dt * cart_accel
        theta_dot_new = theta_dot + dt * pole_accel
        theta_new = theta + dt * theta_dot_new
        x_new = x + dt * x_dot_new

        return jnp.array([x_new, x_dot_new, theta_new, theta_dot_new]), None

    final_state, _ = lax.scan(euler_step, state, None, length=sim_steps)
    return final_state

def true_policy_rollout(initial_state, p, sigma_l, end_time=10.0, dt=0.1):
    initial_state = jnp.array(initial_state)

    num_steps = int(end_time / dt)

    def step_fn(carry, _):
        curr_state, time, cum_loss = carry
        x = conv_4_to_5_single(curr_state)
        action = apply_policy(x, p)
        new_state = perform_action_step(curr_state, action)
        new_state = new_state + np.random.randn(new_state.shape[0]) * (0.1 * (new_state-curr_state))
        new_state = new_state.at[2].set(wrap_angle(new_state[2]))
        new_time = time + dt
        new_cum_loss = cum_loss + find_loss(new_state, sigma_l)
        state_with_time = jnp.concatenate([new_state, jnp.array([action,new_time])])
        return (new_state, new_time, new_cum_loss), state_with_time

    (_, _, total_loss), states = lax.scan(step_fn, (initial_state, 0.0, 0.0), xs=None, length=num_steps)

    initial_with_time = jnp.concatenate([initial_state, jnp.array([0.0, 0.0])])[None, :]
    states = jnp.concatenate([initial_with_time, states],axis=0)
    return states, total_loss

def true_policy_loss(initial_state, p, sigma_l, horizon=5.0, dt=0.1):
    initial_state = jnp.array(initial_state)
    num_steps = int(horizon / dt)

    @jit
    def step_fn(carry, _):
        curr_state, cum_loss = carry
        x = conv_4_to_5_single(curr_state)
        action = apply_policy(x, p)
        new_state = perform_action_step(curr_state, action)
        new_state = new_state + np.random.randn(new_state.shape[0]) * (0.8 * (new_state-curr_state))
        new_state = new_state.at[2].set(wrap_angle(new_state[2]))
        new_cum_loss = cum_loss + find_loss(new_state, sigma_l)
        return (new_state, new_cum_loss), None  # returning dummy output

    (_, total_loss), _ = lax.scan(step_fn, (initial_state, 0.0), xs=None, length=num_steps)
    return total_loss

def true_fit_policy(initial_state, initial_guess, sigma_l, horizon=5.0, dt=0.1):
    initial_guess = np.asarray(initial_guess, dtype=np.float64)
    @jit
    def obj_func(p):
        return true_policy_loss(initial_state, p, sigma_l, horizon, dt)

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

def true_repeated_fit_policy(initial_state, sigma_l, samples, end_time=5.0):
    guesses = np.random.randn(samples, 5) * np.array([25.0, 10.0, 50.0, 25.0, 10.0])

    best_loss = float('inf')
    best_p = None

    for i in range(samples):
        p_rand, loss_rand = true_fit_policy(initial_state, guesses[i], sigma_l, end_time)
        if loss_rand < best_loss:
            best_loss = loss_rand
            best_p = p_rand

    return best_p, best_loss

def p_scan (initial_state, initial_p, sigma_l, scan_index, scan_min, scan_max, horizon = 5):
    # Generate 1000 linearly spaced values
    scan_values = jnp.linspace(scan_min, scan_max, 1000)

    def evaluate_loss(x):
        modified_p = initial_p.at[scan_index].set(x)
        return true_policy_loss(initial_state, modified_p, sigma_l, horizon)
    losses = vmap(evaluate_loss)(scan_values)

    # Convert to NumPy for plotting
    scan_values_np = np.array(scan_values)
    losses_np = np.array(losses)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(scan_values_np, losses_np, label="Loss vs. parameter")
    plt.xlabel(f'Value at index {scan_index}')
    plt.ylabel('True Policy Loss')
    plt.title('Scan of Policy Loss over Parameter Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()