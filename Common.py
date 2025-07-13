import numpy as np
import jax.numpy as jnp
from jax import vmap,grad,jit,lax

def save_jnp_array(filename, arr):
    np_arr = np.array(arr)        # Convert to NumPy
    np.save(filename, np_arr)     # Saved as .npy file

def load_jnp_array(filename):
    np_arr = np.load(filename + ".npy")
    return jnp.array(np_arr)      # Convert back to JAX array

@jit
def conv_4_to_5_single(x):
    return jnp.concatenate([x[:2], jnp.array([jnp.sin(x[2]), jnp.cos(x[2])]), x[3:]])

@jit
def conv_4_to_5_array(X):
    sin_cos = jnp.stack([jnp.sin(X[:, 2]), jnp.cos(X[:, 2])], axis=1)
    return jnp.concatenate([X[:, :2], sin_cos, X[:, 3:]], axis=1)

@jit
def conv_5_to_4_single(x):
    angle = jnp.arctan2(x[2], x[3])  # note: correct order is sin, cos
    return jnp.concatenate([x[:2], jnp.array([angle]), x[4:]])

@jit
def conv_5_to_4_array(X):
    angles = jnp.arctan2(X[:, 2], X[:, 3])  # correct order: sin, cos
    angles = angles[:, None]  # expand to keep column dimension
    return jnp.concatenate([X[:, :2], angles, X[:, 4:]], axis=1)