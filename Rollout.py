import numpy as np
import jax.numpy as jnp
from CartPole import *

def rollout(initial_state, end_time=10):
    cartpole = CartPole(visual=False)
    cartpole.setState(initial_state)

    time=0
    states = [jnp.append(cartpole.getState(),time)]
    axis_names = ['Cart Location (m)','Cart Velocity (m/s)','Pole Angle (rad)','Pole Velocity (rad/s)','Time (s)']
    while time<end_time:
        cartpole.performAction(0.0)
        cartpole.remap_angle()
        time+=cartpole.delta_time
        states.append(jnp.append(cartpole.getState(),time))

    return jnp.stack(states), axis_names

def rollout_action(initial_state, action, end_time=10):
    cartpole = CartPole(visual=False)
    cartpole.setState(initial_state)

    time=0
    states = [jnp.append(cartpole.getState(),time)]
    axis_names = ['Cart Location (m)','Cart Velocity (m/s)','Pole Angle (rad)','Pole Velocity (rad/s)','Time (s)']
    while time<end_time:
        cartpole.performAction(action)
        cartpole.remap_angle()
        time+=cartpole.delta_time
        states.append(jnp.append(cartpole.getState(),time))
        
    return jnp.stack(states), axis_names

def plotGraph(state_array, horisontal_index, vertical_index, axis_names):
    plt.figure(figsize=(8, 4))
    plt.plot(state_array[:,horisontal_index], state_array[:,vertical_index], color='blue')

    plt.xlabel(axis_names[horisontal_index])
    plt.ylabel(axis_names[vertical_index])
    plt.title(f'Line Plot of {axis_names[vertical_index]} vs {axis_names[horisontal_index]}')
    plt.grid(True)
    plt.legend()

    plt.autoscale(enable=True, axis='both', tight=True)

    plt.show()

def plot_model_graph(state_array, true_state_array, horisontal_index, vertical_index, axis_names):
    plt.figure(figsize=(8, 4))

    plt.plot(state_array[:, horisontal_index], state_array[:, vertical_index], color='blue', label='Model Rollout')
    plt.plot(true_state_array[:, horisontal_index], true_state_array[:, vertical_index], color='red', label='True Rollout')

    plt.xlabel(axis_names[horisontal_index])
    plt.ylabel(axis_names[vertical_index])
    plt.title(f'Line Plot of {axis_names[vertical_index]} vs {axis_names[horisontal_index]}')
    plt.grid(True)
    plt.legend()  # This will now use the labels provided

    plt.autoscale(enable=True, axis='both', tight=True)
    plt.show()

def oneDimLinearScan(variable_1_index):
    cartpole = CartPole(visual=False)

    min_1 = -1
    max_1 = 1
    if variable_1_index==2:
        min_1 = -np.pi/2
        max_1 = np.pi/2

    initial_state = np.random.uniform(-1, 1, size=4)
    initial_state[2] *= np.pi/2
        
    linear_data = []
    axis_names = ['Cart Location (m)','Cart Velocity (m/s)','Pole Angle (rad)','Pole Velocity (rad/s)']
    axis_names.append('Initial '+ axis_names[variable_1_index])
    step_1 = (max_1-min_1)/40
    for variable_1 in np.arange(min_1,max_1+step_1,step_1):
        initial_state[variable_1_index] = variable_1
        cartpole.setState(initial_state)
        cartpole.performAction(0.0)
        Y = cartpole.getState() - initial_state
        linear_data.append(np.append(Y,variable_1))
    return np.array(linear_data), axis_names

def twoDimLinearScan(variable_1_index,variable_2_index):
    cartpole = CartPole(visual=False)

    min_1 = -1
    max_1 = 1
    min_2 = -1
    max_2 = 1
    if variable_1_index==2:
        min_1 = -np.pi/2
        max_1 = np.pi/2
    if variable_2_index==2:
        min_2 = -np.pi/2
        max_2 = np.pi/2

    initial_state = np.random.uniform(-1, 1, size=4)
    initial_state[2] *= np.pi/2
        
    linear_data = []
    axis_names = ['Cart Location','Cart Velocity','Pole Angle','Pole Velocity']
    axis_names.append('Initial '+ axis_names[variable_2_index])
    axis_names.append('Initial '+ axis_names[variable_1_index])
    step_1 = (max_1-min_1)/40
    step_2 = (max_2-min_2)/40
    for variable_1 in np.arange(min_1, max_1 + step_1, step_1):
        for variable_2 in np.arange(min_2,max_2+step_2,step_2):
            initial_state[variable_1_index] = variable_1
            initial_state[variable_2_index] = variable_2
            cartpole.setState(initial_state)
            cartpole.performAction(0.0)
            Y = cartpole.getState() - initial_state
            linear_data.append(np.append(Y,[variable_2,variable_1]))
    return np.array(linear_data), axis_names

def plotContours(state_array, horisontal_index, vertical_index, contour_indexes, axis_names):
    x = state_array[horisontal_index, :]
    y = state_array[vertical_index, :]
    for i, contour_index in enumerate(contour_indexes):
        z = state_array[contour_index, :]
        cs = plt.tricontour(x, y, z, levels=jnp.arange(jnp.min(z),jnp.max(z),(jnp.max(z)-jnp.min(z))/5), linewidths=2)
        #cs.collections[0].set_label(f'Col {axis_names[contour_index]} = {level}')

    plt.xlabel(axis_names[horisontal_index])
    plt.ylabel(axis_names[vertical_index])
    plt.title('Multiple Tricontours')
    plt.grid(True)
    plt.show()
