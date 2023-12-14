import numpy as np

# Initialization of parameters
env_L = 10
env_C = 10
epsilon = float(0.8)
Episodes = 10000
actions = [0, 1, 2, 3]
q_values = np.zeros((env_L, env_C, len(actions)))

# Define the environment and reward matrix
rec = np.ones((10, 10))
# Reward Matrix:
rec[:, :] = -1  
rec[:, 9] = rec[0:6, 0] = rec[2, 4:9] = rec[5, 1:5] = rec[3:4, 4] = -10
rec[0, 5] = 5
rec[9, 5] = 100

# Functions Used

def final(i, j):
    # Check if the state is a final state (has a reward)
    if rec[i, j] == -1.:
        return False
    else:
        return True

def start():
    # Randomly initialize a non-final state
    i = np.random.randint(env_L)
    j = np.random.randint(env_C)
    while final(i, j):
        i = np.random.randint(env_L)
        j = np.random.randint(env_C)
    return i, j

def new_action(i, j, epsilon):
    # Choose a new action using epsilon-greedy policy
    if np.random.random() < epsilon:
        return np.argmax(q_values[i, j])  # Exploitation
    else:
        return np.random.randint(4)  # Exploration

def new_location(i, j, indice_action):
    # Update the location based on the chosen action
    # 0: right, 1: left, 2: down, 3: up

    if indice_action == 0 and j < env_C - 1:
        j += 1
    elif indice_action == 1 and j > 0:
        j -= 1
    elif indice_action == 2 and i < env_L - 1:
        i += 1
    elif indice_action == 3 and i > 0:
        i -= 1

    return i, j

def shortest_path(start_L, start_C):
    # Find the shortest path from a starting state to a final state
    if final(start_L, start_C):
        return []
    else:
        shortest = []
        shortest.append((start_L, start_C))

        while not final(start_L, start_C):
            action_index = new_action(start_L, start_C, 1.)

            start_L, start_C = new_location(start_L, start_C, action_index)
            shortest.append((start_L, start_C))
        return shortest

# Training

epsilon = 0.8
DF = 0.9
learning_rate = 0.9

# Training loop
for episode in range(10000):
    i, j = start()

    # Update Q-values until a final state is reached
    while not final(i, j):
        action_index = new_action(i, j, epsilon)
        old_i, old_j = i, j
        i, j = new_location(i, j, action_index)

        reward = rec[i, j]
        old_value = q_values[old_i, old_j, action_index]
        temporal_difference = reward + (DF * np.max(q_values[i, j])) - old_value

        new_q_value = old_value + (learning_rate * temporal_difference)
        q_values[old_i, old_j, action_index] = new_q_value

print("Training done")

print(shortest_path(1, 1))
