from itertools import count

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from model.car import Car
from dqn.model import DQN
from dqn.utils import ReplayMemory

# constants
NUM_STEPS = 1500
LR = 1e-4

# Initialise pytorch
# apple silicon was very slow for some reason
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

env = gym.make('MountainCar-v0')

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

print("The observation space: {}".format(env.observation_space))
print("The action space: {}".format(env.action_space))

# Initialise deep Q network
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100000)

# Initialise model class
model_car = Car(device, policy_net, target_net, optimizer, memory)

episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

num_episodes = 1000

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state

    # for last few episodes, show in human format
    # if i_episode > 90:
    #     env = gym.make('MountainCar-v0', render_mode="human")

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = model_car.select_action(state, env)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        model_car.optimize_model()

        model_car.update_weights()    

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

env.close()
print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

# -------------------------------------------------
# Run the simulation with a given policy
env = gym.make('MountainCar-v0', render_mode="human")
state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
for step in range(NUM_STEPS):
    # apply the action
    action = model_car.select_action(state, env)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated

    if terminated:
            next_state = None
    else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    # Store the transition in memory
    memory.push(state, action, next_state, reward)

    # Move to the next state
    state = next_state
    
    # If the epsiode is up, then start another one
    if done or truncated:
        env.reset()
        break

# Close the env
env.close()