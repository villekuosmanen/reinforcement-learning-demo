from itertools import count

import gymnasium as gym
import torch
import torch.optim as optim

from dqn.model import DQN
from model.car import Car

# TODO this should not be needed
LR = 1e-4

# Initialise pytorch
# apple silicon was very slow for some reason
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

env = gym.make('MountainCar-v0', render_mode="human")# Get number of actions from gym action space

n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# Initialise deep Q network
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)


# Load pre-trained model
print("loading model car...")
model_car = Car(device, policy_net, target_net, optimizer).from_file("good.pth")
print("model car loaded.")

# Run the simulation with a given policy
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
for t in count():
    # apply the action
    action = model_car.select_action(state, env)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated

    if terminated:
            next_state = None
    else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    # Move to the next state
    state = next_state
    
    # If the epsiode is up, then start another one
    if done or truncated:
        env.reset()
        break

# Close the env
env.close()