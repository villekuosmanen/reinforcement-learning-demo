import gymnasium as gym
import matplotlib.pyplot as plt
import time

# constants
num_steps = 1500

env = gym.make('MountainCar-v0', render_mode="human")
obs = env.reset()

for step in range(num_steps):
    # take random action, but you can also do something more intelligent
    # action = my_intelligent_agent_fn(obs) 
    random_action = env.action_space.sample()
    
    # apply the action
    observation, reward, done, truncated, info = env.step(random_action)
    
    # Render the env
    env.render()

    # Wait a bit before the next frame unless you want to see a crazy fast video
    time.sleep(0.001)
    
    # If the epsiode is up, then start another one
    if done or truncated:
        env.reset()

# Close the env
env.close()