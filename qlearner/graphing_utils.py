import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from policy import *

def running_average(arr):
    new_arr = []
    avg = 0.0
    for i, val in enumerate(arr):
        avg = (avg*(i)+val) / (i+1) 
        new_arr.append(avg)
    return np.array(new_arr)

def smooth(scalars, weight=0.9):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def plot_mean_rewards(agent, figures_dir):
    x = [(i+1)*np.mean(agent.ep_len) for i in np.arange(len(agent.ep_rewards))]
    y = running_average(agent.ep_rewards)

    plt.plot(x, y, alpha=0.3)
    plt.plot(x, smooth(y, .9))
    plt.xlabel('iterations')
    plt.ylabel('mean episode reward')
    plt.title('mean episode reward')
    plt.legend(["mean ep rewards", "mean ep rewards smoothed"])
    plt.savefig(os.path.join(figures_dir,f'qlearner_{len(agent.ep_rewards)}_mean_rewards.pdf'))

def plot_mean_episode_length(agent, figures_dir):
    x = [(i+1)*np.mean(agent.ep_len) for i in np.arange(len(agent.ep_rewards))]
    y = running_average(agent.ep_len)

    plt.plot(x, y, alpha=0.3)
    plt.plot(x, smooth(y, .9))
    plt.xlabel('iterations')
    plt.ylabel('mean episode length')
    plt.title('mean episode length')
    plt.legend(["mean ep length", "mean ep length smoothed"])
    plt.savefig(os.path.join(figures_dir,f'qlearner_{len(agent.ep_rewards)}_mean_episode_length.pdf'))

def eval_qlearner(Qtable, env, n, epsilon):

    episode_rewards = []
    for episode in tqdm(range(n)):

        state, state_info = env.reset()
        state = tuple(state_info["obs_vector"])

        terminated = False

        while not terminated:
            action = epsilon_greedy_policy(
                Qtable, env, state, epsilon
            )
            action = np.array(action, dtype=np.int64)
            _, reward, terminated, _, state_info = env.step(action)
            state = tuple(state_info["obs_vector"])
        
        if terminated:
            episode_rewards.append(reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward