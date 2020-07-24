import gym
import numpy as np
import torch
import sys
from matplotlib import pyplot as plt

sys.path.insert(0, "../../agents")

from q_learning import QLearning  # pylint: disable=import-error

EPISODES = 5000

if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    agent = QLearning(env, epsilon=0.8, gamma=0.5, lr=0.01)
    
    episode_rew = []
    for episode in range(EPISODES):
        # Deciding first action
        action = env.action_space.sample()
        state = env.reset()
        ep_rew = 0
        while True:
            next_state, reward, done, _ = env.step(action)
            # env.render()
            ep_rew += reward

            agent.update((state, action, reward, next_state))
            state = next_state
            agent.get_action(state)

            if done:
                episode_rew.append(ep_rew)
                break
    env.close()

    plt.plot(episode_rew)
    plt.show()

    state = env.reset()
    while True:
        action = agent.get_action(state, explore=False)
        next_state, reward, done, _ = env.step(action)
        env.render()
        state = next_state
        if done:
            break