from qlearner import qlearner
import gym_examples
import gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env


def test():
    env = gym.make("gym_examples/CNN-v0")
    agent = qlearner.QLearner(
        env,
        lr=0.1,
        epsilion=0.5,
        q_path="qleaner_checkpoints",
        reply_memory_path="qleaner_checkpoints",
    )
    agent.train(100)
    print(agent.Qtable)


if __name__ == "__main__":
    test()
