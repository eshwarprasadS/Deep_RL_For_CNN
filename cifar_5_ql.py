import gym_examples
import gym
import numpy as np
import stable_baselines3
# from stable_baselines3 import PPO
import sb3_contrib as sb3
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from gym.wrappers import FlattenObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from qlearner import qlearner
from qlearner.policy import*
from qlearner.graphing_utils import*

RUN_NAME = "cifar_5_qlearner"
CNN_SAVE_DIR = "logs/qlearner/cnn"
EVAL_SAVE_DIR = "logs/qlearner/cnn_eval"
CHECKPOINT_DIR = "logs/qlearner/checkpoints"
REPLAYMEM_DIR = "logs/qlearner/replaymem"
FIGURES_DIR = "logs/qlearner/figures"

print(sb3.__version__, gym.__version__, stable_baselines3.__version__)
env_id = "gym_examples/CNN-v0"

env = gym.make(env_id, dataset = 'cifar', run_name = 'train', cnn_save_dir = CNN_SAVE_DIR)

agent = qlearner.QLearner(
    env,
    lr=0.1,
    epsilion=0.5,
    q_path=REPLAYMEM_DIR,
    reply_memory_path=REPLAYMEM_DIR,
)

agent.run_experiment()
agent.save(CHECKPOINT_DIR)

eval_env = gym.make(env_id, dataset = 'cifar', run_name = 'eval', cnn_save_dir = EVAL_SAVE_DIR)
mean_r, std_r = eval_qlearner(Qtable = agent.Qtable, env=eval_env, n=50, epsilon=0.1) 

print("Mean reward: ", mean_r)
print("Std reward: ", std_r)

plot_mean_rewards(agent, FIGURES_DIR)
plot_mean_episode_length(agent, FIGURES_DIR)