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

RUN_NAME = "cifar_5_ppo"
CNN_SAVE_DIR = "logs/ppo/cnn"
CHECKPOINT_DIR = "logs/ppo/checkpoints"
TENSORBOARD_DIR = "logs/ppo/tensorboard"

print(sb3.__version__, gym.__version__, stable_baselines3.__version__)
env_id = "gym_examples/CNN-v0"

env = gym.make(env_id, dataset = 'cifar', run_name = 'test', cnn_save_dir = CNN_SAVE_DIR)

# Save a checkpoint every 20 steps
checkpoint_callback = CheckpointCallback(
  save_freq=20,
  save_path=CHECKPOINT_DIR,
  name_prefix="ppo_model"
)

model = MaskablePPO(MaskableActorCriticPolicy, FlattenObservation(env), verbose = 1, 
                    tensorboard_log=TENSORBOARD_DIR,
                    n_steps=32)

model.learn(total_timesteps=7000, callback=checkpoint_callback, progress_bar=True)