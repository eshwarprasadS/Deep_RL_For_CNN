import json
import glob
import argparse
from qlearner import qlearner
from qlearner.policy import*
from qlearner.graphing_utils import*

from itertools import groupby

# import gym_examples
# import gym
# import numpy as np
# import stable_baselines3
# # from stable_baselines3 import PPO
# import sb3_contrib as sb3
# from sb3_contrib import MaskablePPO
# from sb3_contrib.common.wrappers import ActionMasker
# from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# from gym.wrappers import FlattenObservation
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from sb3_contrib.common.maskable.utils import get_action_masks
# from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.callbacks import ProgressBarCallback
# from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor

FIGURES_DIR = "logs\qlearner"


parser = argparse.ArgumentParser(
                    prog='get_top_5_train_models',
                    description='Top 5 train models for a specific run',
                    )

parser.add_argument('-p','--path', dest='path',type=str, help='path for directory containing cnn folder with json files')

args=parser.parse_intermixed_args()
if 'ppo' in args.path:
    FIGURES_DIR="logs\ppo"

print(args.path)

PATH=args.path

def print_model(trajectory):
    model, reward, nparams=trajectory
    for layer in model:
        # print('LAYER',layer)
        layer_type, layer_depth, filter_depth, filter_size, stride, image_size, fc_size, terminate, state_list = layer
        
        print('{}\t filter_depth:{} \tfilter_size:{},\t stride:{},\tfc_size:{}'.format(layer_type,filter_depth,filter_size,stride,fc_size))
    print("Accuracy:{}\tN_params:{}\n================".format(reward,nparams))

top_list=[]
ep_len_list=[]
ep_reward_list=[]

# frequency distribution of layer depths 1-8
freq_dist={i:0 for i in range(1,9)}
print(freq_dist)

avg_len=0
total=0


# for PATH in [QL_PATH, PPO_PATH]:
for filename in glob.glob(PATH+'/cnn/*.json'):
    with open(filename, 'r') as f:
        observation=json.load(f)
        network_tuples, accuracy, size=observation['network_tuples'], observation['accuracy'], observation['model_size']
        freq_dist[len(network_tuples)]+=1

        top_list.append((network_tuples, accuracy, size))
        ep_reward_list.append(accuracy)
        ep_len_list.append(len(network_tuples))
        
        avg_len+=len(network_tuples)
        total+=1

        # print(filename)
        # break

# print("LAST 20 MODELS:")
# for printstr in ['{}\n'.format(i) for i in top_list[-1:-21:-1]]:
#     print(printstr)

agent = qlearner.QLearner(
    None,
    lr=0.1,
    epsilion=0.5,
    q_path="",
    reply_memory_path="",
)

agent.ep_len=ep_len_list[500:]
agent.ep_rewards=ep_reward_list[500:]

top_list.sort(key=lambda tup: tup[1], reverse=True)
print('TOP 20 MODELS:')
for i in top_list[:20]:
    print_model(i)



print('\nAvg. length:', avg_len/total)

print('\nFreq dist:', freq_dist)


print('\n\n================TOP UNIQUE MODELS')
# https://stackoverflow.com/questions/31499259/making-a-sequence-of-tuples-unique-by-a-specific-element
def unique_by_key(elements, key=None):
    if key is None:
        key = lambda e: e
    return {key(el): el for el in elements}.values()

"""
#  assuming I'll get at least 5 unique models from top 20
unique_models=list(unique_by_key(top_list[:20], key=lambda x:x[2]))
# print(unique_models[0])
unique_models=sorted(unique_models,key=lambda x:x[1], reverse=True)
for i in unique_models:
    print_model(i)
"""

top_models = {}

for i in top_list[:20]:
    model, reward, nparams = i
    if nparams not in top_models:
        top_models[nparams] = i
    else:
        ex_m, ex_r, ex_np = top_models[nparams]
        if ex_r < reward:
            top_models[nparams] = i

list_top_models = []

for nparams in top_models:
    list_top_models.append(top_models[nparams])

print("Top performing models are:")
list_top_models.sort(key = lambda tup : tup[1], reverse=True)
for i in list_top_models:
    print_model(i)


plot_mean_rewards(agent, FIGURES_DIR)
plot_mean_episode_length(agent, FIGURES_DIR)
plot_raw_episode_rewards(agent, FIGURES_DIR)


