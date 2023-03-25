"""
Qlearner referred from https://github.com/bowenbaker/metaqnn/tree/a25847f635e9545455f83405453e740646038f7a/libs/grammar
"""
from .policy import *

from collections import defaultdict
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm


class QLearner:
    """All Q-Learning updates and policy generator"""

    def __init__(self, env, lr, epsilion, q_path, reply_memory_path, state=None):
        """
        q_path: path to the directory where the q table needs to be stored
        replay_memory_path: path to the directory where the replay memory needs to be stored
        """
        self.env = env
        self.state = state
        self.lr = lr
        self.epsilon = epsilion

        self.q_path = q_path
        self.replay_memory_path = reply_memory_path

        self.q_table_file_path = self.q_path + "_qtable_" + str(self.epsilon) + ".pkl"
        self.replay_memory_file_path = (
            self.replay_memory_path + "_rep_mem_" + str(self.epsilon) + ".pkl"
        )

        self.initialize_q_table()
        self.initialize_replay_memory()

    def initialize_q_table(self, enforce_new=False):
        if enforce_new:
            self._new_q_table()
            return

        try:
            self._load_q_table()
        except:
            self._new_q_table()
        return

    def _new_q_table(self):
        self.Qtable = defaultdict(lambda: defaultdict(int))

    def _load_q_table(self):
        with open(self.q_table_file_path, "rb") as f:
            self.Qtable = pickle.load(f)

    def initialize_replay_memory(self, enforce_new=False):
        if enforce_new:
            self._new_replay_memory()
            return

        try:
            self._load_replay_memory()
        except:
            self._new_replay_memory()
        return

    def _new_replay_memory(self):
        self.replay_memory = []

    def _load_replay_memory(self, path):
        with open(self.replay_memory_file_path, "rb") as f:
            self.replay_memory = pickle.load(f)

    def generate_net(self):
        # Have Q-Learning agent sample current policy to generate a network and convert network to string format
        self._reset_for_new_walk()
        state_list = self._run_agent()
        state_list = self.stringutils.add_drop_out_states(state_list)
        net_string = self.stringutils.state_list_to_string(state_list)

        # Check if we have already trained this model
        if net_string in self.replay_dictionary["net"].values:
            acc_best_val = self.replay_dictionary[
                self.replay_dictionary["net"] == net_string
            ]["accuracy_best_val"].values[0]
            iter_best_val = self.replay_dictionary[
                self.replay_dictionary["net"] == net_string
            ]["iter_best_val"].values[0]
            acc_last_val = self.replay_dictionary[
                self.replay_dictionary["net"] == net_string
            ]["accuracy_last_val"].values[0]
            iter_last_val = self.replay_dictionary[
                self.replay_dictionary["net"] == net_string
            ]["iter_last_val"].values[0]
            acc_best_test = self.replay_dictionary[
                self.replay_dictionary["net"] == net_string
            ]["accuracy_best_test"].values[0]
            acc_last_test = self.replay_dictionary[
                self.replay_dictionary["net"] == net_string
            ]["accuracy_last_test"].values[0]
            machine_run_on = self.replay_dictionary[
                self.replay_dictionary["net"] == net_string
            ]["machine_run_on"].values[0]
        else:
            acc_best_val = -1.0
            iter_best_val = -1.0
            acc_last_val = -1.0
            iter_last_val = -1.0
            acc_best_test = -1.0
            acc_last_test = -1.0
            machine_run_on = -1.0

        return (
            net_string,
            acc_best_val,
            iter_best_val,
            acc_last_val,
            iter_last_val,
            acc_best_test,
            acc_last_test,
            machine_run_on,
        )

    def sample_replay_for_update(self):
        # Experience replay to update Q-Values
        for i in range(self.state_space_parameters.replay_number):
            net = np.random.choice(self.replay_dictionary["net"])
            accuracy_best_val = self.replay_dictionary[
                self.replay_dictionary["net"] == net
            ]["accuracy_best_val"].values[0]
            accuracy_last_val = self.replay_dictionary[
                self.replay_dictionary["net"] == net
            ]["accuracy_last_val"].values[0]
            state_list = self.stringutils.convert_model_string_to_states(
                cnn.parse("net", net)
            )

            state_list = self.stringutils.remove_drop_out_states(state_list)

            # Convert States so they are bucketed
            state_list = [self.enum.bucket_state(state) for state in state_list]

            self.update_q_value_sequence(
                state_list, self.accuracy_to_reward(accuracy_best_val)
            )

    def update_q_values(self, state_list, action_list, reward):
        last_state, last_action = state_list[-1], action_list[-1]

        self.Qtable[last_state][last_action] = (1 - self.lr) * self.Qtable[last_state][
            last_action
        ] + self.lr * reward

        for i in range(len(state_list) - 2, -1, -1):
            state, action = state_list[i], action_list[i]
            next_state = state_list[i + 1]
            a_max, u_max = greedy_policy(self.Qtable, self.env, next_state)

            self.Qtable[state][action] = (1 - self.lr) * self.Qtable[state][
                action
            ] + self.lr * u_max

    def train(self, n_training_episodes, replay_memory_size=10, write_to_disk=10):
        for episode in tqdm(range(n_training_episodes)):

            state, state_info = self.env.reset()
            state = tuple(state_info["obs_vector"])

            terminated = False
            state_list = [state]
            action_list = []

            while not terminated:
                action = epsilon_greedy_policy(
                    self.Qtable, self.env, state, self.epsilon
                )
                action_list.append(action)
                action = np.array(action, dtype=np.int64)
                _, reward, terminated, _, state_info = self.env.step(action)
                state = tuple(state_info["obs_vector"])
                if not terminated:
                    state_list.append(state)

            self.replay_memory.append((state_list, action_list, reward))

            # TODO: min(len(replay_memory), replay_mem_Size)
            # Comments: I think if it was min they would've written it as min in the paper
            for _ in range(replay_memory_size):
                state_list, action_list, reward = self.replay_memory[
                    np.random.choice(len(self.replay_memory))
                ]
                self.update_q_values(state_list, action_list, reward)

            if not (episode % write_to_disk):
                self.write_qtable_and_replay_memory()

        return

    def write_qtable_and_replay_memory(self):
        if not self.q_path or not self.replay_memory_path:
            raise Exception("Need Q Path and Replay Memory Path to write to disk")
        with open(self.q_table_file_path, "wb") as f:
            pickle.dump(self.Qtable, f)

        with open(self.replay_memory_file_path, "wb") as f:
            pickle.dump(self.replay_memory, f)

    def run_experiment(self, esp_schedule=None, reinitialize_qtable=True):
        # Start fresh when running experiment
        self.initialize_q_table(enforce_new=True)
        self.initialize_replay_memory(enforce_new=True)

        if not esp_schedule:
            esp_schedule = {
                1.0: 15,
                0.9: 15,
                0.8: 15,
                0.7: 15,
                0.6: 15,
                0.5: 15,
                0.4: 15,
                0.3: 15,
                0.2: 15,
                0.1: 15,
            }
        # Initialize new Qtable per experiment
        for epsilion, n_training_eps in esp_schedule:
            if reinitialize_qtable:
                self.initialize_q_table(enforce_new=True)
                self.initialize_replay_memory(enforce_new=True)
            self.epsilon = epsilion
            self.train(n_training_eps, epsilon=epsilion)

        # After the experiement has been run, we'll have multiple Qtables,
        # i.e best network for each epsilion with us
        # TODO: Generate best networks and get best acc?
