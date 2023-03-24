"""
Qlearner referred from https://github.com/bowenbaker/metaqnn/tree/a25847f635e9545455f83405453e740646038f7a/libs/grammar
"""
import pandas as pd


class QLearner:
    """All Q-Learning updates and policy generator
    Args
        state: The starting state for the QLearning Agent
        q_values: A dictionary of q_values --
                        keys: State tuples (State.as_tuple())
                        values: [state list, qvalue list]
        replay_dictionary: A pandas dataframe with columns: 'net' for net strings, and 'accuracy_best_val' for best accuracy
                                    and 'accuracy_last_val' for last accuracy achieved
        output_number : number of output neurons
    """

    def __init__(
        self,
        state_space_parameters,
        epsilon,
        state=None,
        qstore=None,
        replay_dictionary=pd.DataFrame(
            columns=[
                "net",
                "accuracy_best_val",
                "accuracy_last_val",
                "accuracy_best_test",
                "accuracy_last_test",
                "ix_q_value_update",
                "epsilon",
            ]
        ),
    ):
        self.state_list = []

        self.state_space_parameters = state_space_parameters

        # Class that will expand states for us
        self.enum = se.StateEnumerator(state_space_parameters)
        self.stringutils = StateStringUtils(state_space_parameters)

        # Starting State
        self.state = (
            se.State("start", 0, 1, 0, 0, state_space_parameters.image_size, 0, 0)
            if not state
            else state
        )
        self.bucketed_state = self.enum.bucket_state(self.state)

        # Cached Q-Values -- used for q learning update and transition
        self.qstore = QValues() if not qstore else qstore
        self.replay_dictionary = replay_dictionary

        self.epsilon = epsilon  # epsilon: parameter for epsilon greedy strategy

    def update_replay_database(self, new_replay_dic):
        self.replay_dictionary = new_replay_dic

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

    def save_q(self, q_path):
        self.qstore.save_to_csv(os.path.join(q_path, "q_values.csv"))

    def _reset_for_new_walk(self):
        """Reset the state for a new random walk"""
        # Architecture String
        self.state_list = []

        # Starting State
        self.state = se.State(
            "start", 0, 1, 0, 0, self.state_space_parameters.image_size, 0, 0
        )
        self.bucketed_state = self.enum.bucket_state(self.state)

    def _run_agent(self):
        """Have Q-Learning agent sample current policy to generate a network"""
        while self.state.terminate == 0:
            self._transition_q_learning()

        return self.state_list

    def _transition_q_learning(self):
        """Updates self.state according to an epsilon-greedy strategy"""
        if self.bucketed_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(self.bucketed_state, self.qstore.q)

        action_values = self.qstore.q[self.bucketed_state.as_tuple()]
        # epsilon greedy choice
        if np.random.random() < self.epsilon:
            action = se.State(
                state_list=action_values["actions"][
                    np.random.randint(len(action_values["actions"]))
                ]
            )
        else:
            max_q_value = max(action_values["utilities"])
            max_q_indexes = [
                i
                for i in range(len(action_values["actions"]))
                if action_values["utilities"][i] == max_q_value
            ]
            max_actions = [action_values["actions"][i] for i in max_q_indexes]
            action = se.State(
                state_list=max_actions[np.random.randint(len(max_actions))]
            )

        self.state = self.enum.state_action_transition(self.state, action)
        self.bucketed_state = self.enum.bucket_state(self.state)

        self._post_transition_updates()

    def _post_transition_updates(self):
        # State to go in state list
        bucketed_state = self.bucketed_state.copy()

        self.state_list.append(bucketed_state)

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

    def accuracy_to_reward(self, acc):
        """How to define reward from accuracy"""
        return acc

    def update_q_value_sequence(self, states, termination_reward):
        """Update all Q-Values for a sequence."""
        self._update_q_value(states[-2], states[-1], termination_reward)
        for i in reversed(range(len(states) - 2)):
            self._update_q_value(states[i], states[i + 1], 0)

    def _update_q_value(self, start_state, to_state, reward):
        """Update a single Q-Value for start_state given the state we transitioned to and the reward."""
        if start_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(start_state, self.qstore.q)
        if to_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(to_state, self.qstore.q)

        actions = self.qstore.q[start_state.as_tuple()]["actions"]
        values = self.qstore.q[start_state.as_tuple()]["utilities"]

        max_over_next_states = (
            max(self.qstore.q[to_state.as_tuple()]["utilities"])
            if to_state.terminate != 1
            else 0
        )

        action_between_states = self.enum.transition_to_action(
            start_state, to_state
        ).as_tuple()

        # Q_Learning update rule
        values[actions.index(action_between_states)] = values[
            actions.index(action_between_states)
        ] + self.state_space_parameters.learning_rate * (
            reward
            + self.state_space_parameters.discount_factor * max_over_next_states
            - values[actions.index(action_between_states)]
        )

        self.qstore.q[start_state.as_tuple()] = {
            "actions": actions,
            "utilities": values,
        }
