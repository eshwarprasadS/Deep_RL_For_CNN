import random


def greedy_policy(Qtable, env, state):
    if state not in Qtable:
        action = tuple(env.action_space.sample(env.get_valid_action_mask()))
        Qtable[state][action] = 0
        return action, 0

    all_actions = Qtable[state]
    best_action, best_util = [], 0
    for i, (action, util) in enumerate(all_actions.items()):
        if i == 0:
            best_action, best_util = action, util
        else:
            if util > best_util:
                best_action, best_util = action, util

    return best_action, best_util


def epsilon_greedy_policy(Qtable, env, state, epsilon):
    random_num = random.uniform(0, 1)
    if random_num > epsilon:
        action, _ = greedy_policy(Qtable, env, state)
    else:
        action = tuple(env.action_space.sample(env.get_valid_action_mask()))
        Qtable[state][action] = 0

    return action
