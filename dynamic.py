import gym
import numpy as np

class Dynamic(gym.spaces.Discrete):
    def __init__(self, max_space):
        self.n = max_space
        self.availabe_actions = list(range(0, max_space))

    def sample(self):
        return np.random.choice(self.availabe_actions)

    def contains(self, x):
        return x in self.availabe_actions
    
    def disable_actions(self, actions):
        for action in actions:
            self.availabe_actions.remove(action)

    def enable_actions(self):
        self.availabe_actions = list(range(0, self.n))
    
    def __repr__(self):
        return f"Dynamic(max_len = {self.n}, current_len = {len(self.availabe_actions)}, availabe_actions = {self.availabe_actions})"
    
    @property
    def shape(self):
        return (len(self.availabe_actions),)