import gym
from gym import spaces
import pygame
import numpy as np
# from gym.spaces.dynamic import Dynamic
from gym_examples.envs.pytorch_parser.pytorch_parser_function import generate_and_train
from torchvision import datasets
from torchvision.transforms import ToTensor

class StarWarsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
  
        self.train_data = datasets.MNIST(
                            root = 'data',
                            train = True,                         
                            transform = ToTensor(), 
                            download = True,            
                            )
        self.test_data = datasets.MNIST(
                            root = 'data', 
                            train = False, 
                            transform = ToTensor()
                            )
        
        train_idx = (self.train_data.targets==0) | (self.train_data.targets==1) | (self.train_data.targets==2)
        test_idx = (self.test_data.targets==0) | (self.test_data.targets==1) | (self.test_data.targets==2)

        self.train_data.data = self.train_data.data[train_idx]
        self.train_data.targets = self.train_data.targets[train_idx]
        self.test_data.data = self.test_data.data[test_idx]
        self.test_data.targets = self.test_data.targets[test_idx]

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "r2d2": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "c3po": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "vader": spaces.Box(0, size -1, shape=(2,), dtype=int)
            }
        )

        # We have 4 actions, corresponding to 0: "right", 1: "up", 2: "left", 3: "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([-1, 0]),
            2: np.array([0, -1]),
            3: np.array([1, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        return {"r2d2": self._agent_location, "c3po": self._target_location, "vader": self._enemy_location}
    
    def _get_info(self):
        return {"c3po_distance": np.linalg.norm(self._agent_location - self._target_location, ord=1),
                "vader_distance": np.linalg.norm(self._agent_location - self._enemy_location, ord=1)}

    def get_valid_action_mask(self):
        # Disable actions that would move the agent off the grid
        invalid_actions = []
        # self.action_space.enable_actions()
        if self._agent_location[1] == self.size - 1:
            invalid_actions.append(0)
        if self._agent_location[0] == 0:
            invalid_actions.append(1)
        if self._agent_location[1] == 0:
            invalid_actions.append(2)
        if self._agent_location[0] == self.size - 1:
            invalid_actions.append(3)
        
        mask = np.ones(self.action_space.n, dtype=np.int8)
        mask[invalid_actions] = 0

        return mask
        # self.action_space.disable_actions(actions)
        # return self.action_space.available_actions

    def reset(self, seed=None, options=None):
        # Seed self.np_random
        super().reset(seed=seed)

        # Choose r2d2 location randomly at the beginning
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # sample c3po's location randomly until it does not coincide with the r2d2's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        # sample vader's location randomly until it does not coincide with either of the previous locs
        self._enemy_location = self._agent_location
        while np.array_equal(self._enemy_location, self._agent_location) or np.array_equal(self._enemy_location, self._target_location):
            self._enemy_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        #adjust action space according to r2d2's location
        # self.adjust_action_space()    

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
    
        direction = self._action_to_direction[action]
        # clip to the grid for bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        #adjust action space according to r2d2's location
        # self.adjust_action_space()
        terminated = False
        # An episode is done if r2d2 found c3po or effing DIED to vader
        won = np.array_equal(self._agent_location, self._target_location)
        died = np.array_equal(self._agent_location, self._enemy_location)
        if won:
            terminated = True
            reward = 1 
        elif died:
            terminated = True
            reward = -1
        else:
            reward = -0.005

        if terminated:
            acc = generate_and_train([('conv',1,5,3,1,28,0,0,[]),
                                      ('pool',1,0,2,2,28,0,0,[]),
                                      ('conv',1,10,3,1,26,0,0,[]),
                                      ('pool',1,0,2,2,28,0,0,[]),
                                      ('fc',0,0,0,0,0,100,0,[]),
                                      ('fc',0,0,0,0,0,10,0,[])], 
                                      self.train_data, self.test_data)
            
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        #Draw c3po
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        #Draw r2d2
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        #Draw vader
        pygame.draw.rect(
            canvas,
            (0, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
