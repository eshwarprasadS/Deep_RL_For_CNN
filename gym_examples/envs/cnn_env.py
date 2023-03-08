import gym
from gym import spaces
import pygame
import numpy as np

class StarWarsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):

        self.layer_depth_limit = 8
        self.layer_depth = 0
        self.observation_space = spaces.Dict(
            {
                "layer_type": spaces.Discrete(4), # conv, pool, fc, softmax
                "layer_depth": spaces.Discrete(8), # Current depth of network (8 layers max)
                "filter_depth": spaces.Discrete(5), # Used for conv (0, 64, 128, 256, 512) -- 0 is no filter
                "filter_size": spaces.Discrete(4), # Used for conv and pool (1,3,5) -- 0 is no filter
                "fc_size": spaces.Discrete(4) # Used for fc and softmax -- number of neurons in layer (512, 256, 128, 64)
                # "image_size": spaces.Discrete(4) # Used for any layer that maintains square input (conv and pool) returned in info dict
            }
                 )


        #action space is a dictionary of the same size as the observation space
        # self.action_space = spaces.Dict(
        #     {
        #         "layer_type": spaces.Discrete(4), # conv, pool, fc, softmax
        #         "layer_depth": spaces.Discrete(8), # Current depth of network (8 layers max)
        #         "filter_depth": spaces.Discrete(5), # Used for conv (64, 128, 256, 512) -- 0 is no filter
        #         "filter_size": spaces.Discrete(4), # Used for conv and pool (1,3,5) -- 0 is no filter
        #         "fc_size": spaces.Discrete(4), # Used for fc and softmax -- number of neurons in layer (512, 256, 128, 64)
        #         "terminal": spaces.Discrete(2) # 0 if not terminal, 1 if terminal
        #     }
        #          )
        self.action_space = spaces.MultiDiscrete([4, 8, 5, 4, 4, 2]) # 4, 8, 5, 4, 4, 2 corresponds to the number of discrete actions in each space

        self._discrete_to_layer_type = {    # Maps discrete action space to layer type
            0: "conv",
            1: "pool",
            2: "fc",
            3: "softmax"
        }

        self._discrete_to_filter_depth = {    # Maps discrete action space to filter depth
            0: 0,
            1: 10,
            2: 20,
            3: 32,
            4: 64
        }

        self._discrete_to_filter_size = {    # Maps discrete action space to filter size
            0: 0,
            1: 1,
            2: 3,
            3: 5
        }

        self._discrete_to_fc_size = {    # Maps discrete action space to fc size
            0: 512,
            1: 256,
            2: 128,
            3: 64
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
    
    def get_valid_action_mask(self):
        # # Disable actions that would move the agent off the grid
        # invalid_actions = []

        # create a mask representing valid actions in each dimension
        md_mask = tuple([np.ones(x,) for x in self.action_space.nvec])

        if self.layer_depth < self.layer_depth_limit-1:
            pass
        # # self.action_space.enable_actions()
        # if self._agent_location[1] == self.size - 1:
        #     invalid_actions.append(0)
        # if self._agent_location[0] == 0:
        #     invalid_actions.append(1)
        # if self._agent_location[1] == 0:
        #     invalid_actions.append(2)
        # if self._agent_location[0] == self.size - 1:
        #     invalid_actions.append(3)
        else:
            md_mask[5][0] = 0

        # mask = np.ones(self.action_space.n, dtype=np.int8)
        # mask[invalid_actions] = 0

        return md_mask

    def _get_obs(self):
        return {"r2d2": self._agent_location, "c3po": self._target_location, "vader": self._enemy_location}
    
    def _get_info(self):
        return {"c3po_distance": np.linalg.norm(self._agent_location - self._target_location, ord=1),
                "vader_distance": np.linalg.norm(self._agent_location - self._enemy_location, ord=1)}



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

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self.layer_depth+=1
        direction = self._action_to_direction[action]
        # clip to the grid for bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
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
