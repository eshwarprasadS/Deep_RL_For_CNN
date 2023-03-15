import gym
from gym import spaces
import pygame
import numpy as np
import math
from gym_examples.envs.pytorch_parser.pytorch_parser_function import generate_and_train




class CNNEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):

        self.layer_depth_limit = 7
        self.layer_depth = 0
        self.max_image_size_for_fc = 28
        self.current_image_size = 28 # change this after each action

        self.is_start_state = True #change this to False once we take the first action

        self.allow_consecutive_pooling = False
        self.allow_initial_pooling = False

        self.max_fc_layers_allowed = 2
        self.cur_num_fc_layers = 0 

        self.current_state = [] #array of dictionaries

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

        self._layer_type_to_discrete = {    # Maps discrete action space to layer type
            "conv": 0,
            "pool": 1,
            "fc": 2,
            "softmax": 3
        }

        self._discrete_to_filter_depth = {    # Maps discrete action space to filter depth
            0: 0,
            1: 10,
            2: 20,
            3: 32,
            4: 64
        }

        self._filter_depth_to_discrete = {    # Maps discrete action space to filter depth
            0: 0,
            10: 1,
            20: 2,
            32: 3,
            64: 4
        }

        self._discrete_to_filter_size = {    # Maps discrete action space to filter size
            0: 0,
            1: 1,
            2: 3,
            3: 5
        }
        self._filter_size_to_discrete = {    # Maps discrete action space to filter size   
            0: 0,
            1: 1,
            3: 2,
            5: 3
        }
        self._discrete_to_fc_size = {    # Maps discrete action space to fc size
            0: 512,
            1: 256,
            2: 128,
            3: 64
        }
        self._fc_size_to_discrete = {    # Maps discrete action space to fc size
            512: 0,
            256: 1,
            128: 2,
            64: 3
        }
        self._state_elem_to_index = {
            "layer_type": 0,
            "layer_depth": 1,
            "filter_depth": 2,
            "filter_size": 3,
            "fc_size": 4,
            "terminal": 5
        }

        # initiate a conv layer with filter size 5 and 10 filters for all envs
        # TODO: change this to a random conv layer, or random layer type or decide on how to initiate a dummy layer
        self.current_image_size = self._calculate_image_size(self.current_image_size, 
                                                                 5, 1)
        self.current_state.append({

            "layer_type": self._discrete_to_layer_type[0], # conv 
            "layer_depth": self.layer_depth, # depth 0
            "filter_depth": self._discrete_to_filter_depth[1], # 10 filters
            "filter_size": self._discrete_to_filter_size[3], # filter size 5
            "fc_size": self._discrete_to_fc_size[0], # any, doesn't matter for conv
            "image_size": self.current_image_size
        })

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

        '''
        Assuming current state looks like this-
         spaces.Dict(
            {
                "layer_type": spaces.Discrete(4), # conv, pool, fc, softmax
                "layer_depth": spaces.Discrete(8), # Current depth of network (8 layers max)
                "filter_depth": spaces.Discrete(5), # Used for conv (64, 128, 256, 512) -- 0 is no filter
                "filter_size": spaces.Discrete(4), # Used for conv and pool (1,3,5) -- 0 is no filter
                "fc_size": spaces.Discrete(4), # Used for fc and softmax -- number of neurons in layer (512, 256, 128, 64)
                "terminal": spaces.Discrete(2) # 0 if not terminal, 1 if terminal
            }
                 )
        '''
        # # Disable actions that would move the agent off the grid
        # invalid_actions = []

        # create a mask representing valid actions in each dimension
        #start with all actions disabled
        md_mask = tuple([np.zeros(x,) for x in self.action_space.nvec])

        if self.layer_depth < self.layer_depth_limit:
            print('inside layer depth limit')
            #enable valid convolutions
            if self.is_start_state or \
                (self.is_start_state == False and self.current_state[-1]["layer_type"] in \
                set(["conv", "pool"])):

                print('inside conv or pool')
                for d in self._discrete_to_filter_depth:
                    for f in self._valid_filter_sizes_for_image():
                        self._enable_convolution(md_mask, f, d)
                         

            #TODO: add global average pooling??


            #enable valid pooling layers
            if (self.is_start_state == False and \
                (self.current_state[-1]["layer_type"] == "conv" or 
                (self.current_state[-1]["layer_type"] == "pool" and self.allow_consecutive_pooling))) or \
                (self.is_start_state and self.allow_initial_pooling): 

                for f in self._valid_filter_sizes_for_image():
                    self._enable_pooling(md_mask, f)
                    


            #enable valid fully connected layers
            if (self.is_start_state == False and self._allow_fully_connected() and 
            self.current_state[-1]["layer_type"] in ['start', 'conv', 'pool']):
                
                for fc_sz in self._valid_fc_sizes_for_image():            
                    self._enable_fc(md_mask, fc_sz)
                    
            #enable all FC layers if we are transitioning from conv or pool layers
            if (self.is_start_state == False and self._allow_fully_connected() and
            self.current_state[-1]["layer_type"] in ['conv', 'pool']):
                
                for fc_sz in self._discrete_to_fc_size:            
                    self._enable_fc(md_mask, fc_sz)
            
            #fc to fc transitions
            if self.is_start_state == False and self.current_state[-1]["layer_type"] == 'fc' \
            and self.cur_num_fc_layers < self.max_fc_layers_allowed - 1:
                
                for fc_sz in self._valid_fc_sizes_for_image():
                    self.cur_num_fc_layers += 1
                    self._enable_fc(md_mask, fc_sz)    

            # Enable terminal if its an FC layer
            # TODO: Do we add terminal from other layers as well?
            if self.is_start_state == False and self.current_state[-1]["layer_type"] == 'fc':
                md_mask[self._state_elem_to_index["terminal"]][1] = 1

            #enable only layer_depth+1 action if layer depth limit is not reached
            md_mask[self._state_elem_to_index["layer_depth"]][self.layer_depth+1] = 1
        else:
            #enable only terminate action if layer depth limit is reached
            md_mask[self._state_elem_to_index["terminal"]][1] = 1

        # mask = np.ones(self.action_space.n, dtype=np.int8)
        # mask[invalid_actions] = 0

        return md_mask

    def _enable_convolution(self, md_mask, f, d):
        # print('inside enable conv')
        md_mask[self._state_elem_to_index["layer_type"]][self._layer_type_to_discrete["conv"]] = 1
        md_mask[self._state_elem_to_index["filter_depth"]][d] = 1
        md_mask[self._state_elem_to_index["filter_size"]][f] = 1

    def _enable_pooling(self, md_mask, f):
        md_mask[self._state_elem_to_index["layer_type"]][self._layer_type_to_discrete["pool"]] = 1
        md_mask[self._state_elem_to_index["filter_size"]][f] = 1  

    def _enable_fc(self, md_mask, fc_sz):
        md_mask[self._state_elem_to_index["layer_type"]][self._layer_type_to_discrete["fc"]] = 1
        md_mask[self._state_elem_to_index["fc_size"]][fc_sz] = 1      

    
    def _allow_fully_connected(self):
        return self.current_image_size <= self.max_image_size_for_fc
        
    def _valid_filter_sizes_for_image(self):
        return [c for c in self._discrete_to_filter_size if self._discrete_to_filter_size[c] < self.current_image_size]

    def _valid_fc_sizes_for_image(self):
        if self.current_state[-1]["layer_type"] == 'fc':
            return [fc for fc in self._discrete_to_fc_size if self._discrete_to_fc_size[fc] < self.current_state[-1]["fc_size"]]
        
        return self._discrete_to_layer_type.keys()

    def _calculate_image_size(self, image_size, filter_size, stride):
        new_size = int(math.ceil(float(image_size - filter_size + 1) / float(stride)))
        return new_size    
    
    def _get_obs(self):
        return {"layer_type": self._layer_type_to_discrete[self.current_state[-1]["layer_type"]],
                "layer_depth": self.current_state[-1]["layer_depth"], 
                "filter_depth": self._filter_depth_to_discrete[self.current_state[-1]["filter_depth"]],
                "filter_size": self._filter_size_to_discrete[self.current_state[-1]["filter_size"]], 
                "fc_size": self._fc_size_to_discrete[self.current_state[-1]["fc_size"]]
                }
        # return {"current_state": self.current_state}
    
    def _get_info(self):
        return {
                "current_network" : self.current_state,
                "current_image_size": self.current_image_size, 
                "current_layer_depth": self.layer_depth,
                "current_num_fc_layers": self.cur_num_fc_layers
                }



    def reset(self, seed=None, options=None):
        # Seed self.np_random
        super().reset(seed=seed)

        self.is_start_state = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):

        if self.is_start_state:
            self.is_start_state = False

        terminated = False

        if action[self._state_elem_to_index["terminal"]] == 1:
            terminated = True

        if terminated:

            # Pass an array of tuple containing- 
            # layer_type, 
            # layer_depth, 
            # filter_depth, 
            # filter_size, 
            # stride, 
            # image_size, 
            # fc_size, 
            # terminate, 
            # state_list

            layersList = []

            for s in self.current_state:
                layersList.append(s["layer_type"], 
                                  s["layer_depth"], 
                                  s["filter_depth"],
                                  s["filter_size"],
                                  1, #stride hardcoded to 1
                                  s["image_size"],
                                  s["fc_size"],
                                  0,
                                  []
                                  )

            reward = generate_and_train(layersList, self.train_data, self.test_data)

        else:
            reward = 0

        self.layer_depth += 1    

        self.current_state.append({

            "layer_type": self._discrete_to_layer_type[action[self._state_elem_to_index["layer_type"]]], # conv, pool, fc, softmax
            "layer_depth": self.layer_depth, # Current depth of network (8 layers max)
            "filter_depth": self._discrete_to_filter_depth[action[self._state_elem_to_index["filter_depth"]]], # Used for conv (0, 64, 128, 256, 512) -- 0 is no filter
            "filter_size": self._discrete_to_filter_size[action[self._state_elem_to_index["filter_size"]]], # Used for conv and pool (1,3,5) -- 0 is no filter
            "fc_size": self._discrete_to_fc_size[action[self._state_elem_to_index["fc_size"]]],
            "image_size": self.current_image_size
        })  

        if self._discrete_to_layer_type[action[self._state_elem_to_index["layer_type"]]] == "conv" or self._discrete_to_layer_type[action[self._state_elem_to_index["layer_type"]]] == "pool":
            self.current_image_size = self._calculate_image_size(self.current_image_size, 
                                                                 self.current_state[-1]["filter_size"], 1)

        # self.current_image_size = self._calculate_image_size(self, self.current_image_size, 
        #                                                      self.current_state[-1]["filter_size"], 1)  


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
