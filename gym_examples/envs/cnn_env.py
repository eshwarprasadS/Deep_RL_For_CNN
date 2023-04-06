import gym
from gym import spaces
import pygame
import numpy as np
import math
from gym_examples.envs.pytorch_parser.pytorch_parser_function import generate_and_train
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch

# TODO: Implement approach B for start state
# TODO: Terminal transition from every state
# TODO: Diff strides for pooling
# TODO: Diff pool sizes

class CNNEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, verbose=True, dataset = 'mnist'):
        super(CNNEnv, self).__init__()

        self.dataset = dataset    
        self.verbose = verbose

        if self.dataset == 'mnist':
        
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
        
        elif self.dataset == 'cifar10':

            self.train_data = datasets.CIFAR10(
                                root = 'data',
                                train = True,
                                transform = ToTensor(),
                                download = True,
                                )
            self.test_data = datasets.CIFAR10(
                                root = 'data',
                                train = False,
                                transform = ToTensor()
                                )
            subset_targets = [0, 1, 2, 3, 4]

            self.train_data.targets = np.array(self.train_data.targets)
            self.test_data.targets = np.array(self.test_data.targets)

            train_idx = np.isin(self.train_data.targets, subset_targets)
            test_idx = np.isin(self.test_data.targets, subset_targets)

            self.train_data.data = self.train_data.data[train_idx]
            self.train_data.targets = torch.from_numpy(self.train_data.targets[train_idx]).type(torch.LongTensor)
            self.test_data.data = self.test_data.data[test_idx]
            self.test_data.targets = torch.from_numpy(self.test_data.targets[test_idx]).type(torch.LongTensor)

        self.layer_depth_limit = 8

        self.max_image_size_for_fc = 28

        self.current_image_size = 28 # change this after each action
        self.layer_depth = 0
        self.is_start_state = 1 #change this to 0 once we take the first action
        self.cur_num_fc_layers = 0

        self.fc_terminate = False
        self.allow_consecutive_pooling = False
        self.allow_initial_pooling = False

        self.max_fc_layers_allowed = 2


        self.current_state = [] #array of dictionaries

        self.observation_space = spaces.Dict(
            {
                "layer_type": spaces.Discrete(3), # conv, pool, fc
                "layer_depth": spaces.Discrete(9), # Current depth of network (START layer + 8 layers max)
                "filter_depth": spaces.Discrete(5), # Used for conv (0, 10, 20, 32, 64)
                "filter_size": spaces.Discrete(4), # Used for conv and pool (0, 1, 3, 5)
                "fc_size": spaces.Discrete(5), # Used for fc -- number of neurons in layer (0, 512, 256, 128, 64)
                "is_start": spaces.Discrete(2), # 0 if not start state, 1 if start state
                "pool_size_and_stride": spaces.Discrete(4) # (0,0), (5,3), (3,2), (2,2)
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
        #         "terminal": spaces.Discrete(2), # 0 if not terminal, 1 if terminal
        #         "pool_size_and_stride": spaces.Discrete(4)
        #     }
        #          )
        self.action_space = spaces.MultiDiscrete([3, 9, 5, 4, 5, 2, 4]) # 3, 9, 5, 4, 5, 2, 4 corresponds to the number of discrete actions in each space

        self._discrete_to_layer_type = {    # Maps discrete action space to layer type
            0: "conv",
            1: "pool",
            2: "fc"
        }

        self._layer_type_to_discrete = {    # Maps discrete action space to layer type
            "conv": 0,
            "pool": 1,
            "fc": 2
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

        self._discrete_to_pool_size = {
            0: (0, 0),
            1: (5, 3),
            2: (3, 2),
            3: (2, 2)
        }

        self._pool_size_to_discrete = {   # Maps discrete action space to filter size
            0: 0,
            5: 1,
            3: 2,
            2: 3
        }

        self._discrete_to_fc_size = {    # Maps discrete action space to fc size
            0: 0,
            1: 512,
            2: 256,
            3: 128,
            4: 64
        }
        self._fc_size_to_discrete = {    # Maps discrete action space to fc size
            0: 0,
            512: 1,
            256: 2,
            128: 3,
            64: 4
        }
        self._state_elem_to_index = {
            "layer_type": 0,
            "layer_depth": 1,
            "filter_depth": 2,
            "filter_size": 3,
            "fc_size": 4,
            "terminal": 5,
            "pool_size_and_stride": 6,
        }

        # initiate a conv layer with filter size 5 and 10 filters for all envs
        # TODO: change this to a random conv layer, or random layer type or decide on how to initiate a dummy layer
        # self.current_image_size = self._calculate_image_size(self.current_image_size,
        #                                                          self.current_state[-1])

        self.current_state.append({

            "layer_type": self._discrete_to_layer_type[0], # any, doesn't matter for conv
            "layer_depth": self.layer_depth, # any, doesn't matter for conv
            "filter_depth": self._discrete_to_filter_depth[0], # any, doesn't matter for conv
            "filter_size": self._discrete_to_filter_size[0], # any, doesn't matter for conv
            "fc_size": self._discrete_to_fc_size[0], # any, doesn't matter for conv
            "image_size": self.current_image_size, # 28
            "pool_size": self._discrete_to_pool_size[0][0], # any, doesn't matter for conv
            "pool_stride": self._discrete_to_pool_size[0][1], # any, doesn't matter for conv
            "is_start": self.is_start_state # Yes, this is the start state
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
        md_mask = tuple([np.zeros(x, dtype=np.int8) for x in self.action_space.nvec])

        # print('inside layer depth limit')
        # enable valid convolutions
        if self.is_start_state or \
            (self.is_start_state == False and self.current_state[-1]["layer_type"] in \
            set(["conv", "pool"])):

            # print('enabling convolutions')
            for d in list(self._discrete_to_filter_depth.keys())[1:]:
                for f in self._valid_filter_sizes_for_image():
                    self._enable_convolution(md_mask, f, d)

        else:
            # enable only 0 filter depth and 0 filter size as convolutions are not allowed
            md_mask[self._state_elem_to_index["filter_depth"]][0] = 1
            md_mask[self._state_elem_to_index["filter_size"]][0] = 1


        #TODO: add global average pooling??


        # enable valid pooling layers
        if (self.is_start_state == False and \
            (self.current_state[-1]["layer_type"] == "conv" or
            (self.current_state[-1]["layer_type"] == "pool" and self.allow_consecutive_pooling))) or \
            (self.is_start_state and self.allow_initial_pooling):

            # print('enabling valid pools')
            for f in self._valid_pooling_sizes_for_image():
                self._enable_pooling(md_mask, f)
        else:
            # enable only 0 pool size and stride as pooling is not allowed
            md_mask[self._state_elem_to_index["pool_size_and_stride"]][0] = 1

        # # enable valid fully connected layers
        # if (self.is_start_state == False and self._allow_fully_connected() and
        # self.current_state[-1]["layer_type"] in ['start', 'conv', 'pool']):

        #     for fc_sz in self._valid_fc_sizes_for_image():
        #         self._enable_fc(md_mask, fc_sz)

        # enable all FC layers if we are transitioning from conv or pool layers
        if (self.is_start_state == False and self._allow_fully_connected() and
        self.current_state[-1]["layer_type"] in ['conv', 'pool']):

            for fc_sz in list(self._discrete_to_fc_size.keys())[1:]:
                self._enable_fc(md_mask, fc_sz)

        #fc to fc transitions, only allow smaller fc layers than the previous one and only 2 fc layers
        elif self.is_start_state == False and self.current_state[-1]["layer_type"] == 'fc' \
        and self.cur_num_fc_layers < self.max_fc_layers_allowed:

            for fc_sz in self._valid_fc_sizes_for_image():
                # self.cur_num_fc_layers += 1
                self._enable_fc(md_mask, fc_sz)

            md_mask[self._state_elem_to_index["terminal"]][1] = 1
            self.fc_terminate = True


        else:
            # enable only 0 fc size as fc layers are not allowed
            md_mask[self._state_elem_to_index["fc_size"]][0] = 1

        # Enable terminal if its an FC layer
        # TODO: Do we add terminal from other layers as well? -> Yes | Done
        # if self.is_start_state == False and self.current_state[-1]["layer_type"] == 'fc':
        #     md_mask[self._state_elem_to_index["terminal"]][1] = 1

        if self.fc_terminate == False:
            # Enable terminal and non terminal from every layer type, if not start state and layer depth limit is not reached
            if self.layer_depth < self.layer_depth_limit-1 and self.is_start_state == False:
                md_mask[self._state_elem_to_index["terminal"]][0] = 1
                md_mask[self._state_elem_to_index["terminal"]][1] = 1

            # Enable only terminal if layer depth limit is reached
            elif self.is_start_state == False:
                md_mask[self._state_elem_to_index["terminal"]][1] = 1

            elif self.is_start_state == True: #allow only non terminal action from start state
                md_mask[self._state_elem_to_index["terminal"]][0] = 1

        # enable only layer_depth+1 action if layer depth limit is not reached
        if self.layer_depth < self.layer_depth_limit-1:
            md_mask[self._state_elem_to_index["layer_depth"]][self.layer_depth+1] = 1


        # mask = np.ones(self.action_space.n, dtype=np.int8)
        # mask[invalid_actions] = 0

        return md_mask

    def action_masks(self):
        mask = self.get_valid_action_mask()
        return [y for x in mask for y in x]

    def _enable_convolution(self, md_mask, f, d):
        # print('inside enable conv')
        md_mask[self._state_elem_to_index["layer_type"]][self._layer_type_to_discrete["conv"]] = 1
        md_mask[self._state_elem_to_index["filter_depth"]][d] = 1
        md_mask[self._state_elem_to_index["filter_size"]][f] = 1

    def _enable_pooling(self, md_mask, f):
        md_mask[self._state_elem_to_index["layer_type"]][self._layer_type_to_discrete["pool"]] = 1
        md_mask[self._state_elem_to_index["pool_size_and_stride"]][f] = 1

    def _enable_fc(self, md_mask, fc_sz):
        md_mask[self._state_elem_to_index["layer_type"]][self._layer_type_to_discrete["fc"]] = 1
        md_mask[self._state_elem_to_index["fc_size"]][fc_sz] = 1


    def _allow_fully_connected(self):
        return self.current_image_size <= self.max_image_size_for_fc

    def _valid_pooling_sizes_for_image(self):
        return [p for p in list(self._discrete_to_pool_size.keys())[1:] if self._discrete_to_pool_size[p][0] < self.current_image_size]

    def _valid_filter_sizes_for_image(self):
        return [c for c in list(self._discrete_to_filter_size.keys())[1:] if self._discrete_to_filter_size[c] < self.current_image_size]

    def _valid_fc_sizes_for_image(self):
        if self.current_state[-1]["layer_type"] == 'fc':
            return [fc for fc in list(self._discrete_to_fc_size.keys())[1:] if self._discrete_to_fc_size[fc] < self.current_state[-1]["fc_size"]]

        return list(self._discrete_to_layer_type.keys())[1:]

    def _calculate_image_size(self, image_size, prev_layer):

        new_size = None

        if prev_layer["layer_type"] == "conv":
            new_size = int(math.ceil(float(image_size - prev_layer["filter_size"] + 1) / float(1)))
        else:
            new_size = int(math.ceil(float(image_size - prev_layer["pool_size"] + 1) / float(prev_layer["pool_stride"])))

        return new_size

    def _get_obs(self):

        """ Should be same format as-
            "layer_type": spaces.Discrete(4), # conv, pool, fc, softmax
            "layer_depth": spaces.Discrete(8), # Current depth of network (8 layers max)
            "filter_depth": spaces.Discrete(5), # Used for conv (0, 64, 128, 256, 512) -- 0 is no filter
            "filter_size": spaces.Discrete(4), # Used for conv and pool (1,3,5) -- 0 is no filter
            "fc_size": spaces.Discrete(4), # Used for fc and softmax -- number of neurons in layer (512, 256, 128, 64)
            "is_start": spaces.Discrete(2), # 0 if not start state, 1 if start state
            "pool_size_and_stride": spaces.Discrete(4) # (5,3), (3,2), (2,2) -- 0 is no filter
        """

        return {"layer_type": self._layer_type_to_discrete[self.current_state[-1]["layer_type"]],
                "layer_depth": self.current_state[-1]["layer_depth"],
                "filter_depth": self._filter_depth_to_discrete[self.current_state[-1]["filter_depth"]],
                "filter_size": self._filter_size_to_discrete[self.current_state[-1]["filter_size"]],
                "fc_size": self._fc_size_to_discrete[self.current_state[-1]["fc_size"]],
                "is_start": self.is_start_state,
                "pool_size_and_stride": self._pool_size_to_discrete[self.current_state[-1]["pool_size"]], #since size and stride is a pair we can return same discrete for both
                }
        # return {"current_state": self.current_state}

    def _get_info(self):
        return {
                "current_network" : self.current_state,
                "current_image_size": self.current_image_size,
                "current_layer_depth": self.layer_depth,
                "current_num_fc_layers": self.cur_num_fc_layers,
                "obs_vector": np.array(list(self._get_obs().values()))}



    def reset(self, seed=None, options=None):
        # Seed self.np_random
        super().reset(seed=seed)

        self.current_image_size = 28 # change this after each action
        self.layer_depth = 0
        self.is_start_state = 1 #change this to 0 once we take the first action
        self.cur_num_fc_layers = 0
        self.current_state = [] #array of dictionaries

        self.current_state.append({

            "layer_type": self._discrete_to_layer_type[0], # any, doesn't matter for conv
            "layer_depth": self.layer_depth, # any, doesn't matter for conv
            "filter_depth": self._discrete_to_filter_depth[0], # any, doesn't matter for conv
            "filter_size": self._discrete_to_filter_size[0], # any, doesn't matter for conv
            "fc_size": self._discrete_to_fc_size[0], # any, doesn't matter for conv
            "image_size": self.current_image_size, # 28
            "pool_size": self._discrete_to_pool_size[0][0], # any, doesn't matter for conv
            "pool_stride": self._discrete_to_pool_size[0][1], # any, doesn't matter for conv
            "is_start": self.is_start_state # Yes, this is the start state
        })

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        if self.is_start_state:
            self.is_start_state = 0

        terminated = False

        if self.layer_depth < self.layer_depth_limit-1:
            self.layer_depth += 1

        layer = {
            "layer_type": self._discrete_to_layer_type[action[self._state_elem_to_index["layer_type"]]], # conv, pool, fc
            "layer_depth": self.layer_depth, # Current depth of network (8 layers max)
            "filter_depth": self._discrete_to_filter_depth[action[self._state_elem_to_index["filter_depth"]]], # Used for conv (0, 64, 128, 256, 512) -- 0 is no filter
            "filter_size": self._discrete_to_filter_size[action[self._state_elem_to_index["filter_size"]]], # Used for conv and pool (0, 1,3,5) -- 0 is no filter
            "fc_size": self._discrete_to_fc_size[action[self._state_elem_to_index["fc_size"]]],
            "image_size": self.current_image_size,
            "is_start": self.is_start_state,
            "pool_size": self._discrete_to_pool_size[action[self._state_elem_to_index["pool_size_and_stride"]]][0],
            "pool_stride": self._discrete_to_pool_size[action[self._state_elem_to_index["pool_size_and_stride"]]][1],
        }


        if layer["layer_type"] == "conv" or layer["layer_type"] == "pool":
            self.current_image_size = self._calculate_image_size(self.current_image_size,
                                                                 layer)

        layer["image_size"] = self.current_image_size
        # if self._discrete_to_layer_type[action[self._state_elem_to_index["layer_type"]]] == "conv" or self._discrete_to_layer_type[action[self._state_elem_to_index["layer_type"]]] == "pool":
        #     print('current_state layer type =', self.current_state[-1]["layer_type"])
        #     self.current_image_size = self._calculate_image_size(self.current_image_size,
        #                                                          self.current_state[-1])

        # self.current_image_size = self._calculate_image_size(self, self.current_image_size,
        #                                                      self.current_state[-1]["filter_size"], 1)
        if layer["layer_type"] == "fc":
            self.cur_num_fc_layers += 1

        self.current_state.append(layer)
        # self.current_state.append({

        #     "layer_type": self._discrete_to_layer_type[action[self._state_elem_to_index["layer_type"]]], # conv, pool, fc
        #     "layer_depth": self.layer_depth, # Current depth of network (8 layers max)
        #     "filter_depth": self._discrete_to_filter_depth[action[self._state_elem_to_index["filter_depth"]]], # Used for conv (0, 64, 128, 256, 512) -- 0 is no filter
        #     "filter_size": self._discrete_to_filter_size[action[self._state_elem_to_index["filter_size"]]], # Used for conv and pool (0, 1,3,5) -- 0 is no filter
        #     "fc_size": self._discrete_to_fc_size[action[self._state_elem_to_index["fc_size"]]],
        #     "image_size": self.current_image_size,
        #     "is_start": self.is_start_state,
        #     "pool_size": self._discrete_to_pool_size[action[self._state_elem_to_index["pool_size_and_stride"]]][0],
        #     "pool_stride": self._discrete_to_pool_size[action[self._state_elem_to_index["pool_size_and_stride"]]][1],
        # })

        # terminate manually if no more layers can be added
        mask = self.get_valid_action_mask()
        if np.sum(mask[0]) == 0:
            terminated = True

        # terminate if action is to terminate
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

            for s in self.current_state[1:]: #ignoring start state
                if s["layer_type"] == "conv":
                    layersList.append((s["layer_type"],
                                    s["layer_depth"],
                                    s["filter_depth"],
                                    s["filter_size"],
                                    1, # stride hardcoded to 1 for conv
                                    s["image_size"],
                                    s["fc_size"],
                                    0,
                                    []
                                    ))
                else:
                    layersList.append((s["layer_type"],
                                    s["layer_depth"],
                                    s["filter_depth"],
                                    s["pool_size"],
                                    s["pool_stride"],
                                    s["image_size"],
                                    s["fc_size"],
                                    0,
                                    []
                                    ))
            # print('layersList = ', layersList)
            if self.dataset == "mnist":
                reward, model_size = generate_and_train(layersList, self.train_data, self.test_data, dataset_name=self.dataset, n_classes=3)
            elif self.dataset == "cifar10":
                print('network = ', layersList, 'terminated =', terminated)
                reward, model_size = generate_and_train(layersList, self.train_data, self.test_data, dataset_name=self.dataset, n_classes=5)
            if self.verbose:
                print('network = ', self.current_state[1:], 'terminated =', terminated, 'reward =', reward)

        else:
            reward = 0

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
