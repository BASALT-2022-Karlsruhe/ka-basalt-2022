from collections import OrderedDict

import gym
from gym.spaces import Box, Dict, Discrete
import numpy as np


class RewardModelWrapper(gym.Wrapper):
    """Replaces the environment reward with the one obtained under a reward model"""

    def __init__(self, env, reward_model, reward_model_kwargs):
        super().__init__(env)
        self.reward_model = reward_model
        self.reward_model_kwargs = reward_model_kwargs

        self.last_obs = None

    def reset(self):
        initial_obs = self.env.reset()
        self.last_obs = initial_obs
        return initial_obs

    def step(self, action):
        obs, _, terminated, info = self.env.step(action)

        if self.reward_model_kwargs["action_dependent"]:
            reward = self.reward_model(self.last_obs, action)
        else:
            reward = self.reward_model(obs)
        return obs, reward, terminated, info


class DictToBoxActionSpace(gym.Wrapper):
    """Converts env with gym.Dict action space into having a gym.Box action space"""

    def __init__(self, env, action_transformer, action_mapper):
        super().__init__(env)

        if not isinstance(self.env.action_space, Dict):
            raise ValueError("Original action space is not of type gym.Dict.")

        self.action_transformer = action_transformer
        self.action_mapper = action_mapper

        self.action_space = self._convert_to_box_action_space(
            self.env.action_space)

        # check action space conversion
        random_action = self.env.action_space.sample()
        numpy_action = self.dict_to_numpy(random_action)
        assert numpy_action in self.action_space
        dict_action = self.numpy_to_dict(numpy_action, no_batch=True)
        assert dict_action in self.env.action_space
        numpy_action2 = self.dict_to_numpy(dict_action)
        dict_action2 = self.numpy_to_dict(numpy_action2, no_batch=True)
        assert (numpy_action == numpy_action2).all()
        for key in random_action.keys():
            if isinstance(dict_action[key], np.ndarray):
                assert (dict_action[key] == dict_action2[key]).all()
            else:
                assert dict_action[key] == dict_action2[key]

    def dict_to_numpy(self, dict_action):
        dict_numpy_action = self.action_transformer.dict_to_numpy(dict_action)
        return np.concatenate((dict_numpy_action["buttons"], dict_numpy_action["camera"]))

    def numpy_to_dict(self, numpy_action, no_batch=False):
        dict_numpy_action = {
            "buttons": numpy_action[:-2], "camera": numpy_action[-2:]}
        if not no_batch:
            dict_numpy_action = self.action_mapper.to_factored(
                dict_numpy_action)
        dict_action = self.action_transformer.numpy_to_dict(dict_numpy_action)
        dict_action["ESC"] = np.array(0)
        dict_action["swapHands"] = np.array(0)
        dict_action["pickItem"] = np.array(0)
        ordered_keys = self.env.action_space.spaces.keys()
        dict_action = OrderedDict(
            {key: dict_action[key] for key in ordered_keys})
        return dict_action

    def _convert_to_box_action_space(self, action_space):
        lower_bounds = []
        higher_bounds = []
        box_lower_bounds = []
        box_higher_bounds = []
        for name, space in action_space.spaces.items():
            if name in ["pickItem", "swapHands", "ESC"]:
                continue
            if isinstance(space, Discrete):
                lower_bounds.append(0)
                higher_bounds.append(space.n - 1)
            elif isinstance(space, Box):
                box_lower_bounds.extend(space.low)
                box_higher_bounds.extend(space.high)
            else:
                raise ValueError(f"Space type {type(space)} is not supported.")
        # dtype=np.int64 because this the output of the CameraQuantizer
        return Box(low=np.array(lower_bounds + box_lower_bounds), high=np.array(higher_bounds + box_higher_bounds), dtype=np.int64)

    def step(self, action):
        # transform vector action to dict action
        dict_action = self.numpy_to_dict(action)

        obs, reward, terminated, info = self.env.step(dict_action)

        return obs, reward, terminated, info
