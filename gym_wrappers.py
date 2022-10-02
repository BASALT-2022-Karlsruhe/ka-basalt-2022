from collections import OrderedDict

import gym
from gym.spaces import Tuple, Box, Dict, Discrete, MultiDiscrete
import numpy as np
import torch as th


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

    def __init__(self, env, minerl_agent):
        super().__init__(env)

        if not isinstance(self.env.action_space, Dict):
            raise ValueError("Original action space is not of type gym.Dict.")

        self.minerl_agent = minerl_agent

        self.action_space = MultiDiscrete([121, 8100])

        if False:
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

    """
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
    """

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
        # transform vector action to dict action to MineRL action
        dict_action = {"buttons": action[..., 0], "camera": action[..., 1]}
        minerl_action = self.minerl_agent._agent_action_to_env(dict_action)

        # TODO implement policy that controls the remaining actions:
        minerl_action["ESC"] = np.array(0)
        minerl_action["swapHands"] = np.array(0)
        minerl_action["pickItem"] = np.array(0)

        obs, reward, terminated, info = self.env.step(minerl_action)

        return obs, reward, terminated, info


class HiddenStateObservationSpace(gym.Wrapper):
    def __init__(self, env, minerl_agent):
        super().__init__(env)
        self.minerl_agent = minerl_agent

        # TODO determine state_in shapes from architecture
        img_shape = self.minerl_agent._env_obs_to_agent(
            self.env.observation_space.sample())["img"].shape
        first_shape = self.minerl_agent._dummy_first.shape
        self.observation_space = Dict({
            "img": Box(-10, 10, shape=img_shape),
            "first": Box(-10, 10, shape=first_shape),
            "state_in1": Box(-10, 10, shape=(4, 1, 128)),
            "state_in2": Box(-10, 10, shape=(4, 128, 1024)),
            "state_in3": Box(-10, 10, shape=(4, 128, 1024))
        })

    def add_hidden_state(self, obs):
        obs["first"] = self.minerl_agent._dummy_first.bool()
        state_in1 = []
        for i in range(len(self.minerl_agent.hidden_state)):
            if self.minerl_agent.hidden_state[i][0] is None:
                nan_tensor = th.zeros((1, 1, 128), dtype=th.float32)
                nan_tensor[:, :, :] = float("nan")
                state_in1.append(nan_tensor)
            else:
                state_in1.append(self.minerl_agent.hidden_state[i][0])
        obs["state_in1"] = th.cat(tuple(state_in1), dim=0)
        obs["state_in2"] = th.cat(tuple(self.minerl_agent.hidden_state[i][1][0]
                                  for i in range(len(self.minerl_agent.hidden_state))), dim=0)
        obs["state_in3"] = th.cat(tuple(self.minerl_agent.hidden_state[i][1][1]
                                  for i in range(len(self.minerl_agent.hidden_state))), dim=0)
        return obs

    def reset(self):
        self.minerl_agent.reset()
        obs = self.env.reset()
        agent_obs = self.minerl_agent._env_obs_to_agent(obs)
        augmented_obs = self.add_hidden_state(agent_obs)
        return augmented_obs

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        agent_obs = self.minerl_agent._env_obs_to_agent(obs)
        augmented_obs = self.add_hidden_state(agent_obs)
        return augmented_obs, reward, terminated, info
