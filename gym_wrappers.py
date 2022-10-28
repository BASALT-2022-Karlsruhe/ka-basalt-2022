import gym
import numpy as np
import torch as th
from gym.spaces import Box, Dict, MultiDiscrete


class ObservationToInfos(gym.Wrapper):
    """
    Adds the observation to the infos dict.
    Useful when adding other wrappers that
    transform the original observation.
    """

    def __init_(self, env):
        super().__init__(env)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info["original_obs"] = obs
        return obs, rew, done, info


class ObservationToCPU(gym.Wrapper):
    """Transfers Tensor observations to CPU"""

    def __init__(self, env):
        super().__init__(env)

    def _to_cpu(self, obs):
        if isinstance(obs, dict):
            for key, obs_val in obs.items():
                if isinstance(obs_val, th.Tensor):
                    obs[key] = obs[key].cpu()
        elif isinstance(obs, th.Tensor):
            obs = obs.cpu()
        return obs

    def reset(self):
        obs = self.env.reset()
        return self._to_cpu(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return self._to_cpu(obs), rew, done, info


class RewardModelWrapper(gym.Wrapper):
    """Replaces the environment reward with the one obtained
    under a reward model"""

    def __init__(
        self, env, reward_model, reward_model_kwargs={"action_dependent": False}
    ):
        super().__init__(env)
        self.reward_model = reward_model
        self.reward_model_kwargs = reward_model_kwargs

        self.last_obs = None

    def update_reward_model(self, reward_model, reward_model_kwargs):
        self.reward_model = reward_model
        self.reward_model_kwargs = reward_model_kwargs

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


class DictToMultiDiscreteActionSpace(gym.Wrapper):
    """Converts Dict to MultiDiscrete action space for MineRL envs"""

    def __init__(self, env, minerl_agent=None):
        super().__init__(env)

        if not isinstance(self.env.action_space, Dict):
            raise ValueError("Original action space is not of type gym.Dict.")

        assert minerl_agent is not None

        self.minerl_agent = minerl_agent

        # first dimension = camera, second dimension = buttons
        self.action_space = MultiDiscrete([121, 8641])

        # check action space conversion
        random_env_action = self.env.action_space.sample()
        agent_action = self.minerl_agent._env_action_to_agent(random_env_action)
        array_action = self.to_array_action(agent_action)
        assert array_action.squeeze() in self.action_space

        agent_action2 = self.to_agent_action(array_action)
        env_action = self.minerl_agent._agent_action_to_env(agent_action2)
        env_action["ESC"] = env_action["swapHands"] = env_action["pickItem"] = np.array(
            0
        )
        assert env_action in self.env.action_space

    def to_array_action(self, agent_action):
        if isinstance(agent_action["camera"], np.ndarray):
            array_action = np.concatenate(
                (agent_action["camera"], agent_action["buttons"]), -1
            )
        elif isinstance(agent_action["camera"], th.Tensor):
            array_action = th.cat(
                (agent_action["camera"], agent_action["buttons"]), dim=-1
            )
        return array_action

    def to_agent_action(self, array_action):
        agent_action = {"camera": array_action[..., 0], "buttons": array_action[..., 1]}
        return agent_action

    def step(self, action):
        if len(action.shape) < 2:
            action = action[np.newaxis, :]
        # transform array action to agent action to MineRL action
        agent_action = self.to_agent_action(action)
        minerl_action = self.minerl_agent._agent_action_to_env(agent_action)

        # TODO implement policy that controls the remaining actions (especially ESC):
        minerl_action["ESC"] = np.array(0)
        minerl_action["swapHands"] = minerl_action["pickItem"] = np.array(0)

        obs, reward, terminated, info = self.env.step(minerl_action)

        return obs, reward, terminated, info


class HiddenStateObservationSpace(gym.Wrapper):
    """Augments the observation space by the hidden state of the MineRLAgent"""

    def __init__(self, env, minerl_agent=None):
        super().__init__(env)

        assert minerl_agent is not None

        self.minerl_agent = minerl_agent

        # define observation space based on model architecture
        img_shape = self.minerl_agent._env_obs_to_agent(
            self.env.observation_space.sample()
        )["img"].shape
        first_shape = self.minerl_agent._dummy_first.shape
        img_width = img_shape[2]
        hidden_width = self.minerl_agent.policy.net.lastlayer.layer.in_features

        self.observation_space = Dict(
            {
                "img": Box(0, 255, shape=img_shape, dtype=np.uint8),
                "first": Box(-10, 10, shape=first_shape),
                "state_in1": Box(-10, 10, shape=(4, 1, img_width)),
                "state_in2": Box(-10, 10, shape=(4, img_width, hidden_width)),
                "state_in3": Box(-10, 10, shape=(4, img_width, hidden_width)),
            }
        )

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
        obs["state_in2"] = th.cat(
            tuple(
                self.minerl_agent.hidden_state[i][1][0]
                for i in range(len(self.minerl_agent.hidden_state))
            ),
            dim=0,
        )
        obs["state_in3"] = th.cat(
            tuple(
                self.minerl_agent.hidden_state[i][1][1]
                for i in range(len(self.minerl_agent.hidden_state))
            ),
            dim=0,
        )
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
