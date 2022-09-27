from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from copy import deepcopy

import gym
import numpy as np
import torch as th
from gym3.types import DictType
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.distributions import Distribution
from openai_vpt.agent import POLICY_KWARGS, PI_HEAD_KWARGS, MineRLAgent


def make_SB3_policy_wrapper(env: gym.Env, minerl_agent: MineRLAgent, lr: Union[float, int]):
    sb3_ac_policy = MinecraftActorCriticPolicy(
        env.observation_space,
        env.action_space,
        lr,
        minerl_agent,
        POLICY_KWARGS,
        PI_HEAD_KWARGS)
    return sb3_ac_policy


class MinecraftActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            minerl_agent,
            policy_kwargs,
            pi_head_kwargs,
            **kwargs):

        # TODO check compatibility of MineRL observation_space
        super(MinecraftActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )

        self.minerl_agent = minerl_agent
        self.minerl_policy_kwargs = policy_kwargs
        self.minerl_pi_head_kwargs

        self.ortho_init = False

        # TODO Action distribution -> this needs to be implement using the DictActionHead
        # self.action_dist = HierarchicalDistribution(...)

    def forward(self, obs: th.Tensor, first: th.Tensor, state_in, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param first: the initial hidden state
        :param state_in: list of intermediate hidden states of previous inference
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # TODO it seems like SB3 requires compatibility with batches of observations, MineRL does not have that as far as I can tell.. or does it?
        (pi_logits, vpred, _), state_out = self.minerl_agent.policy(
            obs, first, state_in)
        action = self.minerl_agent.policy.pi_head.sample(
            pi_logits, deterministic=deterministic)
        value = self.minerl_agent.policy.value_head.denormalize(vpred)[:, 0]
        log_prob = self.minerl_agent.policy.pi_head.log_prob(action, pi_logits)
        return action, value, log_prob

    # TODO do we need to modify this?
    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(
            self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    # TODO here they construct action_net and value_net for the output
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        latent_dim_pi = self.minerl_agent.policy.net.hidsize

        # TODO is the output of action_net an action or a probability distribution?
        self.action_net = self.minerl_agent.policy.pi_head

        self.value_net = self.minerl_agent.policy.value_head

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    # TODO the following functions compatible with our action head
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        if isinstance(self.action_dist, HierarchicalDistribution):
            return self.action_dist.proba_distribution(mean_actions)
        else:
            raise ValueError(("Invalid distribution!"))

    def _predict(self, observation: th.Tensor, first, state_in, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        (pi_logits, vpred, _), state_out = self.minerl_agent.policy(
            observation, first, state_in)
        action = self.minerl_agent.policy.pi_head.sample(
            pi_logits, deterministic=deterministic)

        return action

    def evaluate_actions(self, obs: th.Tensor, first, state_in, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

        (pi_logits, vpred, _), state_out = self.minerl_agent.policy(
            obs, first, state_in)
        action = self.minerl_agent.policy.pi_head.sample(
            pi_logits, deterministic=deterministic)
        value = self.minerl_agent.policy.value_head.denormalize(vpred)[:, 0]
        log_prob = self.minerl_agent.policy.pi_head.log_prob(action, pi_logits)
        return value, log_prob, self.minerl_agent.policy.pi_head.entropy(pi_logits)

    def get_distribution(self, obs: th.Tensor, first, state_in) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        (latent_pi, latent_vf), state_out = self.minerl_agent.policy.net(
            obs, first, state_in)

        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor, first, state_in) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        (_, latent_vf), state_out = self.minerl_agent.policy.net(obs, first, state_in)
        return self.value_net(latent_vf)


# TODO create custom Distribution wrapper for the hierarchical MineRL action space using the existing DictActionHead class
class HierarchicalDistribution(Distribution):
    def __init__(self, pi_head):
        super().__init__()
        #self.action_dims = ...

    def proba_distribution_net(self, latent_dim):
        # action_logits = ...
        return action_logits


if __name__ == "__main__":
    import pickle
    from stable_baselines3 import PPO
    import minerl

    def load_model_parameters(path_to_model_file):
        agent_parameters = pickle.load(open(path_to_model_file, "rb"))
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
        return policy_kwargs, pi_head_kwargs

    # Setup environment
    env = gym.make("MineRLBasaltFindCave-v0")

    # Setup agent
    in_model = "data/VPT-models/foundation-model-1x.model"
    in_weights = "data/VPT-models/foundation-model-1x.weights"
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    minerl_agent = MineRLAgent(
        env,
        device="cpu",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    minerl_agent.load_weights(in_weights)

    # Setup PPO
    model = PPO(
        policy=MinecraftActorCriticPolicy,
        policy_kwargs={
            "minerl_agent": minerl_agent,
            "policy_kwargs": POLICY_KWARGS,
            "pi_head_kwargs": PI_HEAD_KWARGS
        },
        env=env,
        seed=0,
        n_steps=100,
        batch_size=10,
        ent_coef=0.0,
        learning_rate=0.0003,
        verbose=1)

    # Train
    # TODO this assumes that env contains a reward function...
    model.learn(10)
