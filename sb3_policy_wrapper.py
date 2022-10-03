from typing import Dict, List, Optional, Tuple

import gym
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from openai_vpt.agent import MineRLAgent


class MinecraftActorCriticPolicy(ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms wrapping OpenAI's VPT models.
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param minerl_agent: MineRL agent to be wrapped
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            minerl_agent: MineRLAgent,
            **kwargs):

        self.minerl_agent = minerl_agent

        super(MinecraftActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )

        self.ortho_init = False

    def forward(self, observation: Dict[str, Tuple], deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        # TODO it seems like SB3 requires compatibility with batches of observations, does MineRLAgent support batches?
        # unpack observation
        obs, first, state_in = self.unpack_dict_obs(observation)

        # inference
        (pi_logits, vpred, _), state_out = self.minerl_agent.policy(
            obs, first, state_in)

        # action sampling
        action = self.action_net.sample(
            pi_logits, deterministic=deterministic)

        value = self.value_net.denormalize(vpred)[:, 0]
        log_prob = self.action_net.logprob(action, pi_logits)

        # convert agent action into array so it can pass through the SB3 functions
        array_action = th.cat((action["camera"], action["buttons"]), dim=-1)

        return array_action, value, log_prob

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        # Setup action and value heads
        self.action_net = self.minerl_agent.policy.pi_head
        self.value_net = self.minerl_agent.policy.value_head

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def evaluate_actions(self, obs: Dict[str, th.Tensor], actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

        # convert array actions to agent actions
        agent_actions = {"camera": actions[..., 0], "buttons": actions[..., 1]}

        # unpack observation
        img_obs, first, state_in = self.unpack_dict_obs(obs)

        # inference
        (pi_logits, vpred, _), state_out = self.minerl_agent.policy(
            img_obs, first, state_in)

        value = self.value_net.denormalize(vpred)[:, 0]
        log_prob = self.action_net.logprob(agent_actions, pi_logits)
        entropy = self.action_net.entropy(pi_logits)

        return value, log_prob, entropy

    def predict_values(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: 
        :return: the estimated values.
        """

        # unpack observation
        img_obs, first, state_in = self.unpack_dict_obs(obs)

        # inference
        (_, latent_vf), state_out = self.minerl_agent.policy.net(
            img_obs, state_in, {"first": first})
        value = self.value_net(latent_vf)

        return value

    def unpack_dict_obs(self, obs: Dict[str, th.Tensor]) -> Tuple[Dict[str, th.Tensor], th.Tensor, List[Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]]]:
        """
        Unpack the observation dictionary 

        :param obs: 
        :return: the agent image observation, the first input tensor and the hidden state tensors
        """

        img_obs = {"img": obs["img"]}
        first_obs = obs["first"].bool()
        state_in_obs = []

        for i in range(len(self.minerl_agent.hidden_state)):
            state_in1 = obs["state_in1"][:, i, :, :]
            if th.isnan(state_in1).all():
                state_in1 = None
            state_in_tuple = (
                state_in1, (obs["state_in2"][:, i, :, :], obs["state_in3"][:, i, :, :]))
            state_in_obs.append(state_in_tuple)

        return img_obs, first_obs, state_in_obs


if __name__ == "__main__":
    import pickle

    from stable_baselines3 import PPO
    import minerl

    from openai_vpt.agent import MineRLAgent
    from gym_wrappers import RewardModelWrapper, DictToMultiDiscreteActionSpace, HiddenStateObservationSpace

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
        device="cpu",  # "cuda" for GPU usage!
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs
    )
    minerl_agent.load_weights(in_weights)

    # Make env compatible with SB3
    wrapped_env = DictToMultiDiscreteActionSpace(env, minerl_agent)
    wrapped_env = HiddenStateObservationSpace(wrapped_env, minerl_agent)

    # Augment MineRL env with reward model
    wrapped_env = RewardModelWrapper(wrapped_env, lambda obs: 0., {
                                     "action_dependent": False})

    # Setup PPO
    model = PPO(
        policy=MinecraftActorCriticPolicy,
        policy_kwargs={
            "minerl_agent": minerl_agent,
            "optimizer_class": th.optim.Adam,
        },
        env=wrapped_env,
        seed=0,
        n_steps=100,
        batch_size=10,
        ent_coef=0.0,
        learning_rate=0.0003,
        verbose=1)

    # Train
    model.learn(1)

    wrapped_env.close()
    print("Finished")
