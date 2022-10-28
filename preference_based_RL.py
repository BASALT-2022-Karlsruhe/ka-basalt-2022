"""Training script for preference-based reinforcment learning."""
import pickle
from argparse import ArgumentParser
from symbol import comparison

import gym
import gym.spaces as spaces
import minerl  # noqa
import numpy as np
import torch as th
from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import CnnRewardNet, NormalizedRewardNet, RewardNet
from imitation.util import logger as imit_logger
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3.ppo.ppo import PPO

import sb3_minerl_envs  # noqa
import wandb
from impala_based_models import ImpalaRegressor
from openai_vpt.agent import MineRLAgent
from sb3_policy_wrapper import MinecraftActorCriticPolicy


class ImpalaRewardNet(RewardNet):
    """Reward network based on the ImpalaCNN."""

    def __init__(self, observation_space, action_space, model_width=1):
        super().__init__(observation_space, action_space)
        self.net = ImpalaRegressor(cnn_width=model_width)

    def forward(self, state, action, next_state, done):
        rewards = self.net(state).squeeze()
        return rewards


def load_model_parameters(path_to_model_file):
    """Load VPT model params."""
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def preference_based_RL_train(
    env_str,
    in_model,
    in_weights_policy,
    out_weights_policy,
    in_weights_rewardnet,
    out_weights_rewardnet,
    max_episode_steps,
    reward_net_arch,
):
    """Training workflow for preference-based RL."""
    # Hyperparameters

    seed = 0

    # Reward model training
    n_iterations = 5
    n_epochs_reward_model = 3
    n_comparisons = 10
    comparison_queue_size = None  # None = unbounded queue size
    initial_comparison_frac = 0.1
    fragment_length = 1

    # PPO
    n_total_steps_ppo = 30
    n_epochs_ppo = 1
    n_steps_ppo = 10
    lr_ppo = 0.0003
    batch_size_ppo = 10
    ent_coef_ppo = 0.0

    # Setup W&B config
    if wandb.run is not None:
        wandb.config.seed = seed

        wandb.config.reward_net_arch = reward_net_arch
        wandb.config.n_iterations = n_iterations
        wandb.config.n_epochs_reward_model = n_epochs_reward_model
        wandb.config.n_comparisons = n_comparisons
        wandb.config.comparison_queue_size = comparison_queue_size
        wandb.config.initial_comparison_frac = initial_comparison_frac
        wandb.config.fragment_length = fragment_length

        wandb.config.n_steps_ppo = n_steps_ppo
        wandb.config.batch_size_ppo = batch_size_ppo
        wandb.config.ent_coef_ppo = ent_coef_ppo
        wandb.config.learning_rate_ppo = lr_ppo
        wandb.config.n_epochs_ppo = n_epochs_ppo
        wandb.config.n_total_steps_ppo = n_total_steps_ppo

    # Setup logger
    logger = imit_logger.configure()

    # Setup MineRL environment
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    minerl_env_str = "MineRLBasalt" + env_str
    env = gym.make(minerl_env_str + "-v0")

    # Setup MineRL agent
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    minerl_agent = MineRLAgent(
        env,
        device=device,
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    minerl_agent.load_weights(in_weights_policy)

    # Freeze most params if using small dataset
    for param in minerl_agent.policy.parameters():
        param.requires_grad = False
    # Unfreeze final layers and policy and value head
    for param in minerl_agent.policy.net.lastlayer.parameters():
        param.requires_grad = True
    for param in minerl_agent.policy.pi_head.parameters():
        param.requires_grad = True
    for param in minerl_agent.policy.value_head.parameters():
        param.requires_grad = True

    # Setup MineRL VecEnv
    venv = make_vec_env(
        minerl_env_str + "SB3-v0",
        # Keep this at 1 since we are not keeping track of multiple hidden states
        n_envs=1,
        # This should be sufficiently high for the given task
        max_episode_steps=max_episode_steps,
        env_make_kwargs={"minerl_agent": minerl_agent},
    )

    # Define reward model
    image_obs_space = spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
    if reward_net_arch == "CNN":
        reward_net = CnnRewardNet(image_obs_space, venv.action_space, use_action=False)
    elif reward_net_arch == "ImpalaCNN":
        reward_net = ImpalaRewardNet(image_obs_space, venv.action_space)
    else:
        raise ValueError(f"Reward network architecture unknown: {reward_net_arch}")
    if in_weights_rewardnet:
        reward_net.load_state_dict(th.load(in_weights_rewardnet))
    reward_net = NormalizedRewardNet(reward_net, RunningNorm)
    preference_model = preference_comparisons.PreferenceModel(reward_net)

    fragmenter = preference_comparisons.MineRLFragmenter(
        warning_threshold=0,
        seed=seed,
        custom_logger=logger,
    )

    gatherer = preference_comparisons.PrefCollectGatherer(
        pref_collect_address="http://127.0.0.1:8000",
        video_output_dir="/home/aicrowd/pref-collect/videofiles/",
        custom_logger=logger,
    )

    reward_trainer = preference_comparisons.BasicRewardTrainer(
        model=reward_net,
        loss=preference_comparisons.CrossEntropyRewardLoss(preference_model),
        epochs=n_epochs_reward_model,
        custom_logger=logger,
    )

    agent = PPO(
        policy=MinecraftActorCriticPolicy,
        policy_kwargs={
            "minerl_agent": minerl_agent,
            "optimizer_class": th.optim.Adam,
        },
        env=venv,
        seed=seed,
        n_steps=n_steps_ppo // venv.num_envs,
        batch_size=batch_size_ppo,
        ent_coef=ent_coef_ppo,
        learning_rate=lr_ppo,
        n_epochs=n_epochs_ppo,
    )

    trajectory_generator = preference_comparisons.MineRLAgentTrainer(
        algorithm=agent,
        reward_fn=reward_net,
        venv=venv,
        exploration_frac=0.0,
        seed=seed,
        custom_logger=logger,
    )

    pref_comparisons = preference_comparisons.MineRLPreferenceComparisons(
        trajectory_generator,
        reward_net,
        num_iterations=n_iterations,
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        comparison_queue_size=comparison_queue_size,
        fragment_length=fragment_length,
        transition_oversampling=1,
        initial_comparison_frac=initial_comparison_frac,
        allow_variable_horizon=True,
        seed=seed,
        initial_epoch_multiplier=1,
    )

    # Run training
    pref_comparisons.train(
        total_timesteps=n_total_steps_ppo,  # For good performance this should be 1_000_000
        total_comparisons=n_comparisons,  # For good performance this should be 5_000
    )

    venv.close()

    # Save finetuned policy weights
    state_dict = minerl_agent.policy.state_dict()
    th.save(state_dict, out_weights_policy)

    # Save reward network
    th.save(reward_net.state_dict(), out_weights_rewardnet)

    print("Finished")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        help="Environment name from [FindCave, MakeWaterfall, \
            CreateVillageAnimalPen, BuildVillageHouse]",
        default="FindCave",
    )
    parser.add_argument(
        "--in-model",
        type=str,
        help="Path to the .model file of the policy to be finetuned",
        default="data/VPT-models/foundation-model-1x.model",
    )
    parser.add_argument(
        "--in-weights-policy",
        type=str,
        help="Path to the .weights file of the policy to be finetuned",
        default="data/VPT-models/foundation-model-1x.weights",
    )
    parser.add_argument(
        "--out-weights-policy",
        type=str,
        help="Path where finetuned policy weights will be saved",
        default="train/PrefRLFinetuned.weights",
    )
    parser.add_argument(
        "--in-weights-rewardnet",
        type=str,
        help="Path to the .weights file of the reward network",
        default=None,
    )
    parser.add_argument(
        "--out-weights-rewardnet",
        type=str,
        help="Path where reward network weights will be saved",
        default="train/PrefRLRewardNet.weights",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        help="Maximum number of steps in each episode.",
        default=10,
    )
    parser.add_argument(
        "--reward-net-arch",
        type=str,
        help='Reward network architecture. Either "CNN" or "ImpalaCNN".',
        default="CNN",
    )

    args = parser.parse_args()
    preference_based_RL_train(
        args.env,
        args.in_model,
        args.in_weights_policy,
        args.out_weights_policy,
        args.in_weights_rewardnet,
        args.out_weights_rewardnet,
        args.max_episode_steps,
        args.reward_net_arch,
    )
