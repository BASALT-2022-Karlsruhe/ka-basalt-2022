"""Training script for auto-laballed preference-based RL."""
import os
from argparse import ArgumentParser

import gym
import gym.spaces as spaces
import minerl  # noqa
import numpy as np
import torch as th
from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import CnnRewardNet, NormalizedRewardNet
from imitation.rewards.reward_wrapper import MineRLRewardVecEnvWrapper
from imitation.util import logger as imit_logger
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from stable_baselines3.ppo.ppo import PPO
from wandb.integration.sb3 import WandbCallback

import sb3_minerl_envs  # noqa
import wandb
from impala_based_models import ImpalaRewardNet
from openai_vpt.agent import MineRLAgent
from sb3_policy_wrapper import MinecraftActorCriticPolicy
from utils.utils import load_model_parameters


def auto_preference_based_RL_train(
    env_str,
    in_model,
    in_weights_policy,
    out_weights_policy,
    in_weights_rewardnet,
    out_weights_rewardnet,
    max_episode_steps,
    reward_net_arch,
    expert_data,
    agent_data,
):
    """Training workflow for auto-labelled preference-based RL."""
    use_wandb = wandb.run is not None
    skip_reward_training = False
    # Hyperparameters

    seed = 0

    # Reward model training
    n_epochs_reward_model = 5
    batch_size_reward_model = 4
    lr_reward_model = 0.001
    n_comparisons = 1000  # 5k takes ~3h per epoch with ImpalaCNN
    fragment_length = 40  # max frames per batch ~ 300 to fit on 16GB RAM
    discount_factor = 0.99

    # PPO
    n_total_steps_ppo = 20000
    n_epochs_ppo = 3
    n_steps_ppo = 512
    lr_ppo = 0.000181
    batch_size_ppo = 64
    ent_coef_ppo = 0.01
    # linear lr annealing (p = 1 - steps/n_total_steps)
    lr_schedule = lambda p: lr_ppo * p

    if use_wandb:
        # Setup W&B config
        wandb.config.seed = seed

        wandb.config.reward_net_arch = reward_net_arch
        wandb.config.n_epochs_reward_model = n_epochs_reward_model
        wandb.config.batch_size_reward_model = batch_size_reward_model
        wandb.config.lr_reward_model = lr_reward_model
        wandb.config.n_comparisons = n_comparisons
        wandb.config.fragment_length = fragment_length
        wandb.config.discount_factor = discount_factor

    # Setup logger
    logger = imit_logger.configure(
        "train/wandb/log",
        ["stdout", "log", "json", "wandb"]
    )

    # Setup MineRL environment
    minerl_env_str = "MineRLBasalt" + env_str
    
    env = gym.make(minerl_env_str + "-v0")

    # Setup MineRL agent
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    minerl_agent = MineRLAgent(
        env,
        device=device,
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    minerl_agent.load_weights(in_weights_policy)

    # Freeze all policy params
    for param in minerl_agent.policy.parameters():
        param.requires_grad = False
    # Unfreeze last layer and policy and value head
    for param in minerl_agent.policy.net.lastlayer.parameters():
        param.requires_grad = True
    for param in minerl_agent.policy.pi_head.parameters():
        param.requires_grad = True
    for param in minerl_agent.policy.value_head.parameters():
        param.requires_grad = True

    # Setup MineRL VecEnv to use with SB3 and imitation
    venv = make_vec_env(
        minerl_env_str + "SB3-v0",
        # Keep this at 1 since we are not keeping track of multiple hidden states
        n_envs=1,
        # This should be sufficiently high for the given task
        max_episode_steps=max_episode_steps,
        env_make_kwargs={"minerl_agent": minerl_agent},
    )
    if use_wandb:
        venv = VecVideoRecorder(
            venv,
            f"train/videos/{wandb.run.id}",
            record_video_trigger=lambda x: x % 2000 == 0,
            video_length=400,
        )

    # Define reward model
    image_obs_space = spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
    if reward_net_arch == "CNN":
        reward_net = CnnRewardNet(image_obs_space, venv.action_space, use_action=False)
    elif reward_net_arch == "ImpalaCNN":
        reward_net = ImpalaRewardNet(image_obs_space, venv.action_space)
    else:
        raise ValueError(f"Unknown reward network architecture: {reward_net_arch}")
    reward_net = NormalizedRewardNet(reward_net, RunningNorm)
    if in_weights_rewardnet is not None and os.path.isfile(in_weights_rewardnet):
        reward_net.load_state_dict(th.load(in_weights_rewardnet))
    # Move reward model to GPU if possible
    reward_net.to(device)
    preference_model = preference_comparisons.PreferenceModel(
        reward_net,
        discount_factor=discount_factor
    )

    if not skip_reward_training:

        # Create trajectory generators
        expert_generator = preference_comparisons.FromVideoTrajectoryGenerator(
            expert_data,
            minerl_agent,
            seed=seed,
            custom_logger=logger,
        )
        agent_generator = preference_comparisons.FromVideoTrajectoryGenerator(
            agent_data,
            minerl_agent,
            seed=seed,
            custom_logger=logger,
        )

        # Load dataset
        dataset = preference_comparisons.AutoPreferenceDataset(
            expert_generator,
            agent_generator,
            fragment_length,
            later_fragments_preferred=True,
        )

        # Add comparisons to the dataset
        dataset.push(num_samples=n_comparisons)

    if not skip_reward_training:
        # Setup reward trainer
        reward_trainer = preference_comparisons.BasicRewardTrainer(
            model=reward_net,
            loss=preference_comparisons.CrossEntropyRewardLoss(preference_model),
            epochs=n_epochs_reward_model,
            batch_size=batch_size_reward_model,
            lr=lr_reward_model,
            custom_logger=logger,
        )

        # Train the reward model
        with logger.accumulate_means("reward"):
            reward_trainer.train(dataset)
        reward_loss = logger.name_to_value["mean/reward/loss"]
        reward_accuracy = logger.name_to_value["mean/reward/accuracy"]

        print(f"Reward loss {reward_loss}, reward accuracy: {reward_accuracy}")

    # Endow the environment with the learned reward function
    venv = MineRLRewardVecEnvWrapper(venv, reward_net.predict_processed)
    venv = VecMonitor(venv)

    # Setup RL algorithm
    algorithm = PPO(
        policy=MinecraftActorCriticPolicy,
        policy_kwargs={
            "minerl_agent": minerl_agent,
            "optimizer_class": th.optim.Adam,
            # see https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
            "optimizer_kwargs": {"eps": 1e-5}
        },
        env=venv,
        seed=0,
        gamma=discount_factor,
        n_steps=n_steps_ppo // venv.num_envs,
        batch_size=batch_size_ppo,
        ent_coef=ent_coef_ppo,
        learning_rate=lr_schedule,
        n_epochs=n_epochs_ppo,
        device=device,
        verbose=1,
        tensorboard_log=f"train/runs/{wandb.run.id}" if use_wandb else None,
    )

    # Train agent
    algorithm.learn(
        total_timesteps=n_total_steps_ppo,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"train/models/{wandb.run.id}",
            verbose=2
        ) if use_wandb else None,
    )

    venv.close()

    # Save finetuned policy weights
    state_dict = minerl_agent.policy.state_dict()
    th.save(state_dict, out_weights_policy)

    # Save trained reward network
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
    parser.add_argument(
        "--expert-data",
        type=str,
        help="Path to expert trajectories",
        default="data/MineRLBasaltFindCave-v0",
    )
    parser.add_argument(
        "--agent-data",
        type=str,
        help="Path to agent trajectories",
        default="data/agent/MineRLBasaltFindCave-v0",
    )

    args = parser.parse_args()
    auto_preference_based_RL_train(
        args.env,
        args.in_model,
        args.in_weights_policy,
        args.out_weights_policy,
        args.in_weights_rewardnet,
        args.out_weights_rewardnet,
        args.max_episode_steps,
        args.reward_net_arch,
        args.expert_data,
        args.agent_data,
    )
