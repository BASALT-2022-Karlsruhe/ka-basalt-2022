# TODO
# - load trajectory pairs and gathered preferences from given path
# - load or initialize reward model
# - train the reward model with minibatch stochastic gradient descent, for each minibatch
#   - feed observation-action-pairs through the reward model to receive a reward trajectory
#   - use the Bradly-Terry model to compute the estimated preferences
#   - compute loss using the gathered preferences
#   - do the backward pass to compute gradients
#   - take a gradient step
# - save the updated reward model


# Basic learning from human preferences using imitation

from argparse import ArgumentParser
import pickle
import time

import gym
import minerl
import torch as th
import numpy as np

from openai_vpt.agent import PI_HEAD_KWARGS, MineRLAgent
from data_loader import DataLoader
from openai_vpt.lib.tree_util import tree_map

from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import RewardNet, BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor

# Questions:
# - should the reward model depend on observations _and_ actions?
#   - we need to parse th .jsonl file and encode actions appropriately
#   - we need to pass frames through a neural network
#   - what is an observation? what is an action? what is the frame resolution that we want to assign preferences on?
#   - we could summarize whole video clips into observations/actions
# - are we able to use the trajectory objects provided by imitation? e.g. TrajectoryWitRew
#    - it seems like we can if we transfer the data to numpy arrays.. but is this efficient? we could also build these clases with pytorch compaibility
# - we can also instead of building a PreferenceGatherer use the PreferenceDataset which we populate ourselves

class MineRLTrajectoryDataet(preference_comparisons.TrajectoryDataset):
    pass

class MineRLRewardNet(RewardNet):
    passs

class MineRLFragmenter(preference_comparisons.Fragmenter):
    pass

class MineRLPreferenceGatherer(preference_comparisons.PreferenceGatherer):
    pass


# Originally this code was designed for a small dataset of ~20 demonstrations per task.
# The settings might not be the best for the full BASALT dataset (thousands of demonstrations).
# Use this flag to switch between the two settings
USING_FULL_DATASET = True

EPOCHS = 1 if USING_FULL_DATASET else 2
# Needs to be <= number of videos
BATCH_SIZE = 8 if USING_FULL_DATASET else 8
# Ideally more than batch size to create
# variation in datasets (otherwise, you will
# get a bunch of consecutive samples)
# Decrease this (and batch_size) if you run out of memory
N_WORKERS = 8 if USING_FULL_DATASET else 8
DEVICE = "cuda"

LOSS_REPORT_RATE = 100

# Tuned with bit of trial and error
LEARNING_RATE = 0.000181
# OpenAI VPT BC weight decay
# WEIGHT_DECAY = 0.039428
WEIGHT_DECAY = 0.0
# KL loss to the original model was not used in OpenAI VPT
KL_LOSS_WEIGHT = 1.0
MAX_GRAD_NORM = 5.0

MAX_BATCHES = 2000 if USING_FULL_DATASET else int(1e9)

def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def preference_learning_train(data_dir, in_model, in_weights, out_weights):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    env = gym.make("MineRLBasaltFindCave-v0")
    agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)

    # Create a copy which will have the original parameters
    original_agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    original_agent.load_weights(in_weights)

    # Create reward network
    reward_net = BasicRewardNet(
        env.observation_space, env.action_space, normalize_input_layer=RunningNorm
    )

    env.close()

    # Setup preference comparison objects
    fragmenter = preference_comparisons.RandomFragmenter(warning_threshold=0, seed=0)

    gatherer = preference_comparisons.HumanGatherer(seed=0)

    preference_model = preference_comparisons.PreferenceModel(reward_net)

    reward_trainer = preference_comparisons.BasicRewardTrainer(
        model=reward_net,
        loss=preference_comparisons.CrossEntropyRewardLoss(preference_model),
        epochs=3,
    )

    trajectory_generator = preference_comparisons.(
        algorithm=original_agent,
        reward_fn=reward_net,
        venv=venv,
        exploration_frac=0.0,
        seed=0,
    )

    pref_comparisons = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        num_iterations=5,
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        fragment_length=10,
        transition_oversampling=1,
        initial_comparison_frac=0.1,
        allow_variable_horizon=True,
        seed=0,
        initial_epoch_multiplier=1,
    )

    policy = agent.policy
    original_policy = original_agent.policy

    # Freeze most params if using small dataset
    for param in policy.parameters():
        param.requires_grad = False
    # Unfreeze final layers
    trainable_parameters = []
    for param in policy.net.lastlayer.parameters():
        param.requires_grad = True
        trainable_parameters.append(param)
    for param in policy.pi_head.parameters():
        param.requires_grad = True
        trainable_parameters.append(param)

    # Parameters taken from the OpenAI VPT paper
    optimizer = th.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS,
    )

    start_time = time.time()

    # Keep track of the hidden state per episode/trajectory.
    # DataLoader provides unique id for each episode, which will
    # be different even for the same trajectory when it is loaded
    # up again
    episode_hidden_states = {}
    dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)

    loss_sum = 0
    for batch_i, (batch_images, batch_actions, batch_episode_id) in enumerate(data_loader):
        batch_loss = 0
        for image, action, episode_id in zip(batch_images, batch_actions, batch_episode_id):
            if image is None and action is None:
                # A work-item was done. Remove hidden state
                if episode_id in episode_hidden_states:
                    removed_hidden_state = episode_hidden_states.pop(episode_id)
                    del removed_hidden_state
                continue

            agent_action = agent._env_action_to_agent(action, to_torch=True, check_if_null=True)
            if agent_action is None:
                # Action was null
                continue

            agent_obs = agent._env_obs_to_agent({"pov": image})
            if episode_id not in episode_hidden_states:
                episode_hidden_states[episode_id] = policy.initial_state(1)
            agent_state = episode_hidden_states[episode_id]

            pi_distribution, _, new_agent_state = policy.get_output_for_observation(
                agent_obs,
                agent_state,
                dummy_first
            )

            with th.no_grad():
                original_pi_distribution, _, _ = original_policy.get_output_for_observation(
                    agent_obs,
                    agent_state,
                    dummy_first
                )

            log_prob  = policy.get_logprob_of_action(pi_distribution, agent_action)
            kl_div = policy.get_kl_of_action_dists(pi_distribution, original_pi_distribution)

            # Make sure we do not try to backprop through sequence
            # (fails with current accumulation)
            new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
            episode_hidden_states[episode_id] = new_agent_state

            # Finally, update the agent to increase the probability of the
            # taken action.
            # Remember to take mean over batch losses
            loss = (-log_prob + KL_LOSS_WEIGHT * kl_div) / BATCH_SIZE
            batch_loss += loss.item()
            loss.backward()

        th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        loss_sum += batch_loss
        if batch_i % LOSS_REPORT_RATE == 0:
            time_since_start = time.time() - start_time
            print(f"Time: {time_since_start:.2f}, Batches: {batch_i}, Avrg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
            loss_sum = 0

        if batch_i > MAX_BATCHES:
            break

    state_dict = policy.state_dict()
    th.save(state_dict, out_weights)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the directory containing recordings and preferences to be trained on")
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be finetuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be finetuned")
    parser.add_argument("--out-weights", required=True, type=str, help="Path where finetuned weights will be saved")

    args = parser.parse_args()
    learning_from_preferences_train(args.data_dir, args.in_model, args.in_weights, args.out_weights)
