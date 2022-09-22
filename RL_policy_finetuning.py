# TODO
# - create MineRLAgent from existing BC pretrained model
# - load MineRLRewardNet trained with human feedback
# - use StableBaselines3 to finetune the agent under the reward model
# - save the resulting MineRLAgent

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

def PPO_policy_finetuning(in_model, in_weights, out_weights):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    env = gym.make("MineRLBasaltFindCave-v0")
    agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)

    # Create a copy which will have the original parameters
    original_agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    original_agent.load_weights(in_weights)

    # TODO feed MineRLAgent into PPO
    #model = PPO("MlpPolicy", env, verbose=1)
    #model.learn(total_timesteps=25000)

    # TODO write new policy params


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be finetuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be finetuned")
    parser.add_argument("--out-weights", required=True, type=str, help="Path where finetuned weights will be saved")

    args = parser.parse_args()
    PPO_policy_finetuning(args.in_model, args.in_weights, args.out_weights)

