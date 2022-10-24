import torch as th
import torch.nn as nn
from openai_vpt.lib.impala_cnn import ImpalaCNN
from openai_vpt.lib.util import FanInInitReLULayer
from typing import Dict

import gym
from behavioural_cloning import load_model_parameters
from openai_vpt.agent import MineRLAgent


class ImpalaLinear(nn.Module):
    """ImpalaCNN followed by a linear layer.

    :param cnn_outsize: impala output dimension
    :param output_size: output size of the linear layer.
    :param dense_init_norm_kwargs: kwargs for linear FanInInitReLULayer
    :param init_norm_kwargs: kwargs for 2d and 3d conv FanInInitReLULayer
    """

    def __init__(
        self,
        cnn_outsize: int,
        output_size: int,
        dense_init_norm_kwargs: Dict = {},
        init_norm_kwargs: Dict = {},
        **kwargs,
    ):
        super().__init__()
        self.cnn = ImpalaCNN(
            outsize=cnn_outsize,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **kwargs,
        )
        self.linear = FanInInitReLULayer(
            cnn_outsize,
            output_size,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )

    def forward(self, img):
        return self.linear(self.cnn(img))

    def load_cnn_weights(self, path="data/VPT-models/ImpalaCNN-1x.weights"):
        self.cnn.load_state_dict(th.load(path))


class ImpalaClassifier(nn.Module):
    """Classification network based on ImpalaCNN"""

    def __init__(self, output_size, cnn_outsize=256, hidden_size=512):
        super().__init__()
        self.impala_linear = ImpalaLinear(cnn_outsize, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, obs):
        return self.output_layer(nn.functional.ReLU(self.impala_linear(obs)))


class ImpalaRewardModel(nn.Module):
    """Reward model based on ImpalaCNN"""

    def __init__(self, cnn_outsize=256, hidden_size=512):
        super().__init__()
        self.impala_linear = ImpalaLinear(cnn_outsize, hidden_size)
        self.normalized_reward_layer = nn.Linear(hidden_size, 1)

    def forward(self, obs):
        return self.normalized_reward_layer(nn.functional.ReLU(self.impala_linear(obs)))


def save_impala_cnn_weights(model_width=1):
    """
    Saves ImpalaCNN weights.

    TODO check whether ImpalaCNN even depends on model_width ...
    model_width in [1, 2, 3] according to the three VPT model sizes
    """
    # Setup a MineRL environment
    minerl_env_str = "MineRLBasaltFindCave"
    env = gym.make(minerl_env_str + "-v0")

    # Setup MineRL agent
    in_model = f"data/VPT-models/foundation-model-{model_width}x.model"
    in_weights = f"data/VPT-models/foundation-model-{model_width}x.weights"
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    minerl_agent = MineRLAgent(
        env,
        device="cpu",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    minerl_agent.load_weights(in_weights)

    # Get ImpalaCNN state_dict and save it
    state_dict = minerl_agent.policy.net.img_process.cnn.state_dict()
    th.save(state_dict, f"data/VPT-models/ImpalaCNN-{model_width}x.weights")


def load_impala_cnn_weights(
    model_width=1,
    weights_path="data/VPT-models/ImpalaCNN-1x.weights",
):
    """Load previously saved"""

    # Load state dict
    state_dict = th.load(f"data/VPT-models/ImpalaCNN-{model_width}x.weights")

    # Create model object
    impala_cnn = ImpalaCNN(
        inshape=[128, 128, 3],
        chans=(64, 128, 128),
        outsize=256,
        nblock=2,
        first_conv_norm=False,
        post_pool_groups=1,
        dense_init_norm_kwargs={"batch_norm": False, "layer_norm": True},
        init_norm_kwargs={"batch_norm": False, "group_norm_groups": 1},
    )

    # Load state dict into model
    impala_cnn.load_state_dict(state_dict)

    return impala_cnn
