

@abc.abstract
def train_reward(
    data_dir="data/MineRLBasaltFindCave-v0",
    in_model="data/VPT-models/foundation-model-1x.model",
    in_weights="data/VPT-models/foundation-model-1x.weights",
    out_model="reward_model",
):


@abc.abstract
def train_from_reward(
    data_dir="data/MineRLBasaltFindCave-v0",
    in_model="data/VPT-models/foundation-model-1x.model",
    in_weights="data/VPT-models/foundation-model-1x.weights",
    out_weights="train/MineRLBasaltFindCave.weights",
    reward_model="",
    env="",
):
    return None