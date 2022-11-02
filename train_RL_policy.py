from RL_policy_finetuning import PPO_policy_finetuning

if __name__ == "__main__":
    PPO_policy_finetuning(
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltFindCave.weights",
    )
