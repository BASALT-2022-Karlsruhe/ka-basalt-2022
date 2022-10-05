# Train one model for each task
from behavioural_cloning import behavioural_cloning_train
from preference_based_RL import preference_based_RL_train

FOUNDATION_MODEL = "foundation-model-1x"
BC_TRAINING = True
PREFRL_TRAINING = True
ENVS = ["FindCave", "MakeWaterfall",
        "CreateVillageAnimalPen", "BuildVillageHouse"]


def main():

    if BC_TRAINING:
        for env in ENVS:
            print(f"===BC Training {env} model===")
            behavioural_cloning_train(
                data_dir=f"data/MineRLBasalt{env}-v0",
                in_model=f"data/VPT-models/{FOUNDATION_MODEL}.model",
                in_weights=f"data/VPT-models/{FOUNDATION_MODEL}.weights",
                out_weights=f"train/BehavioralCloning{env}.weights"
            )

    if PREFRL_TRAINING:
        for env in ENVS:
            print(f"===PrefRL Training {env} model===")
            preference_based_RL_train(
                env_str=env,
                in_model=f"data/VPT-models/{FOUNDATION_MODEL}.model",
                in_weights=f"train/BehavioralCloning{env}.weights",
                out_weights=f"train/PreferenceBasedRL{env}.weights"
            )


if __name__ == "__main__":
    main()
