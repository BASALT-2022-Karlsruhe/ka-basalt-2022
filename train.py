from utils.logs import Logging
from utils import create_subfolders
from behavioural_cloning import behavioural_cloning_train
from preference_based_RL import preference_based_RL_train
from utils.visualizer import visualize_loss
from utils.create_videos import create_videos
from datetime import datetime
import wandb

FOUNDATION_MODEL = "foundation-model-1x"
BC_TRAINING = True
PREFRL_TRAINING = True
ENVS = ["FindCave", "MakeWaterfall",
        "CreateVillageAnimalPen", "BuildVillageHouse"]
NUM_VIDEOS = 1
NUM_MAX_STEPS = [3600, 6000, 6000, 14400]

LOG_FILE = f"log_{datetime.now().strftime('%Y:%m:%d_%H:%M:%S')}.log"

def pre_training():
    """
    executed before training # Add things you want to execute
    """
    wandb.init(project="BASALT")


    create_subfolders.main()
    Logging.setup(name=LOG_FILE)

    Logging.info('Start training')


def post_training():
    """
    executed after training  # Add things you want to execute
    """
    Logging.info("Saving loss plot...")
    visualize_loss(log_file_path=f"/home/aicrowd/train/{LOG_FILE}")
    Logging.info("Creating videos...")
    for i, env in enumerate(ENVS):
        create_videos(env, FOUNDATION_MODEL, NUM_VIDEOS, NUM_MAX_STEPS[i])
    Logging.info("End training")

def main():
    pre_training()

    if BC_TRAINING:
        for env in ENVS:
            Logging.info(f"===BC Training {env} model===")
            behavioural_cloning_train(
                data_dir=f"data/MineRLBasalt{env}-v0",
                in_model=f"data/VPT-models/{FOUNDATION_MODEL}.model",
                in_weights=f"data/VPT-models/{FOUNDATION_MODEL}.weights",
                out_weights=f"train/BehavioralCloning{env}.weights"
            )

    if PREFRL_TRAINING:
        for env in ENVS:
            Logging.info(f"===PrefRL Training {env} model===")
            preference_based_RL_train(
                env_str=env,
                in_model=f"data/VPT-models/{FOUNDATION_MODEL}.model",
                in_weights=f"train/BehavioralCloning{env}.weights",
                out_weights=f"train/PreferenceBasedRL{env}.weights"
            )

    post_training()


if __name__ == "__main__":
    main()
