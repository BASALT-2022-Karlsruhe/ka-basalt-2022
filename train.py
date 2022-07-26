from datetime import datetime
import subprocess

import wandb

from utils import create_subfolders
from behavioural_cloning import behavioural_cloning_train
from preference_based_RL import preference_based_RL_train

from utils.logs import Logging
from utils.create_videos import create_videos


FOUNDATION_MODEL = "foundation-model-1x"

BC_TRAINING = True
PREFRL_TRAINING = False

ENVS = ["FindCave", "MakeWaterfall", "CreateVillageAnimalPen", "BuildVillageHouse"]

ESC_MODELS = []
NUM_VIDEOS = 5
NUM_MAX_STEPS = [3600, 6000, 6000, 14400]

LOG_FILE = f"log_{datetime.now().strftime('%Y:%m:%d_%H:%M:%S')}.log"


def pre_training():
    """
    executed before training # Add things you want to execute
    """

    create_subfolders.main()
    Logging.setup(name=LOG_FILE)

    Logging.info("Start training")


def post_training():
    """
    executed after training  # Add things you want to execute
    """

    Logging.info("Creating videos...")
    for i, env in enumerate(ENVS):
        model = f"data/VPT-models/{FOUNDATION_MODEL}.weights"
        if BC_TRAINING:
            model = f"train/BehavioralCloning{env}.weights"
        elif PREFRL_TRAINING:
            model = f"train/PreferenceBasedRL{env}.weights"

        esc_model = None
        try:
            esc_model = f"train/{ESC_MODELS[i]}.weights"
        except:
            print("No ESC model available.")
        create_videos(
            model, esc_model, env, FOUNDATION_MODEL, NUM_VIDEOS, NUM_MAX_STEPS[i], show=False
        )
    Logging.info("End training")


def main():
    pre_training()

    if BC_TRAINING:
        for env in ENVS:
            run = wandb.init(
                project=f"BC Training {env}", reinit=True, entity="kabasalt_team"
            )
            Logging.info(f"===BC Training {env} model===")
            behavioural_cloning_train(
                data_dir=f"data/MineRLBasalt{env}-v0",
                in_model=f"data/VPT-models/{FOUNDATION_MODEL}.model",
                in_weights=f"data/VPT-models/{FOUNDATION_MODEL}.weights",
                out_weights=f"train/BehavioralCloning{env}.weights",
            )
            run.finish()

    if PREFRL_TRAINING:
        # start PrefCollect
        proc = subprocess.Popen(
            ["python", "pref-collect/manage.py", "runserver", "0.0.0.0:8000"],
        )

        for env in ENVS:
            run = wandb.init(
                project=f"PrefRL Training {env}",
                reinit=True,
                sync_tensorboard=True,
                monitor_gym=True,
                entity="kabasalt_team",
            )
            Logging.info(f"===PrefRL Training {env} model===")
            preference_based_RL_train(
                env_str=env,
                in_model=f"data/VPT-models/{FOUNDATION_MODEL}.model",
                in_weights=f"train/BehavioralCloning{env}.weights"
                if BC_TRAINING
                else f"data/VPT-models/{FOUNDATION_MODEL}.weights",
                out_weights=f"train/PreferenceBasedRL{env}.weights",
            )
            run.finish()
            # Flush query database
            subprocess.run(["python", "pref-collect/manage.py", "flush"])

        # stop PrefCollect
        proc.terminate()

    post_training()


if __name__ == "__main__":
    main()
