import os
from datetime import datetime

import wandb

from utils import create_subfolders
from behavioural_cloning import behavioural_cloning_train
from preference_based_RL import preference_based_RL_train

from utils.logs import Logging
from utils.create_videos import create_videos
#from utils.visualizer import visualize_loss

FOUNDATION_MODEL = "foundation-model-1x"
BC_TRAINING = True
PREFRL_TRAINING = False
ENVS = ["FindCave", "MakeWaterfall",
        "CreateVillageAnimalPen", "BuildVillageHouse"]
NUM_VIDEOS = 1
NUM_MAX_STEPS = [3600, 6000, 6000, 14400]

LOG_FILE = f"log_{datetime.now().strftime('%Y:%m:%d_%H:%M:%S')}.log"



epochs=(1,)
batch_sizes=(16,32,64)
n_workers=(50,75,100)
learning_rates=(0.001,0.000181,0.00001)
weight_decays=(0.0,0.01)
kl_loss_weights=(0.0,)
max_batches=(1000,2700,5000)


def pre_training():
    """
    executed before training # Add things you want to execute
    """


    create_subfolders.main()
    Logging.setup(name=LOG_FILE)

    Logging.info('Start training')


def post_training():
    """
    executed after training  # Add things you want to execute
    """
    # Logging.info("Saving loss plot...")
    # visualize_loss(log_file_path=f"/home/aicrowd/train/{LOG_FILE}")

    Logging.info("Creating videos...")
    for i, env in enumerate(ENVS):
        create_videos(env, FOUNDATION_MODEL, NUM_VIDEOS, NUM_MAX_STEPS[i])
    Logging.info("End training")

def main():
    for epoch in epochs:
        for batch_size in batch_sizes:
            for n_worker in n_workers:
                if n_worker > batch_size:
                    for learning_rate in learning_rates:
                        for weight_decay in weight_decays:
                            for kl_loss_weight in kl_loss_weights:
                                for max_batch in max_batches:
                                    os.environ["EPOCHS"]=str(epoch)
                                    os.environ["BATCH_SIZE"]=str(batch_size)
                                    os.environ["N_WORKERS"]=str(n_worker)
                                    os.environ["LEARNING_RATE"]=str(learning_rate)
                                    os.environ["WEIGHT_DECAY"]=str(weight_decay)
                                    os.environ["KL_LOSS_WEIGHT"]=str(kl_loss_weight)
                                    os.environ["MAX_BATCHES"]=str(max_batch)
                                    try:
                                        pre_training()
                                        if BC_TRAINING:
                                            for env in ENVS:
                                                run = wandb.init(project=f"BC Training {env}", reinit=True, entity="kabasalt_team")
                                                Logging.info(f"===BC Training {env} model===")
                                                behavioural_cloning_train(
                                                    data_dir=f"data/MineRLBasalt{env}-v0",
                                                    in_model=f"data/VPT-models/{FOUNDATION_MODEL}.model",
                                                    in_weights=f"data/VPT-models/{FOUNDATION_MODEL}.weights",
                                                    out_weights=f"train/BehavioralCloning{env}.weights"
                                                )
                                                run.finish()

                                        if PREFRL_TRAINING:
                                            for env in ENVS:
                                                run = wandb.init(project=f"PrefRL Training {env}", reinit=True, entity="kabasalt_team")
                                                Logging.info(f"===PrefRL Training {env} model===")
                                                preference_based_RL_train(
                                                    env_str=env,
                                                    in_model=f"data/VPT-models/{FOUNDATION_MODEL}.model",
                                                    in_weights=f"train/BehavioralCloning{env}.weights",
                                                    out_weights=f"train/PreferenceBasedRL{env}.weights"
                                                )
                                                run.finish()
                                        post_training()
                                    except:
                                        pass
    print('======== FINISHED =========')


if __name__ == "__main__":
    main()
