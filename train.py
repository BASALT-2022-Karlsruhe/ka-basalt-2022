"""Train script for MineRL BASALT competition."""
import subprocess
from datetime import datetime

import wandb
from auto_preference_based_RL import auto_preference_based_RL_train
from behavioural_cloning import behavioural_cloning_train
from generate_agent_trajectories import generate_trajectories
from preference_based_RL import preference_based_RL_train
from utils import create_subfolders
from utils.create_videos import create_videos
from utils.logs import Logging

LOG_FILE = f"log_{datetime.now().strftime('%Y:%m:%d_%H:%M:%S')}.log"
USE_WANDB = True
WANDB_LOG_DIR = "train"

BC_PREFIX = "BehavioralCloning"
PREFRL_PREFIX = "PreferenceBasedRL"
ENVS = [
    "FindCave",
    "MakeWaterfall",
    "CreateVillageAnimalPen",
    "BuildVillageHouse"
]

REWARD_NET_ARCHITECTURE = "CNN"

# Model and weights paths
FOUNDATION_MODEL = "foundation-model-2x"
FOUNDATION_MODEL_FILE = "{FOUNDATION_MODEL}.model" if int(FOUNDATION_MODEL.split("-")[-1][0]) != 2 else "2x.model" 

MODEL_PATH = f"data/VPT-models/{FOUNDATION_MODEL_FILE}"

FOUNDATION_WEIGHTS_PATH = f"data/VPT-models/{FOUNDATION_MODEL}.weights"

BC_WEIGHTS_PATH = f"train/{BC_PREFIX}{{}}.weights"
PREFRL_PRETRAINED_POLICY_WEIGHTS_PATH = (
    f"train/{PREFRL_PREFIX}{{}}_PretrainedPolicy.weights"
)
PREFRL_PRETRAINED_REWARDNET_WEIGHTS_PATH = (
    f"train/{PREFRL_PREFIX}{{}}_PretrainedRewardNet.weights"
)
PREFRL_POLICY_WEIGHTS_PATH = f"train/{PREFRL_PREFIX}{{}}_Policy.weights"
PREFRL_REWARDNET_WEIGHTS_PATH = f"train/{PREFRL_PREFIX}{{}}_RewardNet.weights"

ESC_WEIGHTS_PATHS = []

# Weights to be used in the submission
SUBMISSION_WEIGHTS_PATH = PREFRL_PRETRAINED_POLICY_WEIGHTS_PATH

# Trajectory data dirs
EXPERT_DATA_DIR = "data/MineRLBasalt{}-v0"
AGENT_DATA_DIR = f"data/{FOUNDATION_MODEL}/MineRLBasalt{{}}-v0"

# Control training components
BC_TRAINING = True
AGENT_DATA_GENERATION = False
PREFRL_PRETRAINING = False
PREFRL_TRAINING = False

INITIAL_POLICY_WEIGHTS_PATH = FOUNDATION_WEIGHTS_PATH
INITIAL_REWARDNET_WEIGHTS_PATH = ""

# Generation and evaluation parameters
GENERATE_NUM_EPISODES = 10
NUM_EVAL_VIDEOS = 0
NUM_MAX_STEPS = {
    "FindCave": 3600,
    "MakeWaterfall": 6000,
    "CreateVillageAnimalPen": 6000,
    "BuildVillageHouse": 14400,
}


def pre_training():
    """Execute before training."""
    create_subfolders.main()
    Logging.setup(name=LOG_FILE)

    Logging.info("Start training")


def post_training(policy_weights_path):
    """Execute after training.

    Args:
        policy_weights_path (str): Path to trained policy weights
    """
    if NUM_EVAL_VIDEOS > 0:
        for i, env in enumerate(ENVS):
            Logging.info(f"===Creating evaluation videos for {env}===")

            esc_weights_path = None
            try:
                esc_weights_path = ESC_WEIGHTS_PATHS[i]
            except IndexError:
                Logging.info("No ESC model available.")

            create_videos(
                policy_weights_path.format(env),
                esc_weights_path,
                env,
                FOUNDATION_MODEL,
                NUM_EVAL_VIDEOS,
                NUM_MAX_STEPS[env],
                show=False,
            )
    Logging.info("End training")


def main():
    """Run the training pipeline."""
    pre_training()

    next_policy_weights_path = INITIAL_POLICY_WEIGHTS_PATH
    next_rewardnet_weights_path = INITIAL_REWARDNET_WEIGHTS_PATH

    if BC_TRAINING:
        for env in ENVS:
            run = (
                wandb.init(
                    project=f"BC Training {env}",
                    reinit=True,
                    entity="kabasalt_team",
                    dir=WANDB_LOG_DIR,
                )
                if USE_WANDB
                else None
            )
            Logging.info(f"===BC Training {env} model===")
            behavioural_cloning_train(
                data_dir=EXPERT_DATA_DIR.format(env),
                in_model=MODEL_PATH,
                in_weights=next_policy_weights_path.format(env),
                out_weights=BC_WEIGHTS_PATH.format(env),
            )
            if USE_WANDB:
                run.finish()
        next_policy_weights_path = BC_WEIGHTS_PATH

    if AGENT_DATA_GENERATION:
        for i, env in enumerate(ENVS):
            Logging.info(f"===Data generation with {env} model===")

            generate_trajectories(
                MODEL_PATH,
                next_policy_weights_path.format(env),
                env,
                n_episodes=GENERATE_NUM_EPISODES,
                max_steps=NUM_MAX_STEPS[env],
                video_dir=AGENT_DATA_DIR.format(env),
            )

    if PREFRL_PRETRAINING:
        for i, env in enumerate(ENVS):
            run = (
                wandb.init(
                    project=f"PrefRL Pretraining {env}",
                    reinit=True,
                    sync_tensorboard=True,
                    monitor_gym=True,
                    entity="kabasalt_team",
                    dir=WANDB_LOG_DIR,
                )
                if USE_WANDB
                else None
            )
            Logging.info(f"===Reward network Training {env}===")
            auto_preference_based_RL_train(
                env_str=env,
                in_model=MODEL_PATH,
                in_weights_policy=next_policy_weights_path.format(env),
                out_weights_policy=PREFRL_PRETRAINED_POLICY_WEIGHTS_PATH.format(env),
                in_weights_rewardnet=next_rewardnet_weights_path.format(env),
                out_weights_rewardnet=PREFRL_PRETRAINED_REWARDNET_WEIGHTS_PATH.format(
                    env,
                ),
                max_episode_steps=NUM_MAX_STEPS[env],
                reward_net_arch=REWARD_NET_ARCHITECTURE,
                expert_data=EXPERT_DATA_DIR.format(env),
                agent_data=AGENT_DATA_DIR.format(env),
            )
            if USE_WANDB:
                run.finish()
            next_policy_weights_path = PREFRL_PRETRAINED_POLICY_WEIGHTS_PATH
            next_rewardnet_weights_path = PREFRL_PRETRAINED_REWARDNET_WEIGHTS_PATH

    if PREFRL_TRAINING:

        # Start PrefCollect
        proc = subprocess.Popen(
            ["python", "pref-collect/manage.py", "runserver", "0.0.0.0:8000"],
        )

        for i, env in enumerate(ENVS):
            run = (
                wandb.init(
                    project=f"PrefRL Training {env}",
                    reinit=True,
                    sync_tensorboard=True,
                    monitor_gym=True,
                    entity="kabasalt_team",
                    dir=WANDB_LOG_DIR,
                )
                if USE_WANDB
                else None
            )
            Logging.info(f"===PrefRL Training {env} model===")
            preference_based_RL_train(
                env_str=env,
                in_model=MODEL_PATH,
                in_weights_policy=next_policy_weights_path.format(env),
                out_weights_policy=PREFRL_POLICY_WEIGHTS_PATH.format(env),
                in_weights_rewardnet=next_rewardnet_weights_path.format(env),
                out_weights_rewardnet=PREFRL_REWARDNET_WEIGHTS_PATH.format(env),
                max_episode_steps=NUM_MAX_STEPS[env],
                reward_net_arch=REWARD_NET_ARCHITECTURE,
            )
            if USE_WANDB:
                run.finish()

            # Flush PrefCollect's query database
            subprocess.run(["python", "pref-collect/manage.py", "flush"])

        # Stop PrefCollect
        proc.terminate()

        next_policy_weights_path = PREFRL_POLICY_WEIGHTS_PATH
        next_rewardnet_weights_path = PREFRL_REWARDNET_WEIGHTS_PATH

    post_training(next_policy_weights_path)


if __name__ == "__main__":
    from pyvirtualdisplay import Display

    disp = Display().start()
    main()
    disp.stop()
