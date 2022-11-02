from datetime import datetime

from utils import create_subfolders
from utils.create_videos import create_videos
from utils.logs import Logging

FOUNDATION_MODEL = "foundation-model-1x"
BC_MODEL = False
PREF_PRETRAINING = True
PREF_RL = False
ENVS = ["FindCave"]#, "MakeWaterfall", "CreateVillageAnimalPen", "BuildVillageHouse"]
ESC_MODELS = []
NUM_VIDEOS = 3
NUM_MAX_STEPS = [3600, 6000, 6000, 14400]

LOG_FILE = f"log_videos_{datetime.now().strftime('%Y:%m:%d_%H:%M:%S')}.log"
create_subfolders.main()
Logging.setup(name=LOG_FILE)

model = f"data/VPT-models/{FOUNDATION_MODEL}.weights"

for i, env in enumerate(ENVS):
    Logging.info(f"Creating videos for {env}")
    esc_model = None
    try:
        esc_model = f"train/{ESC_MODELS[i]}.weights"
    except IndexError:
        Logging.info("No ESC model available.")

    if BC_MODEL:
        model = f"train/BehavioralCloning{env}.weights"
    if PREF_PRETRAINING:
        model = f"train/PreferenceBasedRL{env}_PretrainedPolicy.weights"
    if PREF_RL:
        model = f"train/PreferenceBasedRL{env}_Policy.weights"

    Logging.info(f"Using policy from at {model}")

    create_videos(
        model, esc_model, env, FOUNDATION_MODEL, NUM_VIDEOS, NUM_MAX_STEPS[i], show=False
    )
