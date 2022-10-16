from datetime import datetime
from utils.create_videos import create_videos
from utils.logs import Logging
from utils import create_subfolders

FOUNDATION_MODEL = "foundation-model-1x"
ENVS = ["FindCave", "MakeWaterfall", "CreateVillageAnimalPen", "BuildVillageHouse"]
NUM_VIDEOS = 1
NUM_MAX_STEPS = [3600, 6000, 6000, 14400]

LOG_FILE = f"log_videos_{datetime.now().strftime('%Y:%m:%d_%H:%M:%S')}.log"
create_subfolders.main()
Logging.setup(name=LOG_FILE)

for i, env in enumerate(ENVS):
    Logging.info(f"Creating videos for {env}")
    create_videos(env, FOUNDATION_MODEL, NUM_VIDEOS, NUM_MAX_STEPS[i])