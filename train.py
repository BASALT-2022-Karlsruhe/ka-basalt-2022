import logging
import os
# Train one model for each task
from behavioural_cloning import behavioural_cloning_train
from datetime import datetime
# datetime object containing current date and time
now = datetime.now()

# TODO: move create dir and loggin to own files / classes
# Create folders
# TODO: Move to utilils file or so
def create_subfolder(subfolder):
    path = f"/home/aicrowd/train/{subfolder}"
    if not os.path.exists(path):
        os.mkdir(path)

create_subfolder("videos")
create_subfolder("logs")
create_subfolder("reports")

print("Directories created")

logging.basicConfig(filename="/home/aicrowd/train/logs/training.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()

def main():
    logger.info('Start training')
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Start training - date and time =", dt_string)

    print("===Training FindCave model===")
    logger.info("===Training FindCave model===")

    behavioural_cloning_train(
        data_dir="data/MineRLBasaltFindCave-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltFindCave.weights"
    )

    print("===Training MakeWaterfall model===")
    logger.info("===Training MakeWaterfall model===")

    behavioural_cloning_train(
        data_dir="data/MineRLBasaltMakeWaterfall-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltMakeWaterfall.weights"
    )

    print("===Training CreateVillageAnimalPen model===")
    logger.info("===Training CreateVillageAnimalPen model===")

    behavioural_cloning_train(
        data_dir="data/MineRLBasaltCreateVillageAnimalPen-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltCreateVillageAnimalPen.weights"
    )

    print("===Training BuildVillageHouse model===")
    logger.info("===Training BuildVillageHouse model===")

    behavioural_cloning_train(
        data_dir="data/MineRLBasaltBuildVillageHouse-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltBuildVillageHouse.weights"
    )

    logger.info("End training")
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("End training - date and time =", dt_string)

if __name__ == "__main__":
    main()
