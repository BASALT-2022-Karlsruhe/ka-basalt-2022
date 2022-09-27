from utils.logs import Logging
from utils import create_subfolders
# Train one model for each task
from behavioural_cloning import behavioural_cloning_train

def pretraining():
    """
    executed before training # Add things you want to execute
    """
    create_subfolders.main()
    Logging.setup()

    Logging.info('Start training')

def posttraining():
    """
    executed after training  # Add things you want to execute
    """
    Logging.info("End training")

def main():
    pretraining()

    Logging.info("===Training FindCave model===")
    behavioural_cloning_train(
        data_dir="data/MineRLBasaltFindCave-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltFindCave.weights"
    )

    Logging.info("===Training MakeWaterfall model===")
    behavioural_cloning_train(
        data_dir="data/MineRLBasaltMakeWaterfall-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltMakeWaterfall.weights"
    )

    Logging.info("===Training CreateVillageAnimalPen model===")
    behavioural_cloning_train(
        data_dir="data/MineRLBasaltCreateVillageAnimalPen-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltCreateVillageAnimalPen.weights"
    )

    Logging.info("===Training BuildVillageHouse model===")
    behavioural_cloning_train(
        data_dir="data/MineRLBasaltBuildVillageHouse-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltBuildVillageHouse.weights"
    )

    posttraining()

if __name__ == "__main__":
    main()
