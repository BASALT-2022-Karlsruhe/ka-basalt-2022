import os
import logging


class Logging(object):
    logger = None

    @staticmethod
    def info(message):
        Logging.logger.info(message)

    @staticmethod
    def setup(name, root="train/logs"):
        file_path = os.path.join(root, name)
        logging.basicConfig(
            level=logging.INFO,
            handlers=[logging.FileHandler(file_path), logging.StreamHandler()],
            format="%(asctime)s %(message)s",
        )
        Logging.logger = logging.getLogger()
