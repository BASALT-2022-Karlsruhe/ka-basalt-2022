import logging


class Logging(object):
    logger = None

    @staticmethod
    def info(message):
        Logging.logger.info(message)

    @staticmethod
    def setup(name):
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(f"/home/aicrowd/train/logs/{name}"),
                logging.StreamHandler(),
            ],
            format="%(asctime)s %(message)s",
        )
        Logging.logger = logging.getLogger()
