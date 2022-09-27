import logging
from datetime import datetime


class Logging(object):
  logger = None

  @staticmethod
  def info(message):
      Logging.logger.info(message)

  @staticmethod
  def setup():
      logging.basicConfig(level=logging.INFO,
                          handlers=[
                              logging.FileHandler(
                                  f"/home/aicrowd/train/logs/log_{datetime.now().strftime('%Y:%m:%d_%H:%M:%S')}.log"),
                              logging.StreamHandler()
                          ],
                          format='%(asctime)s %(message)s'
                          )
      Logging.logger = logging.getLogger()

