import os 
import logging
from accelerate import Accelerator
from accelerate.logging import get_logger

os.makedirs("logs", exist_ok=True)

accelerator = Accelerator()

logger = get_logger("shared_logger")
if logger.hasHandlers():
    logger.logger.handlers.clear()

file_handler = logging.FileHandler("logs/app.log", mode='w')

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
