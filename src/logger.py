import logging, logging.handlers
import os


def get_logger(module_name):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s : %(name)s : %(message)s')
    file_handler = logging.FileHandler(f'./logs/{module_name}.log')
    file_handler.setFormatter(formatter)

    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(file_handler)
    return logger