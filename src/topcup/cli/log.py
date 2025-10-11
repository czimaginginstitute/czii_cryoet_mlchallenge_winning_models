# yourtool/log.py
import logging, sys

def setup_logging(verbosity: int):
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(levelname).1s %(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )