import os
import sys
import logging


def create_logger(args, name=None):
    handler1 = logging.StreamHandler(stream=sys.stdout)
    handler2 = logging.StreamHandler(stream=sys.stderr)
    handler3 = logging.FileHandler(filename=os.path.join(args.local_results_path, 'cout.txt'), mode="a+")
    logging.basicConfig(level=args.log_level, handlers=[handler1, handler2, handler3])
    # logging.basicConfig(handlers=[handler1, handler2])
    logger = logging.getLogger(__name__ if name is None else 'root')
    logger.setLevel(args.log_level)
    return logger
