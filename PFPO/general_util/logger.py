import logging
import os
import sys
from torch import distributed as dist

_root_name = 'FK'


def get_child_logger(child_name):
    # _local_rank = getattr(os.environ, "LOCAL_RANK", "")
    #
    # if _root_name == "FK" and _local_rank:
    #     return logging.getLogger(_root_name + '.' + _local_rank + '.' + child_name)

    return logging.getLogger(_root_name + '.' + child_name)


def setting_logger(log_file: str, local_rank: int = -1):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if local_rank in [-1, 0] else logging.WARNING)

    # global _root_name
    # if local_rank != -1 and _root_name == "FK":
    #     _root_name = _root_name + '.' + str(local_rank)
    logger = logging.getLogger(_root_name)
    logger.setLevel(logging.INFO if local_rank in [-1, 0] else logging.WARNING)

    rf_handler = logging.StreamHandler(sys.stderr)
    rf_handler.setLevel(logging.INFO)
    rf_handler.setFormatter(logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                              datefmt='%m/%d/%Y %H:%M:%S'))

    output_dir = './log_dir'
    if local_rank not in [-1, 0]:
        dist.barrier()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if local_rank == 0:
        dist.barrier()

    if log_file:
        model_name = "-".join(log_file.replace('/', ' ').split()[1:])
        f_handler = logging.FileHandler(os.path.join(
            output_dir, model_name + '-output.log'))
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                                 datefmt='%m/%d/%Y %H:%M:%S'))

        logger.addHandler(f_handler)

    return logger
