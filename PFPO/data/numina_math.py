import json
import sys

from datasets import load_dataset
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class NuminaMathReader:
    def __init__(self, id_field: str = "id", split: str = "train"):
        self.id_field = id_field
        self.split = split

    def __call__(self, file_path):
        data = load_dataset(file_path, split=self.split).to_list()
        logger.info(len(data))

        for i, item in enumerate(data):
            item[self.id_field] = i

        return data
