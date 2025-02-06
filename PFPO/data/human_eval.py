import json
import sys

from datasets import load_dataset
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

sys.set_int_max_str_digits(0)


class HumanEvalReader:
    def __call__(self, file_path: str = "openai_humaneval"):
        data = load_dataset(file_path, split="test").to_list()
        logger.info(len(data))

        return data


class MBPPReader:
    def __init__(self, sanitized: bool = True):
        self.sanitized = sanitized

    def __call__(self, file_path: str = "mbpp"):
        if self.sanitized:
            dataset = load_dataset("mbpp", "sanitized", split="test").to_list()
        else:
            dataset = load_dataset("mbpp", split="test").to_list()

        for item in dataset:
            item["test_list"] = "\n".join(item["test_list"])

        return dataset


class MBPPReaderFixed:
    def __init__(self, sanitized: bool = True):
        self.sanitized = sanitized

    def __call__(self, file_path: str = "mbpp"):
        if self.sanitized:
            dataset = load_dataset("mbpp", "sanitized", split="test").to_list()
        else:
            dataset = load_dataset("mbpp", split="test").to_list()

        for item in dataset:
            item["test_list"] = "\n".join(item["test_list"])

        return dataset
