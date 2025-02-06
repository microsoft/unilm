import json
import sys

from datasets import load_dataset
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

sys.set_int_max_str_digits(0)


class APPsReader:
    def __init__(self, split: str = "train", train_sub_split: str = ""):
        self.train_sub_val_ids = set(json.load(open("apps_train_sub_val_ids.json")))
        self.split = split
        self.train_sub_split = train_sub_split

    def __call__(self, file_path):
        data = load_dataset(file_path, split=self.split, trust_remote_code=True).to_list()
        logger.info(len(data))

        if self.split == "train" and self.train_sub_split:
            if self.train_sub_split == "train":
                data = [item for item in data if item["problem_id"] not in self.train_sub_val_ids]
            elif self.train_sub_split == "val":
                data = [item for item in data if item["problem_id"] in self.train_sub_val_ids]
            else:
                raise ValueError(f"Invalid train_sub_split: {self.train_sub_split} [train, val]")
            logger.info(f"Using {self.train_sub_split} split for training data")
            logger.info(len(data))

        missing_solutions = 0
        missing_test_cases = 0
        for item in data:
            if item["solutions"]:
                item["solutions"] = json.loads(item["solutions"])
            else:
                missing_solutions += 1

            if item["input_output"]:
                item["input_output"] = json.loads(item["input_output"])
            else:
                missing_test_cases += 1

        print(f"Missing solutions: {missing_solutions}")
        print(f"Missing test cases: {missing_test_cases}")

        return data


class APPsWithFunctionName:
    def __init__(self, split: str = "train", train_sub_split: str = "", use_starter_code: bool = False):
        self.train_sub_val_ids = set(json.load(open("apps_train_sub_val_ids.json")))
        self.split = split
        self.train_sub_split = train_sub_split
        self.use_starter_code = use_starter_code

    def __call__(self, file_path):
        data = load_dataset(file_path, split=self.split, trust_remote_code=True).to_list()
        logger.info(len(data))

        if self.split == "train" and self.train_sub_split:
            if self.train_sub_split == "train":
                data = [item for item in data if item["problem_id"] not in self.train_sub_val_ids]
            elif self.train_sub_split == "val":
                data = [item for item in data if item["problem_id"] in self.train_sub_val_ids]
            else:
                raise ValueError(f"Invalid train_sub_split: {self.train_sub_split} [train, val]")
            logger.info(f"Using {self.train_sub_split} split for training data")
            logger.info(len(data))

        missing_solutions = 0
        missing_test_cases = 0
        for item in data:
            if item["solutions"]:
                item["solutions"] = json.loads(item["solutions"])
            else:
                missing_solutions += 1

            if item["input_output"]:
                item["input_output"] = json.loads(item["input_output"])
                if "fn_name" in item["input_output"]:
                    item["fn_name"] = item["input_output"]["fn_name"]
                    if self.use_starter_code:
                        assert item["starter_code"]
                        item["question"] += f"\n\n{item['starter_code']}"
                    else:
                        item["question"] += f"\n\nYou should name the function as `{item['fn_name']}`."
            else:
                missing_test_cases += 1

        print(f"Missing solutions: {missing_solutions}")
        print(f"Missing test cases: {missing_test_cases}")

        return data


class APPsFlatTestCasesReader(APPsWithFunctionName):
    def __call__(self, file_path):
        data = super().__call__(file_path)
        outputs = []
        for item in data:
            if item["input_output"]:
                inputs = item["input_output"]["inputs"]
                inputs = [str(_input) for _input in inputs]
                item["test_inputs"] = inputs
                outputs.append(item)

        return outputs


class PseudoInputsWithFunctionName:
    def __init__(self, use_starter_code: bool = False, train_sub_split: str = "train"):
        self.use_starter_code = use_starter_code
        self.train_sub_val_ids = set([f"apps-train-{x}" for x in json.load(open("apps_train_sub_val_ids.json"))])
        self.train_sub_split = train_sub_split

    def __call__(self, file_path):
        data = json.load(open(file_path, encoding="utf-8"))
        logger.info(len(data))

        if self.train_sub_split == "train":
            data = [item for item in data if item["problem_id"] not in self.train_sub_val_ids]
        elif self.train_sub_split == "val":
            data = [item for item in data if item["problem_id"] in self.train_sub_val_ids]
        else:
            raise ValueError(f"Invalid train_sub_split: {self.train_sub_split} [train, val]")
        logger.info(f"Using {self.train_sub_split} split for training data")
        logger.info(len(data))

        missing_solutions = 0
        missing_test_cases = 0
        for item in data:
            if item["input_output"]:
                if not isinstance(item["input_output"], dict):
                    item["input_output"] = json.loads(item["input_output"])
                if "fn_name" in item["input_output"]:
                    item["fn_name"] = item["input_output"]["fn_name"]
                    if self.use_starter_code:
                        # FIXME: This is a bug for synthesized data, since we didn't generate starter code.
                        #   We should use `fn_name` to compose simple starter_code instead.
                        #   See PseudoInputsWithFunctionNameFixStarterCode for the fix.
                        if "starter_code" in item and item["starter_code"]:
                            item["question"] += f"\n\n{item['starter_code']}"
                    else:
                        item["question"] += f"\n\nYou should name the function as `{item['fn_name']}`."
            else:
                missing_test_cases += 1

        print(f"Missing solutions: {missing_solutions}")
        print(f"Missing test cases: {missing_test_cases}")

        return data


class PseudoInputsWithFunctionNameFixStarterCode:
    def __init__(self, use_starter_code: bool = False, train_sub_split: str = "train"):
        self.use_starter_code = use_starter_code
        self.train_sub_val_ids = set([f"apps-train-{x}" for x in json.load(open("apps_train_sub_val_ids.json"))])
        self.train_sub_split = train_sub_split

    def __call__(self, file_path):
        data = json.load(open(file_path, encoding="utf-8"))
        logger.info(len(data))

        if self.train_sub_split == "train":
            data = [item for item in data if item["problem_id"] not in self.train_sub_val_ids]
        elif self.train_sub_split == "val":
            data = [item for item in data if item["problem_id"] in self.train_sub_val_ids]
        else:
            raise ValueError(f"Invalid train_sub_split: {self.train_sub_split} [train, val]")
        logger.info(f"Using {self.train_sub_split} split for training data")
        logger.info(len(data))

        missing_solutions = 0
        missing_test_cases = 0
        for item in data:
            if item["input_output"]:
                if not isinstance(item["input_output"], dict):
                    item["input_output"] = json.loads(item["input_output"])
                if "fn_name" in item["input_output"]:
                    item["fn_name"] = item["input_output"]["fn_name"]
                    if self.use_starter_code and "starter_code" in item and item["starter_code"]:
                        item["question"] += f"\n\n{item['starter_code']}"
                    else:
                        item["question"] += f"\n\nYou should name the function as `{item['fn_name']}`."
            else:
                missing_test_cases += 1

        print(f"Missing solutions: {missing_solutions}")
        print(f"Missing test cases: {missing_test_cases}")

        return data
