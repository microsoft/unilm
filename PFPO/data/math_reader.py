import json

from datasets import load_dataset


class MetaMathReaderHF:
    def __init__(self, category: str):
        self.category = category

    def __call__(self, file_path: str = "meta-math/MetaMathQA"):
        data = load_dataset(file_path, split="train").to_list()
        for i, data in enumerate(data):
            data["id"] = i

        if not self.category or (self.category and self.category == "all"):
            return data

        if self.category == "math":
            return [item for item in data if "MATH" in item["type"]]
        if self.category == "gsm":
            return [item for item in data if "GSM" in item["type"]]
        raise ValueError(f"Invalid category: {self.category}")


class MetaMathReader:
    def __init__(self, category: str = "math"):
        self.category = category

    def __call__(self, file_path: str = "meta-math/MetaMathQA"):
        data = json.load(open(file_path))

        if not self.category or (self.category and self.category == "all"):
            return data

        if self.category == "math":
            return [item for item in data if "MATH" in item["type"]]
        if self.category == "gsm":
            return [item for item in data if "GSM" in item["type"]]
        raise ValueError(f"Invalid category: {self.category}")

