import os.path

from datasets import load_dataset
from glob import glob
from tqdm import tqdm


class CodeContestReader:
    def __call__(self, file_path):
        if os.path.exists(file_path):
            data = load_dataset("parquet", data_files=file_path)["train"]
        else:
            data = load_dataset("parquet", data_files=list(glob(file_path)))["train"]
        print(len(data))
        outputs = []
        for item in tqdm(data):
            description = item["description"]
            title = item["name"]
            if item["time_limit"]:
                time_limits = item["time_limit"]["seconds"]
            else:
                time_limits = "N/A"
            if item["memory_limit_bytes"]:
                mem_limits = item["memory_limit_bytes"]
            else:
                mem_limits = "N/A"
            pos_solution = item["solutions"]["solution"]
            neg_solution = item["incorrect_solutions"]["solution"]

            outputs.append({
                "name": title,
                "description": description,
                "time_limit": time_limits,
                "memory_limit": mem_limits,
                "pos_solution": pos_solution,
                "neg_solution": neg_solution,
                "id": f"{title}",
            })

        return outputs


class CodeContestFlatReader:
    def __call__(self, file_path):
        if os.path.exists(file_path):
            data = load_dataset("parquet", data_files=file_path)["train"]
        else:
            data = load_dataset("parquet", data_files=list(glob(file_path)))["train"]
        print(len(data))
        outputs = []
        for item in tqdm(data):
            description = item["description"]
            title = item["name"]
            if item["time_limit"]:
                time_limits = item["time_limit"]["seconds"]
            else:
                time_limits = "N/A"
            if item["memory_limit_bytes"]:
                mem_limits = item["memory_limit_bytes"]
            else:
                mem_limits = "N/A"
            pos_solution = item["solutions"]["solution"]
            neg_solution = item["incorrect_solutions"]["solution"]

            for i, solution in enumerate(pos_solution):
                outputs.append({
                    "name": title,
                    "description": description,
                    "time_limit": time_limits,
                    "memory_limit": mem_limits,
                    "solution": solution,
                    "id": f"{title}_{i}",
                    "value": 1,
                })

            for i, solution in enumerate(neg_solution):
                outputs.append({
                    "name": title,
                    "description": description,
                    "time_limit": time_limits,
                    "memory_limit": mem_limits,
                    "solution": solution,
                    "id": f"{title}_{i}",
                    "value": 0,
                })

        return outputs
