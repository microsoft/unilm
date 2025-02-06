import json

from datasets import load_dataset
from typing import Dict


def read_text(file_path: str):
    return open(file_path, encoding="utf-8").read()


def json_read_fn(file_path: str):
    return json.load(open(file_path, encoding="utf-8"))


def hf_datasets_load_fn(**kwargs):
    def func(file_path):
        return load_dataset(file_path, **kwargs)

    return func


def jsonl_read_fn():
    def func(file_path):
        return [json.loads(line) for line in open(file_path, encoding="utf-8").readlines()]

    return func


def compose_message(system_prompt: str = ""):
    def compose_fn(text: str):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        return messages

    return compose_fn


def recompose_template(units: Dict[str, str], compositions: Dict[str, str]) -> Dict[str, str]:
    templates = {}
    for k, v in compositions.items():
        templates[k] = v.format(**units)
    return templates


def compose_template(units: Dict[str, str], composition: str) -> str:
    return composition.format(**units)
