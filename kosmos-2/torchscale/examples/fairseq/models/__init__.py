# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import importlib
import os

MODEL_REGISTRY = {}
MODEL_DATACLASS_REGISTRY = {}
ARCH_MODEL_REGISTRY = {}
ARCH_MODEL_NAME_REGISTRY = {}
ARCH_MODEL_INV_REGISTRY = {}
ARCH_CONFIG_REGISTRY = {}

# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("models." + model_name)

        # extra `model_parser` for sphinx
        if model_name in MODEL_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_archs = parser.add_argument_group("Named architectures")
            group_archs.add_argument(
                "--arch", choices=ARCH_MODEL_INV_REGISTRY[model_name]
            )
            group_args = parser.add_argument_group("Additional command-line arguments")
            MODEL_REGISTRY[model_name].add_args(group_args)
            globals()[model_name + "_parser"] = parser
