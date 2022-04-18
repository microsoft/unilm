#!/usr/bin/env python3
from setuptools import find_packages, setup
setup(
    name="layoutlmv3",
    version="0.1",
    author="LayoutLM Team",
    url="https://github.com/microsoft/unilm/tree/master/layoutlmv3",
    packages=find_packages(),
    python_requires=">=3.7",
    extras_require={"dev": ["flake8", "isort", "black"]},
)