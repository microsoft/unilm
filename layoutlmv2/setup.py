#!/usr/bin/env python3
from setuptools import find_packages, setup
setup(
    name="layoutlm",
    version="0.1.1",
    author="Yang Xu and Yiheng Xu",
    url="https://github.com/microsoft/unilm/tree/master/layoutlmv2",
    description="Code for LayoutLMv2",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.7",
    extras_require={"dev": ["flake8", "isort", "black"]},
)
