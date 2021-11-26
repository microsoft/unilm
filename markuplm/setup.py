#!/usr/bin/env python3
from setuptools import find_packages, setup
setup(
    name="markuplmft",
    version="0.1",
    author="MarkupLM Team",
    packages=find_packages(),
    python_requires=">=3.7",
    extras_require={"dev": ["flake8", "isort", "black"]},
)