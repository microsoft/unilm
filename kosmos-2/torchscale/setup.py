# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from io import open

from setuptools import find_packages, setup

setup(
    name="torchscale",
    version="0.1.1",
    author="TorchScale Team",
    author_email="Shuming.Ma@microsoft.com",
    description="Transformers at any scale",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="Transformers at any scale",
    license="MIT",
    url="https://github.com/msranlp/torchscale",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=["apex", "torch>=1.8", "fairscale==0.4.0", "timm==0.4.12"],
    python_requires=">=3.8.0",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
