from setuptools import setup, find_packages

setup(
    name="vlmo",
    packages=find_packages(
        exclude=[".dfc", ".vscode", "dataset", "notebooks", "result", "scripts"]
    ),
    version="1.0.0",
    license="MIT",
    description="VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts",
    author="Wenhui Wang",
    author_email="wenwan@microsoft.com",
    url="https://github.com/microsoft/unilm/tree/master/vlmo",
    keywords=["vision and language pretraining"],
    install_requires=["torch", "pytorch_lightning"],
)
