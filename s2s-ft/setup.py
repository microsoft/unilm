from io import open
from setuptools import find_packages, setup


extras = {
    'serving': ['pydantic', 'uvicorn', 'fastapi'],
    'serving-tf': ['pydantic', 'uvicorn', 'fastapi'],
    'serving-torch': ['pydantic', 'uvicorn', 'fastapi', 'torch']
}
extras['all'] = [package for package in extras.values()]

setup(
    name="s2s-ft",
    version="0.0.1",
    author="UniLM Team",
    author_email="unilm@microsoft.com",
    description="Fine-Tuning Bidirectional Transformers for Sequence-to-Sequence Learning",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='Fine-Tuning Bidirectional Transformers for Sequence-to-Sequence Learning',
    license='Apache',
    url="https://github.com/microsoft/unilm/tree/master/s2s-ft",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=['numpy',
                      'boto3',
                      'requests',
                      'tqdm',
                      'regex != 2019.12.17',
                      'sentencepiece',
                      'sacremoses',
                      'tensorboardX',
                      'transformers <= 2.10.0'],
    extras_require=extras,
    python_requires='>=3.5.0',
    classifiers=[
          'Programming Language :: Python :: 3',
    ],
)
