from setuptools import setup, find_packages

setup(
    name = "adalm",
    version = "0.0",
    author = "Microsoft",
    author_email = "",
    description = "domain adaptation toolkit",
    keywords = "domain adaptation with extended vocab",
    license='Apache',

    url = "https://github.com/littlefive5/AdaLM",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=['numpy',
                      'boto3',
                      'requests',
                      'tqdm',
                      'urllib3==1.26.5'],
    
    
    python_requires='>=3.5.0',
    tests_require=['pytest'],
    classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

)