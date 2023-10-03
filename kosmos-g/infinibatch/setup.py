from setuptools import setup, find_packages
import site

site.ENABLE_USER_SITE = True
setup(
    name='infinibatch',
    version='0.1.0',
    url='https://github.com/microsoft/infinibatch',
    author='Frank Seide',
    author_email='fseide@microsoft.com',
    description='Infinibatch is a library of checkpointable iterators for randomized data loading of massive data sets in deep neural network training.',
    packages=find_packages()
)
