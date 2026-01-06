from setuptools import find_packages
from setuptools import setup

setup(
    name='bfmc_interfaces',
    version='0.0.0',
    packages=find_packages(
        include=('bfmc_interfaces', 'bfmc_interfaces.*')),
)
