from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Project_01",
    version="0.1.0",
    author="Omprakash",
    packages=find_packages(),
    install_requires=requirements
)
 