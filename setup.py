from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, "r") as f:
        return f.read().splitlines()


setup(
    name="gvs",
    version="0.0.1",
    description="Generative View Stitching",
    packages=find_packages(),
    install_requires=parse_requirements("./requirements.txt"),
)
