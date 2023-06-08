#!/usr/bin/env python3
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "requirements.txt")) as f:
    requirements = f.read().split()

setup(
    name="nn-repo-template",
    version="0.0.0",
    description="A very cool nn repository",
    author="Kai-Hendrik Cohrs",
    author_email="kaicohrs@uv.es",
    # license="MIT",
    url="https://github.com/KaiHCohrs/nn-repo-template",
    # keywords="",
    packages=find_packages(exclude=[]),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English" "Topic :: Scientific/Engineering :: Machine Learning",
    ],
)
