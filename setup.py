"""Installation script for the 'isaacgymenvs' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # RL
    "gym==0.23.1",
    "torch",
    "omegaconf",
    "termcolor",
    "jinja2",
    "hydra-core>=1.2",
    "rl-games>=1.6.0",
    "pyvirtualdisplay",
    "urdfpy==0.0.22",
    "pysdf==0.1.9",
    "warp-lang==0.10.1",
    "trimesh==3.23.5",
    ]



# Installation operation
setup(
    name="isaacgymenvs",
    author="NVIDIA",
    version="1.5.0",
    description="Benchmark environments for high-speed robot learning in NVIDIA IsaacGym.",
    keywords=["robotics", "rl"],
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.6, 3.7, 3.8"],
    zip_safe=False,
)

# EOF
