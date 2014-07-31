import os
from setuptools import setup
from distutils.extension import Extension

setup(
    name="kitti",
    version="0.0.1",
    author="Eric Hunsberger",
    author_email="ehunsber@uwaterloo.ca",
    packages=["kitti"],
    scripts=[],
    url="https://github.com/hunse/kitti",
    license="MIT",
    description="Tools for working with the KITTI dataset in Python",
    requires=[
        "numpy (>=1.7.0)",
        "scipy (>=0.13.0)",
    ]
)
