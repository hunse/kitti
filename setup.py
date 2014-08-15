from setuptools import setup

from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = []
ext_modules.append(Extension(
    "kitti.bp.module",
    ["kitti/bp/module.pyx", "kitti/bp/interp.cpp", "kitti/bp/stereo.cpp", "kitti/bp/bp.cpp"],
    include_dirs=["kitti/bp", "/opt/opencv/include"],
    library_dirs=["/opt/opencv/lib"],
    libraries=["opencv_core", "opencv_imgproc"],
    language="c++",
    extra_compile_args=["-w", "-O3"]))

setup(
    name="kitti",
    version="0.0.1",
    author="Eric Hunsberger",
    author_email="ehunsber@uwaterloo.ca",
    url="https://github.com/hunse/kitti",
    license="MIT",
    description="Tools for working with the KITTI dataset in Python",
    requires=[
        "numpy (>=1.7.0)",
        "scipy (>=0.13.0)",
    ],
    packages=["kitti"],
    scripts=[],
    ext_modules=ext_modules,
    cmdclass = {'build_ext': build_ext},
)
