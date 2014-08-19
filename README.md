kitti
=====

Tools for working with the KITTI dataset in Python.

License
-------

The majority of this project is available under the MIT license. The files in
`kitti/bp` are a notable exception, being a modified version of
Pedro F. Felzenszwalb and Daniel P. Huttenlocher's belief propogation code [1]
licensed under the GNU GPL v2. These files are not essential to any part of the
rest of the project, and are only used to run the optional belief propogation
disparity image interpolation.

[1]: http://cs.brown.edu/~pff/bp/

Setup
-----

To begin working with this project, clone the repository to your machine

    git clone https://github.com/hunse/kitti

Download the KITTI data to a subfolder named ``data`` within this folder.
Most of the tools in this project are for working with the raw KITTI data.
For example, if you download and unpack drive 11 from 2011.09.26, it should
be in the folder ``data/2011_09_26/2011_09_26_drive_0011_sync``. The
calibration files for that day should be in ``data/2011_09_26``.

Since the project uses the location of the Python files to locate the data
folder, the project must be installed in development mode so that it uses the
original source folder

    python setup.py develop

You should now be able to import the project in Python. If you have trouble
with commands like ``kitti.raw.load_video``, check that ``kitti.data.data_dir``
points to the correct location (the location where you put the data), and that
commands like ``kitti.data.get_drive_dir`` return valid paths.

For examples of how to use the commands, look in ``kitti/tests``. Most of the
examples use drive 11, but it should be easy to modify them to use a drive of
your choice.

Happy hacking!

Cython setup
------------

The belief propagation module uses Cython to connect to the C++ BP code. To
build the Cython module, run

    python setup.py build_ext --inplace

This should create the file ``module.so`` in ``kitti/bp``.
