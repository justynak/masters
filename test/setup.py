# Cython compile instructions
#
# Build in-place with:
#   ../.venv/bin/python setup.py build_ext --inplace
# (or `make ext` from the repo root)

from setuptools import setup
from Cython.Build import cythonize

setup(
    name="rTransformApp",
    ext_modules=cythonize("rTransform.pyx", language_level=3),
)
