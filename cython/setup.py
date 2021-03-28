from setuptools import setup
from Cython.Build import cythonize

setup(
    name="Cellular Automata",
    ext_modules = cythonize("ca.pyx"),
    zip_safe=False
)