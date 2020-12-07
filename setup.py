# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir


import sys
import os

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension("_isodisreg",
        ["src/isodisreg/isocdf_seq.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        ),
]


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


requirements = resolve_requirements(
    os.path.join(os.path.dirname(__file__), "requirements.txt"))



setup(
    name='isodisreg',
    version='0.1',
    author='Eva-Maria Walz',
    author_email='evamaria.walz@gmx.de',
    description='Isotonic distributional regression (IDR) is a nonparametric technique for the estimation of distributions of a binary or numeric response variable conditional on numeric or ordinal covariates.',
    long_description=open('README.md').read(),
    license='MIT',
    #install_requires=['numpy', 'pandas', 'scipy', 'progressbar', 'dc_stat_think', 'osqp', 'matplotlib'],
    install_requires=requirements,
    url='https://github.com/evwalz/isodisreg',
    packages=find_packages('src'),
    package_dir={'':'src'},
    package_data={'isodisreg':['packageData/rain.dat']},
    ext_modules=ext_modules,
    #ext_modules=[CMakeExtension('isodisreg._isodisreg')],
    cmdclass={"build_ext": build_ext},
)

