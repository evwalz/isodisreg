# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='idr',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    packages_data = {'idr':['packageData/rain.dat']}, 
    license='MIT',
    description='Isotonic distributional regression (IDR) is a nonparametric technique for the estimation of distributions of a binary or numeric response variable conditional on numeric or ordinal covariates.',
    long_description=open('README.md').read(),
    install_requires=['numpy', 'pandas', 'scipy', 'progressbar', 'dc_stat_think', 'collections', 'osqp', 'matplotlib', 'sklearn'],
    url='https://github.com/evwalz/idr',
    author='Eva-Maria Walz',
    author_email='evamaria.walz@gmx.de'
)

