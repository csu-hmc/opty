#!/usr/bin/env python

from setuptools import setup, find_packages

exec(open('opty/version.py').read())

setup(
    name='opty',
    version=__version__,
    author='Jason K. Moore',
    author_email='moorepants@gmail.com',
    packages=find_packages(),
    url='http://github.com/csu-hmc/opty',
    license='BSD-2-clause',
    description=('Tools for optimizing dynamic systems using direct '
                 'collocation.'),
    long_description=open('README.rst').read(),
    install_requires=[
        'cyipopt>=1.1.0',
        'cython>=0.29.19',
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'sympy>=1.6.0',
    ],
    extras_require={
        'examples': [
            'matplotlib>=3.2.0',
            'pandas',
            'pydy>=0.5.0',
            'tables',
            'yeadon',
        ],
        'doc': [
            'numpydoc',
            'sphinx',
        ],
    },
    tests_require=['pytest'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
    ]
)
