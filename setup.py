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
    description=('Tool for optimizing dynamic systems using direct '
                 'collocation.'),
    long_description=open('README.rst').read(),
    install_requires=[
        'cyipopt>=1.1.0',
        'cython>=0.29.28',
        'numpy>=1.21.5',
        'setuptools>=59.6.0',  # provides distutils for Python >=3.13
        'sympy>=1.9.1',
    ],
    extras_require={
        'optional': [
            'scipy>=1.8.0',
            'matplotlib>=3.5.1',
        ],
        'examples': [
            # 'gait2d',  # when available on PyPi
            'matplotlib>=3.5.1',
            'pandas>=1.3.5',
            'pydy>=0.6.0',
            'pyyaml>=5.4.1',  # gait2d dep
            'scipy>=1.8.0',
            'symmeplot',
            'tables>=3.7.0',
            'yeadon>=1.4.0',
        ],
        'doc': [
            # 'gait2d',  # when available on PyPi
            'joblib >=0.17.0',
            'matplotlib>=3.5.1',
            'numpydoc >=1.2',
            'pydy>=0.6.0',
            'pyyaml>=5.4.1',  # gait2d dep
            'scipy>=1.8.0',
            'sphinx>=4.3.2',
            'sphinx-gallery',
            'sphinx-reredirects',
            'symmeplot',
            'yeadon>=1.4.0',
        ],
    },
    tests_require=['pytest'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Operating System :: OS Independent',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
    ]
)
