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
        'cython>=0.29.19',
        'numpy>=1.19.0',
        'setuptools',  # provides distutils for Python >=3.13
        'sympy>=1.6.0',
    ],
    extras_require={
        'optional': [
            'scipy>=1.5.0',
            'matplotlib>=3.2.0',
        ],
        'examples': [
            # 'gait2d',  # when available on PyPi
            'matplotlib>=3.2.0',
            'pandas',
            'pydy>=0.5.0',
            'pyyaml',  # gait2d dep
            'scipy>=1.5.0',
            'symmeplot',
            'tables',
            'yeadon',
        ],
        'doc': [
            # 'gait2d',  # when available on PyPi
            'joblib',
            'matplotlib>=3.2.0',
            'numpydoc',
            'pydy>=0.5.0',
            'pyyaml',  # gait2d dep
            'scipy>=1.5.0',
            'sphinx',
            'sphinx-gallery',
            'sphinx-reredirects',
            'symmeplot',
            'yeadon',
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
        'Operating System :: OS Independent',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
    ]
)
