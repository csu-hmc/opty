Introduction
============

.. image:: https://img.shields.io/pypi/v/opty.svg
   :target: https://pypi.org/project/opty

.. image:: https://anaconda.org/conda-forge/opty/badges/version.svg
   :target: https://anaconda.org/conda-forge/opty

.. image:: https://readthedocs.org/projects/opty/badge/?version=stable
   :target: http://opty.readthedocs.io

.. image:: https://travis-ci.org/csu-hmc/opty.svg?branch=master
   :target: https://travis-ci.org/csu-hmc/opty

``opty`` utilizes symbolic descriptions of ordinary differential equations
expressed with SymPy_ to form the constraints needed to solve optimal control
and parameter identification problems using the direct collocation method and
non-linear programming. In general, if one can express the continuous first
order ordinary differential equations of the system as symbolic expressions
``opty`` will automatically generate a function to efficiently evaluate the
dynamical constraints and a function that evaluates the sparse Jacobian of the
constraints, which have been optimized for speed and memory consumption. The
translation of the dynamical system description to the NLP form, primarily the
formation of the constraints and the Jacobian of the constraints, manually is
a time consuming and error prone process. ``opty`` eliminates both of those
issues.

.. _SymPy: http://www.sympy.org

Features
--------

- Both implicit and explicit forms of the first order ordinary differential
  equations and differential algebraic equations are supported, i.e. there is
  no need to solve for the derivatives of the dependent variables.
- Backward Euler or Midpoint integration methods.
- Supports both trajectory optimization and parameter identification.
- Easy specification of bounds on free variables.
- Easily specify additional "instance" constraints.
- Automatic parallel execution using openmp if installed.
- Built with support of sympy.physics.mechanics and PyDy in mind.

Installation
============

The required dependencies are as follows:

- python 2.7 or 3.5+
- sympy >= 1.0.0
- ipopt >= 3.11
- numpy >= 1.8.1
- scipy >= 0.14.1
- cython >= 0.20.1
- cyipopt >= 0.1.7

To run all of the examples the following additional dependencies are required:

- matplotlib >= 1.3.1
- pydy >= 0.3.0
- pytables
- pandas
- yeadon
- openmp

**Currently only Linux and Mac are officially supported.** Although, it should
be possible to install this on Windows with an appropriate Cython compilation
toolchain and IPOPT installed from binaries or custom compliation.

If you are installing on Linux or Mac, the easiest way to get started is to
install Anaconda_ (or Miniconda_) and use conda to install opty and any desired
optional dependencies from the Conda Forge channel, e.g.::

   $ conda config --add channels conda-forge
   $ conda install opty matplotlib pytables pandas yeadon pydy

.. _Anaconda: https://www.continuum.io/downloads
.. _Miniconda: https://conda.io/miniconda.html

If you are using Windows or want a custom installation of any of the
dependencies, e.g. Ipopt, you must first install Ipopt along with it's headers.
For example, on Debian based systems you can use the package manager::

   $ sudo apt-get install coinor-libipopt1v5 coinor-libipopt-dev

or prebuilt binaries can be downloaded from
https://www.coin-or.org/download/binary/Ipopt/.

For customized installation (usually desired for performance) follow the
instructions on the IPOPT documentation to compile the library. If you install
to a location other than `/usr/local` on Unix systems you will likely have to
set the ``LD_LIBRARY_PATH`` so that you can link to IPOPT when installing
``cyipopt``.

Once Ipopt is installed and accessible, install conda then create an environment::

   $ conda create -n opty-custom pip numpy scipy cython sympy
   $ source activate opty-custom
   (opty-custom)$ pip install https://github.com/matthias-k/cyipopt/archive/v0.1.7.tar.gz
   (opty-custom)$ pip install opty

If you want to develop opty, create a conda environment with all of the
dependencies installed::

   $ conda config --add channels conda-forge
   $ conda create -n opty-dev python sympy numpy scipy cython ipopt cyipopt matplotlib pytables pydy pandas
   $ source activate opty-dev

Next download the opty source files and install with::

   (opty-dev)$ conda develop /path/to/opty

or::

   (opty-dev)$ cd /path/to/opty
   (opty-dev)$ python setup.py develop

Usage
=====

There are several examples available in the ``examples`` directory. For
example, the optimal torque to swing up a pendulum with minimal energy can be
run with::

   $ python examples/pendulum_swing_up.py

Funding
=======

The work was partially funded by the State of Ohio Third Frontier Commission
through the Wright Center for Sensor Systems Engineering (WCSSE), by the
National Science Foundation under Grant No. 1344954, and by National Center of
Simulation in Rehabilitation Research 2014 Visiting Scholarship at Stanford
University.
