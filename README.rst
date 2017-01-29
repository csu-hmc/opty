Introduction
============

.. image:: https://img.shields.io/pypi/v/opty.svg
   :target: https://pypi.org/project/opty

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
formation of the constraints and the Jabcobian of the constraints, manually is
a time consuming and error prone process. ``opty`` eliminates both of those
issues.

.. _SymPy: http://www.sympy.org

Features
--------

- Both implicit and explicit forms of the first order ordinary differential
  equations are supported, i.e. there is no need to solve for x'.
- Backward Euler or Midpoint integration methods.
- Supports both trajectory optimization and parameter identification.
- Easy specification of bounds on free variables.
- Easily specify additional "instance" constraints.
- Built with support of sympy.physics.mechanics in mind.

Installation
============

The core dependencies are as follows:

- python 2.7 or 3.5+
- sympy >= 0.7.6
- ipopt >= 3.11
- numpy >= 1.8.1
- scipy >= 0.14.1
- cython >= 0.20.1
- cyipopt >= 0.1.7

To run all of the examples the following additional dependencies are required:

- matplotlib >= 1.3.1
- pydy >= 0.2.1
- pytables
- pandas
- yeadon

If you are installing on Linux or Mac, the easiest way to get started is to
install Anaconda_ (or Miniconda_) and use conda to install all of the
dependencies from the Conda Forge channel::

   $ conda config --add channels conda-forge
   $ conda install sympy numpy scipy cython ipopt cyipopt matplotlib pytables pydy pandas

Next download the opty source files and install with::

   $ conda develop /path/to/opty

or::

   $ cd /path/to/opty
   $ python setup.py install

.. _Anaconda: https://www.continuum.io/downloads
.. _Miniconda: https://conda.io/miniconda.html

If you are using Windows or want a custom installation of Ipopt, you must first
install IPOPT along with it's headers. For example, on Debian based systems you
can use the package manager::

   $ sudo apt-get install coinor-libipopt1v5 coinor-libipopt-dev

or prebuilt binaries can be downloaded from
https://www.coin-or.org/download/binary/Ipopt/.

For customized installation (usually desired for performance) follow the
instructions on the IPOPT documentation to compile the library. If you install
to a location other than `/usr/local` you will likely have to set the
``LD_LIBRARY_PATH`` so that you can link to IPOPT when installing ``cyipopt``.

Once Ipopt is installed and accessible, install conda then create an environment::

   $ conda create -n opty pip numpy scipy cython matplotlib pytables sympy pydy pandas
   $ source activate opty
   (opty)$ pip install https://github.com/matthias-k/cyipopt/archive/master.zip
   (opty)$ conda develop /path/to/opty

Usage
=====

There are several examples available in the ``examples`` directory. For
example, the optimal torque to swing up a pendulum with minimal energy can be
found with::

   $ python examples/pendulum_swing_up.py
