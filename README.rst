Introduction
============

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

- sympy >= 0.7.6
- ipopt >= 3.11
- numpy >= 1.8.1
- scipy >= 0.14.1
- cython >= 0.20.1
- cyipopt >= 0.1.7

To run the examples the following additional dependencies are required:

- matplotlib >= 1.3.1
- pydy >= 0.2.1
- pytables

First you must install IPOPT along with it's headers. For example on Debian
based systems you can use the package manager::

   $ sudo apt-get install coinor-libipopt1 coinor-libipopt-dev

For customized installation (usually desired for performance) follow the
instructions on the IPOPT documentation to compile the library. If you install
to a location other than `/usr/local` you will likely have to set the
``LD_LIBRARY_PATH`` so that you can link to IPOPT when installing ``cyipopt``.

Install conda then create an environment::

   $ conda create -n opty pip numpy scipy cython matplotlib pytables
   $ source activate opty
   (opty)$ pip install https://github.com/sympy/sympy/releases/download/sympy-0.7.6.rc1/sympy-0.7.6.rc1.tar.gz
   (opty)$ pip install pydy
   (opty)$ pip install https://bitbucket.org/moorepants/cyipopt/get/tip.zip
   (opty)$ pip install -e /path/to/opty

Usage
=====

There are several examples available in the ``examples`` directory. For example
the optimal torque to swing up a pendulum with minimal energy can be found
with::

   (opty)$ cd examples/
   (opty)$ python pendulum_swing_up.py
