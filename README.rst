Introduction
============

``opty`` utlizes symbolic descriptions of ordinary differential equations
expressed with SymPy to form the constraint functions needed to solve
optimization problems using the direct collocation method. In general, if one
can express first order ordinary differential equations as symbolic expressions
``opty`` will automatically generate a function to evaluate the constraints and
a function that evaluates the sparse Jacobian of the constraints.

Installation
============

Dependencies this runs with:

- ipopt 3.11
- numpy 1.8.1
- scipy 0.14.1
- sympy 0.7.6
- cython
- cyipopt 0.1.4

To run the examples you will additionally need:

- matplotlib 1.3.1
- pydy 0.2.1
- tables

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

There are several examples available in the ``examples`` directory. For example::

   (opty)$ cd examples/
   (opty)$ python pendulum_swing_up.py
