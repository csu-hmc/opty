This script demonstrates an attempt at identifying the controller for a two
link inverted pendulum on a cart by direct collocation. I collect "measured"
data from the system by simulating it with a known optimal controller under the
influence of random lateral force perturbations. I then form the optimization
problem such that we minimize the error in the model's simulated outputs with
respect to the measured outputs. The optimizer searches for the best set of
controller gains (which are unknown) that reproduce the motion and ensure the
dynamics are valid.

Dependencies this runs with:

- numpy 1.8.1
- scipy 0.14.1
- matplotlib 1.3.1
- sympy HEAD of master
- pydy 0.2.1
- cyipopt 0.1.4

Installation
============

Install conda then create an environment::

   $ conda create -n inverted-pendulum-id numpy=1.8.1 scipy=0.14.1 matplotlib=1.3.1
   $ source activate inverted-pendulum-id
   (inverted-pendulum-id)$ git clone git@github.com:sympy/sympy.git
   (inverted-pendulum-id)$ pip install -e sympy
   (inverted-pendulum-id)$ pip install pydy
