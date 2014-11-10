This script demonstrates an attempt at identifying the controller for an N-link
inverted pendulum on a cart by direct collocation. I collect "measured" data
from the system by simulating it with a known optimal controller under the
influence of random lateral force perturbations. I then form the optimization
problem such that we minimize the error in the model's simulated outputs with
respect to the measured outputs. The optimizer searches for the best set of
controller gains (which are unknown) that reproduce the motion and ensure the
dynamics are valid.

``python pendulum.py -h`` gives the options.
