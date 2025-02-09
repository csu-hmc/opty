"""
Standing Balance Control Identification
=======================================

This example shows how to solve the same human control parameter identification
problem presented in [Park2004]_ using simulated noisy measurement data. The
goal is to find a set of human standing balance controller gains from data of
perturbed standing balance. The dynamics model is a 2D planar two-body model
representing a human standing on a antero-posteriorly moving platform, similar
to [Park2004]_'s. The dynamics model is developed in
:download:`<model_park2004.py>`.

.. warning::

   This example requires SciPy, yeadon, and PyDy in addition to opty and its
   required dependencies.

References
----------

.. [Park2004] Park, S., Horak, F. B., & Kuo, A. D. (2004). Postural feedback
   responses scale with biomechanical constraints in human standing.
   Experimental Brain Research, 154(4), 417–427.
   https://doi.org/10.1007/s00221-003-1674-3

"""
from opty import Problem
from opty.utils import sum_of_sines
from scipy.integrate import odeint
import numpy as np
import sympy as sm

from model_park2004 import PlanarStandingHumanOnMovingPlatform

# %%
# Generate the equations of motion and scale the control gains so that the
# values we search for with IPOPT are all close to 0.5 instead of the large
# gain values.
h = PlanarStandingHumanOnMovingPlatform(unscaled_gain=0.5)
h.derive()
eom = h.first_order_implicit()
sm.pprint(sm.simplify(eom))

# %%
# Define the time discretization.
num_nodes = 4000
duration = 20.0
interval = duration/(num_nodes - 1)
time = np.linspace(0.0, duration, num=num_nodes)

# ref noise seems to introduce error in the parameter id
ref_noise = np.random.normal(scale=np.deg2rad(1.0), size=(len(time), 4))
#ref_noise = np.zeros((len(time), 4))

# %%
# Create a sum of sinusoids to excite the platform.
nums = [7, 11, 16, 25, 38, 61, 103, 131, 151, 181, 313, 523]
freq = 2.0*np.pi*np.array(nums, dtype=float)/240.0
pos, vel, accel = sum_of_sines(0.01, freq, time)
accel_meas = accel + np.random.normal(scale=np.deg2rad(0.25), size=accel.shape)

# %%
# Simulate the motion of the human under the sinusoidal excitation and add
# Gaussian measurement noise.
rhs, r, p = h.closed_loop_ode_func(time, ref_noise, accel)
x0 = np.zeros(4)
x = odeint(rhs, x0, time, args=(r, p))
x_meas = x + np.random.normal(scale=np.deg2rad(0.25), size=x.shape)
x_meas_vec = x_meas.T.flatten()


# %%
# Set up the control parameter identification problem.
def obj(free):
    """Minimize the error in the angle, y1."""
    return interval*np.sum((x_meas_vec - free[:4*num_nodes])**2)


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[:4*num_nodes] = 2.0*interval*(free[:4*num_nodes] - x_meas_vec)
    return grad


bounds = {}
for g in h.gain_symbols:
    bounds[g] = (0.0, 1.0)

prob = Problem(
    obj,
    obj_grad,
    eom,
    h.states(),
    num_nodes,
    interval,
    known_parameter_map=h.closed_loop_par_map,
    known_trajectory_map={h.specified['platform_acceleration']: accel_meas},
    bounds=bounds,
    time_symbol=h.time,
    integration_method='midpoint',
)

initial_guess = np.hstack((x_meas_vec,
                           (h.gain_scale_factors*h.numerical_gains).flatten()))
initial_guess = np.hstack((x_meas_vec, np.random.random(8)))
initial_guess = np.hstack((x_meas_vec, np.zeros(8)))
initial_guess = np.zeros(prob.num_free)

# %%
# Find the optimal solution.
solution, info = prob.solve(initial_guess)
p_sol = solution[-8:]

print("Gain initial guess: {}".format(
    h.gain_scale_factors.flatten()*initial_guess[-8:]))
print("Known value of p = {}".format(h.numerical_gains.flatten()))
print("Identified value of p = {}".format(
    h.gain_scale_factors.flatten()*p_sol))
