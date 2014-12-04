#!/usr/bin/env python

import numpy as np
from scipy.integrate import odeint
from opty.direct_collocation import Problem

from model import PlanarStandingHumanOnMovingPlatform


def sum_of_sines(sigma, frequencies, time):
    """Returns a sum of sines centered at zero along with its first and
    second derivatives.

    Parameters
    ==========
    sigma : float
        The desired standard deviation of the series.
    frequencies : iterable of floats
        The frequencies of the sin curves to be included in the sum.
    time : array_like, shape(n,)
        The montonically increasing time vector.

    Returns
    =======
    sines : ndarray, shape(n,)
        A sum of sines.
    sines_prime : ndarray, shape(n,)
        The first derivative of the sum of sines.
    sines_double_prime : ndarray, shape(n,)
        The second derivative of the sum of sines.

    """

    phases = 2.0 * np.pi * np.random.ranf(len(frequencies))

    sines = np.zeros_like(time)
    sines_prime = np.zeros_like(time)
    sines_double_prime = np.zeros_like(time)

    amplitude = sigma / 2.0

    for w, p in zip(frequencies, phases):
        sines += amplitude * np.sin(w * time + p)
        sines_prime += amplitude * w * np.cos(w * time + p)
        sines_double_prime -= amplitude * w**2 * np.sin(w * time + p)

    return sines, sines_prime, sines_double_prime

if __name__ == '__main__':

    print('Generating equations of motion.')
    h = PlanarStandingHumanOnMovingPlatform()
    h.derive()

    num_nodes = 2000
    duration = 20.0
    interval = duration / (num_nodes - 1)
    time = np.linspace(0.0, duration, num=num_nodes)

    ref_noise = np.random.normal(scale=np.deg2rad(1.0), size=(len(time), 4))
    ref_noise = np.zeros((len(time), 4))

    nums = [7, 11, 16, 25, 38, 61, 103, 131, 151, 181, 313, 523]
    freq = 2.0 * np.pi * np.array(nums, dtype=float) / 240.0
    pos, vel, accel = sum_of_sines(0.01, freq, time)

    x0 = np.zeros(4)

    print('Generating right hand side function.')
    rhs, args = h.closed_loop_ode_func(time, ref_noise, accel)

    print('Integrating equations of motion.')
    x = odeint(rhs, x0, time, args=(args,))

    # Add measurement noise to the data.
    x_meas = x + np.random.normal(scale=np.deg2rad(0.25), size=x.shape)
    #x_meas = x

    # TODO : Add measurement noise to the acceleration measurement.

    x_meas_vec = x_meas.T.flatten()

    print('Setting up optimization problem.')

    def obj(free):
        """Minimize the error in the angle, y1."""
        return interval * np.sum((x_meas_vec - free[:4 * num_nodes])**2)

    def obj_grad(free):
        grad = np.zeros_like(free)
        grad[:4 * num_nodes] = 2.0 * interval * (free[:4 * num_nodes] -
                                                 x_meas_vec)
        return grad

    bounds = {}
    for g in h.gain_symbols:
        bounds[g] = (0.0, 1.0)

    prob = Problem(obj, obj_grad,
                   h.first_order_implicit(),
                   h.states(),
                   num_nodes, interval,
                   known_parameter_map=h.closed_loop_par_map,
                   known_trajectory_map={h.specified['platform_acceleration']: accel},
                   bounds=bounds,
                   time_symbol=h.time,
                   integration_method='midpoint')

    #initial_guess = np.hstack((x_meas_vec, (h.numerical_gains_scales *
                                            #h.numerical_gains).flatten()))
    initial_guess = np.hstack((x_meas_vec, np.random.random(8)))
    #initial_guess = np.hstack((x_meas_vec, np.zeros(8)))

    # Find the optimal solution.
    solution, info = prob.solve(initial_guess)
    p_sol = solution[-8:]

    print("Gain initial guess: {}".format(initial_guess[-8:]))
    print("Known value of p = {}".format(h.numerical_gains.flatten()))
    print("Identified value of p = {}".format(h.numerical_gains_scales.flatten() * p_sol))
