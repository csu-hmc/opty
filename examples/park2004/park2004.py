#!/usr/bin/env python

import numpy as np
from scipy.integrate import odeint
from opty.direct_collocation import Problem
from opty.utils import sum_of_sines

from model import PlanarStandingHumanOnMovingPlatform

if __name__ == '__main__':

    print('Generating equations of motion.')
    # We are going to scale the gains so that the values we search for with
    # IPOPT are all close to 0.5 instead of the large gain values.
    h = PlanarStandingHumanOnMovingPlatform(unscaled_gain=0.5)
    h.derive()

    num_nodes = 4000
    duration = 20.0
    interval = duration / (num_nodes - 1)
    time = np.linspace(0.0, duration, num=num_nodes)

    # ref noise seems to introduce error in the parameter id
    ref_noise = np.random.normal(scale=np.deg2rad(1.0), size=(len(time), 4))
    #ref_noise = np.zeros((len(time), 4))

    nums = [7, 11, 16, 25, 38, 61, 103, 131, 151, 181, 313, 523]
    freq = 2.0 * np.pi * np.array(nums, dtype=float) / 240.0
    pos, vel, accel = sum_of_sines(0.01, freq, time)
    accel_meas = accel + np.random.normal(scale=np.deg2rad(0.25),
                                          size=accel.shape)

    x0 = np.zeros(4)

    print('Generating right hand side function.')
    rhs, r, p = h.closed_loop_ode_func(time, ref_noise, accel)

    print('Integrating equations of motion.')
    x = odeint(rhs, x0, time, args=(r, p))

    # Add measurement noise to the data.
    x_meas = x + np.random.normal(scale=np.deg2rad(0.25), size=x.shape)

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
                   known_trajectory_map={h.specified['platform_acceleration']: accel_meas},
                   bounds=bounds,
                   time_symbol=h.time,
                   integration_method='midpoint')

    initial_guess = np.hstack((x_meas_vec, (h.gain_scale_factors *
                                            h.numerical_gains).flatten()))
    initial_guess = np.hstack((x_meas_vec, np.random.random(8)))
    initial_guess = np.hstack((x_meas_vec, np.zeros(8)))
    initial_guess = np.zeros(prob.num_free)

    # Find the optimal solution.
    solution, info = prob.solve(initial_guess)
    p_sol = solution[-8:]

    print("Gain initial guess: {}".format(h.gain_scale_factors.flatten() *
                                          initial_guess[-8:]))
    print("Known value of p = {}".format(h.numerical_gains.flatten()))
    print("Identified value of p = {}".format(h.gain_scale_factors.flatten()
                                              * p_sol))
