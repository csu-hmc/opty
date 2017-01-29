#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d

from .utils import parse_free


def output_equations(x):
    """Returns the outputs of the system. For now just the an array of the
    generalized coordinates.

    Parameters
    ----------
    x : ndarray, shape(N, n)
        The trajectories of the system states.

    Returns
    -------
    y : ndarray, shape(N, o)
        The trajectories of the generalized coordinates.

    Notes
    -----
    The order of the states is assumed to be:

    [coord_1, ..., coord_{n/2}, speed_1, ..., speed_{n/2}]

    [q_1, ..., q_{n/2}, u_1, ...., u_{n/2}]

    As this is what generate_ode_function creates.

    """

    return x[:, :x.shape[1] // 2]


def objective_function(free, num_dis_points, num_states, dis_period,
                       time_measured, y_measured):
    """Returns the norm of the difference in the measured and simulated
    output.

    Parameters
    ----------
    free : ndarray, shape(n * N + q,)
        The flattened state array with n states at N time points and the q
        free model constants.
    num_dis_points : integer
        The number of model discretization points.
    num_states : integer
        The number of system states.
    dis_period : float
        The discretization time interval.
    y_measured : ndarray, shape(M, o)
        The measured trajectories of the o output variables at each sampled
        time instance.

    Returns
    -------
    cost : float
        The cost value.

    Notes
    -----
    This assumes that the states are ordered:

    [coord1, ..., coordn, speed1, ..., speedn]

    y_measured is interpolated at the discretization time points and
    compared to the model output at the discretization time points.

    """
    M, o = y_measured.shape
    N, n = num_dis_points, num_states

    sample_rate = 1.0 / dis_period
    duration = (N - 1) / sample_rate

    model_time = np.linspace(0.0, duration, num=N)

    states, specified, constants = parse_free(free, n, 0, N)

    model_state_trajectory = states.T  # states is shape(n, N) so transpose
    model_outputs = output_equations(model_state_trajectory)

    func = interp1d(time_measured, y_measured, axis=0)

    return dis_period * np.sum((func(model_time).flatten() -
                                model_outputs.flatten())**2)


def objective_function_gradient(free, num_dis_points, num_states,
                                dis_period, time_measured, y_measured):
    """Returns the gradient of the objective function with respect to the
    free parameters.

    Parameters
    ----------
    free : ndarray, shape(N * n + q,)
        The flattened state array with n states at N time points and the q
        free model constants.
    num_dis_points : integer
        The number of model discretization points.
    num_states : integer
        The number of system states.
    dis_period : float
        The discretization time interval.
    y_measured : ndarray, shape(M, o)
        The measured trajectories of the o output variables at each sampled
        time instance.

    Returns
    -------
    gradient : ndarray, shape(N * n + q,)
        The gradient of the cost function with respect to the free
        parameters.

    Warning
    -------
    This is currently only valid if the model outputs (and measurements) are
    simply the states. The chain rule will be needed if the function
    output_equations() is more than a simple selection.

    """

    M, o = y_measured.shape
    N, n = num_dis_points, num_states

    sample_rate = 1.0 / dis_period
    duration = (N - 1) / sample_rate

    model_time = np.linspace(0.0, duration, num=N)

    states, specified, constants = parse_free(free, n, 0, N)

    model_state_trajectory = states.T  # states is shape(n, N)

    # coordinates
    model_outputs = output_equations(model_state_trajectory)  # shape(N, o)

    func = interp1d(time_measured, y_measured, axis=0)

    dobj_dfree = np.zeros_like(free)
    # Set the derivatives with respect to the coordinates, all else are
    # zero.
    # 2 * (xi - xim)
    dobj_dfree[:N * o] = 2.0 * dis_period * (model_outputs -
                                             func(model_time)).T.flatten()

    return dobj_dfree


def wrap_objective(obj_func, *args):
    def wrapped_func(free):
        return obj_func(free, *args)
    return wrapped_func
