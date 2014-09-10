#!/usr/bin/env python

"""This script demonstrates an attempt at identifying the controller for a n
link inverted pendulum on a cart by direct collocation. I collect "measured"
data from the system by simulating it with a known optimal controller under
the influence of random lateral force perturbations. I then form the
optimization problem such that we minimize the error in the model's
simulated outputs with respect to the measured outputs. The optimizer
searches for the best set of controller gains (which are unknown) that
reproduce the motion and ensure the dynamics are valid.

Dependencies this runs with:

    numpy 1.8.1
    scipy 0.14.1
    sympy 0.7.5
    matplotlib 1.3.1
    pydy 0.2.1
    cyipopt 0.1.4

N : number of discretization points
M : number of measured time samples

n : number of states
o : number of model outputs
p : total number of model constants
q : number of free model constants
r : number of free specified inputs

"""

# standard lib
from collections import OrderedDict
import os
import datetime
import hashlib
import time

# external
import numpy as np
import sympy as sym
from scipy.interpolate import interp1d
from scipy.linalg import solve_continuous_are
from scipy.integrate import odeint
from pydy.codegen.code import generate_ode_function
from model import n_link_pendulum_on_cart
import ipopt
import tables
import pandas

# local
from utils import controllable, ufuncify_matrix
from visualization import (plot_sim_results, plot_constraints,
                           animate_pendulum, plot_identified_state_trajectory)


def constants_dict(constants):
    """Returns an ordered dictionary which maps the system constant symbols
    to numerical values. The cart sping is set to 10.0 N/m, the cart damper
    to 5.0 Ns/m and gravity is set to 9.81 m/s and the masses and lengths of
    the pendulums are all set to 1.0 kg and meter, respectively."""
    return OrderedDict(zip(constants, [10.0, 5.0, 9.81] + (len(constants) - 1) * [1.0]))


def state_derivatives(states):
    """Returns functions of time which represent the time derivatives of the
    states."""
    return [state.diff() for state in states]


def f_minus_ma(mass_matrix, forcing_vector, states):
    """Returns Fr + Fr* from the mass_matrix and forcing vector."""

    xdot = sym.Matrix(state_derivatives(states))

    return mass_matrix * xdot - forcing_vector


def compute_controller_gains(num_links):
    """Returns a numerical gain matrix that can be multiplied by the error
    in the states of the n link pendulum on a cart to generate the joint
    torques needed to stabilize the pendulum. The controller follows this
    pattern:

        u(t) = K * [x_eq - x(t)]

    Parameters
    ----------
    n

    Returns
    -------
    K : ndarray, shape(2, n)
        The gains needed to compute joint torques.

    """

    res = n_link_pendulum_on_cart(num_links, cart_force=False,
                                  joint_torques=True, spring_damper=True)

    mass_matrix = res[0]
    forcing_vector = res[1]
    constants = res[2]
    coordinates = res[3]
    speeds = res[4]
    specified = res[5]

    states = coordinates + speeds

    equilibrium_point = np.zeros(len(coordinates) + len(speeds))
    equilibrium_dict = dict(zip(states, equilibrium_point))

    F_A = forcing_vector.jacobian(states)
    F_A = F_A.subs(equilibrium_dict).subs(constants_dict(constants))
    F_A = np.array(F_A.tolist(), dtype=float)

    F_B = forcing_vector.jacobian(specified)
    F_B = F_B.subs(equilibrium_dict).subs(constants_dict(constants))
    F_B = np.array(F_B.tolist(), dtype=float)

    M = mass_matrix.subs(equilibrium_dict).subs(constants_dict(constants))
    M = np.array(M.tolist(), dtype=float)

    invM = np.linalg.inv(M)
    A = np.dot(invM, F_A)
    B = np.dot(invM, F_B)

    assert controllable(A, B)

    Q = np.eye(len(states))
    R = np.eye(len(specified))

    S = solve_continuous_are(A, B, Q, R)

    K = np.dot(np.dot(np.linalg.inv(R), B.T),  S)

    return K


def create_symbolic_controller(states, inputs):
    """"Returns a dictionary with keys that are the joint torque inputs and
    the values are the controller expressions. This can be used to convert
    the symbolic equations of motion from 0 = f(x', x, u, t) to a closed
    loop form 0 = f(x', x, t).

    Parameters
    ----------
    states : sequence of len 2 * (n + 1)
        The SymPy time dependent functions for the system states where n are
        the number of links.
    inputs : sequence of len n
        The SymPy time depednent functions for the system joint torque
        inputs (should not include the lateral force).

    Returns
    -------
    controller_dict : dictionary
        Maps joint torques to control expressions.
    gain_symbols : list of SymPy Symbols
        The symbols used in the gain matrix.
    xeq : list of SymPy Symbols
        The symbols for the equilibrium point.

    """
    num_states = len(states)
    num_inputs = len(inputs)

    xeq = sym.Matrix([x.__class__.__name__ + '_eq' for x in states])

    K = sym.Matrix(num_inputs, num_states, lambda i, j:
                   sym.Symbol('k_{}{}'.format(i, j)))

    x = sym.Matrix(states)
    T = sym.Matrix(inputs)

    gain_symbols = [k for k in K]

    # T = K * (xeq - x) -> 0 = T - K * (xeq - x)

    controller_dict = sym.solve(T - K * (xeq - x), inputs)

    return controller_dict, gain_symbols, xeq


def symbolic_constraints(mass_matrix, forcing_vector, states,
                         controller_dict, equilibrium_dict=None):
    """Returns a vector expression of the zero valued closed loop system
    equations of motion: M * x' - F.

    Parameters
    ----------
    mass_matrix : sympy.Matrix, shape(n, n)
        The system mass matrix, M.
    forcing_vector : sympy.Matrix, shape(n, 1)
        The system forcing vector, F.
    states : iterable of sympy.Function, len(n)
        The functions of time representing the states.
    controll_dict : dictionary
        Maps any input forces in the forcing vector to the symbolic
        controller expressions.
    equilibrium_dit : dictionary
        A dictionary of equilibrium values to substitute.

    Returns
    -------
    constraints : sympy.Matrix, shape(n, 1)
        The closed loop constraint expressions.

    """

    xdot = sym.Matrix(state_derivatives(states))

    if equilibrium_dict is not None:
        for k, v in controller_dict.items():
            controller_dict[k] = v.subs(equilibrium_dict)

    # M * x' = F -> M * x' - F = 0
    system = mass_matrix * xdot - forcing_vector.subs(controller_dict)

    return system


def symbolic_constraints_solved(mass_matrix, forcing_vector, states,
                                controller_dict, equilibrium_dict=None):
    """Returns a vector expression of the zero valued closed loop system
    equations of motion: x' - M^-1 * F.

    Parameters
    ----------
    mass_matrix : sympy.Matrix, shape(n, n)
        The system mass matrix, M.
    forcing_vector : sympy.Matrix, shape(n, 1)
        The system forcing vector, F.
    states : iterable of sympy.Function, len(n)
        The functions of time representing the states.
    controll_dict : dictionary
        Maps any input forces in the forcing vector to the symbolic
        controller expressions.
    equilibrium_dit : dictionary
        A dictionary of equilibrium values to substitute.

    Returns
    -------
    constraints : sympy.Matrix, shape(n, 1)
        The closed loop constraint expressions.

    Notes
    -----
    The mass matrix is symbolically inverted, so this can be potentailly be
    slow for large systems.

    """

    xdot = sym.Matrix(state_derivatives(states))

    if equilibrium_dict is not None:
        for k, v in controller_dict.items():
            controller_dict[k] = v.subs(equilibrium_dict)

    F = forcing_vector.subs(controller_dict)
    constraints = xdot - mass_matrix.LUsolve(F)

    return constraints


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

    return x[:, :x.shape[1] / 2]


def closed_loop_ode_func(system, time, set_point, gain_matrix, lateral_force):
    """Returns a function that evaluates the continous closed loop system
    first order ODEs.

    Parameters
    ----------
    system : tuple, len(6)
        The output of the symbolic EoM generator.
    time : ndarray, shape(M,)
        The monotonically increasing time values that
    set_point : ndarray, shape(n,)
        The set point for the controller.
    gain_matrix : ndarray, shape((n - 1) / 2, n)
        The gain matrix that computes the optimal joint torques given the
        system state.
    lateral_force : ndarray, shape(M,)
        The applied lateral force at each time point. This will be linearly
        interpolated for time points other than those in time.

    Returns
    -------
    rhs : function
        A function that evaluates the right hand side of the first order
        ODEs in a form easily used with odeint.
    args : dictionary
        A dictionary containing the model constant values and the controller
        function.

    """

    # TODO : It will likely be useful to allow more inputs: noise on the
    # equilibrium point (i.e. sensor noise) and noise on the joint torques.
    # 10 cycles /sec * 2 pi rad / cycle

    interp_func = interp1d(time, lateral_force)

    def controller(x, t):
        joint_torques = np.dot(gain_matrix, set_point - x)
        if t > time[-1]:
            lateral_force = interp_func(time[-1])
        else:
            lateral_force = interp_func(t)
        return np.hstack((joint_torques, lateral_force))

    rhs = generate_ode_function(*system, generator='cython')

    args = {'constants': np.array(constants_dict(system[2]).values()),
            'specified': controller}

    return rhs, args


def sum_of_sines(magnitudes, frequencies, time):
    sines = np.zeros_like(time)
    for m, w in zip(magnitudes, frequencies):
        sines += m * np.sin(w * time)
    return sines


def discrete_symbols(states, specified, interval='h'):
    """Returns discrete symbols for each state and specified input along
    with an interval symbol.

    Parameters
    ----------
    states : list of sympy.Functions
        The n functions of time representing the system's states.
    specified : list of sympy.Functions
        The m functions of time representing the system's specified inputs.
    interval : string, optional
        The string to use for the discrete time interval symbol.

    Returns
    -------
    current_states : list of sympy.Symbols
        The n symbols representing the system's ith states.
    previous_states : list of sympy.Symbols
        The n symbols representing the system's (ith - 1) states.
    current_specified : list of sympy.Symbols
        The m symbols representing the system's ith specified inputs.
    interval : sympy.Symbol
        The symbol for the time interval.

    """

    xi = [sym.Symbol(f.__class__.__name__ + 'i') for f in states]
    xp = [sym.Symbol(f.__class__.__name__ + 'p') for f in states]
    si = [sym.Symbol(f.__class__.__name__ + 'i') for f in specified]
    h = sym.Symbol(interval)

    return xi, xp, si, h


def discretize(eoms, states, specified, interval='h'):
    """Returns the constraint equations in a discretized form. Backward
    Euler discretization is used.

    Parameters
    ----------
    states : list of sympy.Functions
        The n functions of time representing the system's states.
    specified : list of sympy.Functions
        The m functions of time representing the system's specified inputs.
    interval : string, optional
        The string to use for the discrete time interval symbol.

    Returns
    -------
    discrete_eoms : sympy.Matrix
        The column vector of the constraint expressions.

    """
    xi, xp, si, h = discrete_symbols(states, specified, interval=interval)

    euler_formula = [(i - p) / h for i, p in zip(xi, xp)]

    # Note : The Derivatives must be substituted before the symbols.
    eoms = eoms.subs(dict(zip(state_derivatives(states), euler_formula)))

    eoms = eoms.subs(dict(zip(states + specified, xi + si)))

    return eoms


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
    dobj_dfree[:N * o] = 2.0 * dis_period * (model_outputs - func(model_time)).T.flatten()

    return dobj_dfree


def wrap_objective(obj_func, *args):
    def wrapped_func(free):
        return obj_func(free, *args)
    return wrapped_func


def general_constraint(eom_vector, state_syms, specified_syms,
                       constant_syms):
    """Returns a function that evaluates the constraints.

    Parameters
    ----------
    discrete_eom_vec : sympy.Matrix, shape(n, 1)
        A column vector containing the discrete symbolic expressions of the
        n constraints.
    state_syms : list of sympy.Functions
        The n functions of time representing the system's states.
    specified_syms : list of sympy.Functions
        The m functions of time representing the system's specified inputs.
    constant_syms : list of sympy.Symbols
        The b symbols representing the system's specified inputs.

    Returns
    -------
    constraints : function
        A function which returns the numerical values of the constraints at
        time points 2,...,N.

    Notes
    -----
    args:
        all current states (x1i, ..., xni)
        all previous states (x1p, ... xnp)
        all current specifieds (s1i, ..., smi)
        constants (c1, ..., cb)
        time interval (h)

        args: (x1i, ..., xni, x1p, ... xnp, s1i, ..., smi, c1, ..., cb, h)
        n: num states
        m: num specified
        b: num constants

    The function should evaluate and return an array:

        [con_1_2, ..., con_1_N, con_2_2, ..., con_2_N, ..., con_n_2, ..., con_n_N]

    for n states and N-1 constraints at the time points.

    """
    xi_syms, xp_syms, si_syms, h_sym = \
        discrete_symbols(state_syms, specified_syms)

    args = [x for x in xi_syms] + [x for x in xp_syms]
    args += [s for s in si_syms] + constant_syms + [h_sym]

    f = ufuncify_matrix(args, eom_vector)

    def constraints(state_values, specified_values, constant_values,
                    interval_value):
        """Returns a vector of constraint values given all of the
        unknowns in the equations of motion over the 2, ..., N time
        steps.

        Parameters
        ----------
        states : ndarray, shape(n, N)
            The array of n states through N time steps.
        specified_values : ndarray, shape(m, N) or shape(N,)
            The array of m specifieds through N time steps.
        constant_values : ndarray, shape(b,)
            The array of b constants.
        interval_value : float
            The value of the dicretization time interval.

        Returns
        -------
        constraints : ndarray, shape(N-1,)
            The array of constraints from t = 2, ..., N.
            [con_1_2, ..., con_1_N, con_2_2, ..., con_2_N, ..., con_n_2, ..., con_n_N]

        """

        if state_values.shape[0] < 2:
            raise ValueError('There should always be at least two states.')

        x_current = state_values[:, 1:]  # n x N - 1
        x_previous = state_values[:, :-1]  # n x N - 1

        # 2n x N - 1
        args = [x for x in x_current] + [x for x in x_previous]

        # 2n + m x N - 1
        if len(specified_values.shape) == 2:
            si = specified_values[:, 1:]
            args += [s for s in si]
        else:
            si = specified_values[1:]
            args += [si]

        # These are scalars so, for now, we need to create arrays for these
        # because my version of ufuncify only works with arrays for all
        # arguments. These are generally very short arrays, so it shouldn't
        # be that much overhead.
        num_constraints = state_values.shape[1] - 1
        ones = np.ones(num_constraints)
        args += [c * ones for c in constant_values]
        args += [interval_value * ones]

        result = np.empty((num_constraints, state_values.shape[0]))

        return f(result, *args).T.flatten()

    return constraints


def general_constraint_jacobian(eom_vector, state_syms, specified_syms,
                                constant_syms, free_constants):
    """Returns a function that evaluates the Jacobian of the constraints.

    Parameters
    ----------
    discrete_eom_vec : sympy.Matrix, shape(n, 1)
        A column vector containing the discrete symbolic expressions of the
        n constraints based on the first order discrete equations of motion.
        This vector should equate to the zero vector.
    state_syms : list of sympy.Functions
        The n functions of time representing the system's states.
    specified_syms : list of sympy.Functions
        The m functions of time representing the system's specified inputs.
    constant_syms : list of sympy.Symbols
        The p symbols representing all of the system's constants.
    free_constants : list of sympy.Symbols
        The q symbols which are a subset of constant_syms that will be free
        to vary in the optimization.

    Returns
    -------
    constraints : function
        A function which returns the numerical values of the constraints at
        time points 2,...,N.

    """
    xi_syms, xp_syms, si_syms, h_sym = \
        discrete_symbols(state_syms, specified_syms)

    # The free parameters are always the n * (N - 1) state values and the
    # user's specified unknown model constants, so the base Jacobian needs
    # to be taken with respect to the ith, and ith - 1 states, and the free
    # model constants.
    # TODO : This needs to eventually support unknown specified inputs too.
    partials = xi_syms + xp_syms + free_constants

    # The arguments to the Jacobian function include all of the free
    # Symbols/Functions in the matrix expression.
    args = xi_syms + xp_syms + si_syms + constant_syms + [h_sym]

    symbolic_jacobian = eom_vector.jacobian(partials)

    jac = ufuncify_matrix(args, symbolic_jacobian)

    # jac is now a function that takes arguments that are made up of all the
    # variables in the discretized equations of motion. It will be used to
    # build the sparse constraint gradient matrix. This Jacobian function
    # returns the non-zero elements needed to build the sparse constraint
    # gradient.

    num_free_constants = len(free_constants)

    def constraints_jacobian(state_values, specified_values,
                             constant_values, interval_value):
        """Returns a sparse matrix of constraint gradient given all of the
        unknowns in the equations of motion over the 2, ..., N time steps.

        Parameters
        ----------
        states : ndarray, shape(n, N)
            The array of n states through N time steps.
        specified_values : ndarray, shape(m, N) or shape(N,)
            The array of m specified inputs through N time steps.
        constant_values : ndarray, shape(p,)
            The array of p constants.
        interval_value : float
            The value of the dicretization time interval.

        Returns
        -------
        constraints_gradient : scipy.sparse.csr_matrix, shape(2 * (N-1), n * N + p)
            A compressed sparse row matrix containing the gradient of the
            constraints where the constaints are along the rows and the free
            parameters are along the columns.

        """
        if state_values.shape[0] < 2:
            raise ValueError('There should always be at least two states.')

        x_current = state_values[:, 1:]  # n x N - 1
        x_previous = state_values[:, :-1]  # n x N - 1

        num_states = state_values.shape[0]  # n
        num_time_steps = state_values.shape[1]  # N
        num_constraint_nodes = num_time_steps - 1  # N - 1

        num_constraints = num_states * (num_constraint_nodes)
        num_free = num_states * num_time_steps + num_free_constants

        # 2n x N - 1
        args = [x for x in x_current] + [x for x in x_previous]

        # 2n + m x N - 1
        if len(specified_values.shape) == 2:
            args += [s for s in specified_values[:, 1:]]
        else:
            args += [specified_values[1:]]

        # These are scalars so, for now, we need to create arrays for these
        # because my version of ufuncify only works with arrays for all
        # arguments. These are generally very short lists of constants, so
        # it shouldn't be that much overhead.

        ones = np.ones(num_constraint_nodes)
        args += [c * ones for c in constant_values]
        args += [interval_value * ones]

        result = np.empty((num_constraint_nodes,
                           symbolic_jacobian.shape[0] *
                           symbolic_jacobian.shape[1]))

        # shape(N - 1, n, 2*n+p) where p is len(free_constants)
        non_zero_derivatives = jac(result, *args)

        # Now loop through the N - 1 constraint nodes to compute the
        # non-zero entries to the gradient matrix (the partials for n states
        # will be computed at each iteration).
        num_partials = (non_zero_derivatives.shape[1] *
                        non_zero_derivatives.shape[2])
        # TODO : The ordered Jacobian values may be able to be gotten by
        # simply flattening non_zero_derivatives.
        jac_vals = np.empty(num_partials * num_constraint_nodes, dtype=float)
        jac_row_idxs = np.empty(len(jac_vals), dtype=int)
        jac_col_idxs = np.empty(len(jac_vals), dtype=int)

        # TODO : Move the computation of the indices out of this function as
        # it only needs to happen one time per problem.

        for i in range(num_constraint_nodes):
            # n: num_states
            # m: num_specified
            # p: num_free_constants

            # the states repeat every N - 1 constraints
            # row_idxs = [0 * (N - 1), 1 * (N - 1),  2 * (N - 1),  n * (N - 1)]

            row_idxs = [j * (num_constraint_nodes) + i
                        for j in range(num_states)]

            # The derivative columns are in this order:
            # [x1i, x2i, ..., xni, x1p, x2p, ..., xnp, p1, ..., pp]
            # So we need to map them to the correct column.

            # first row, the columns indices mapping is:
            # [1, N + 1, ..., N - 1] : [x1p, x1i, 0, ..., 0]
            # [0, N, ..., 2 * (N - 1)] : [x2p, x2i, 0, ..., 0]
            # [-p:] : p1,..., pp  the free constants

            # i=0: [1, ..., n * N + 1, 0, ..., n * N + 0, n * N:n * N + p]
            # i=1: [2, ..., n * N + 2, 1, ..., n * N + 1, n * N:n * N + p]
            # i=2: [3, ..., n * N + 3, 2, ..., n * N + 2, n * N:n * N + p]

            col_idxs = [j * num_time_steps + i + 1 for j in range(num_states)]
            col_idxs += [j * num_time_steps + i for j in range(num_states)]
            col_idxs += [num_states * num_time_steps + j
                         for j in range(num_free_constants)]

            row_idx_permutations = np.repeat(row_idxs, len(col_idxs))
            col_idx_permutations = np.array(list(col_idxs) * len(row_idxs),
                                            dtype=int)

            start = i * num_partials
            stop = (i + 1) * num_partials
            jac_row_idxs[start:stop] = row_idx_permutations
            jac_col_idxs[start:stop] = col_idx_permutations
            jac_vals[start:stop] = non_zero_derivatives[i].flatten()

        return jac_row_idxs, jac_col_idxs, jac_vals

    return constraints_jacobian


def wrap_constraint(func, num_time_steps, num_states,
                    interval_value, constant_syms, specified_syms,
                    fixed_constants, fixed_specified):
    """Returns a function that evaluates all of the constraints or Jacobian
    of the constraints given the system's free parameters.

    Parameters
    ----------
    func : function
        A function that takes the full parameter set an evaulates the
        constraint functions or the Jacobian of the contraint functions.
        i.e. the output of general_constraint or general_jacobian.
    num_time_steps : integer
        The number of time steps.
    num_states : integer
        The number of states in the system.
    interval_value : float
        The interval between the time steps.
    constant_syms : list of sympy.Symbols
        A list of all the constants in system constraint equations.
    specified_syms : list of sympy.Functions
        A list of all the discrete specified inputs.
    fixed_constants : dictionary
        A map of all the system constants which are not free optimization
        parameters to their fixed values.
    fixed_specified : dictionary
        A map of all the system's discrete specified inputs that are not
        free optimization parameters to their fixed values.

    Returns
    -------
    func : function
        A function which returns constraint values given the system's free
        parameters.

    """

    num_free_specified = len(specified_syms) - len(fixed_specified)

    def constraints(free):
        """

        Parameters
        ----------
        free : ndarray

        Returns
        -------
        constraints : ndarray, shape(N-1,)
            The array of constraints from t = 2, ..., N.
            [con_1_2, ..., con_1_N, con_2_2, ..., con_2_N, ..., con_n_2, ..., con_n_N]
        """

        free_states, free_specified, free_constants = \
            parse_free(free, num_states, num_free_specified, num_time_steps)

        all_specified = merge_fixed_free(specified_syms, fixed_specified,
                                         free_specified)

        all_constants = merge_fixed_free(constant_syms, fixed_constants,
                                         free_constants)

        return func(free_states, all_specified, all_constants, interval_value)

    return constraints


def merge_fixed_free(syms, fixed, free):
    """Returns an array with the fixed and free

    This assumes that you have the free constants in the correct order.

    """

    merged = []
    n = 0
    for i, s in enumerate(syms):
        if s in fixed.keys():
            merged.append(fixed[s])
        else:
            merged.append(free[n])
            n += 1
    return np.array(merged)


def parse_free(free, n, r, N):
    """Parses the free parameters vector and returns it's components.

    free : ndarray, shape(n * N + m * M + q)
        The free parameters of the system.
    n : integer
        The number of states.
    r : integer
        The number of free specified inputs.
    N : integer
        The number of time steps.

    Returns
    -------
    states : ndarray, shape(n, N)
        The array of n states through N time steps.
    specified_values : ndarray, shape(r, N) or shape(N,), or None
        The array of r specified inputs through N time steps.
    constant_values : ndarray, shape(q,)
        The array of q constants.

    """

    len_states = n * N
    len_specified = r * N

    free_states = free[:len_states].reshape((n, N))

    if r == 0:
        free_specified = None
    else:
        free_specified = free[len_states:len_states + len_specified]
        if r > 1:
            free_specified = free_specified.reshape((r, N))

    free_constants = free[len_states + len_specified:]

    return free_states, free_specified, free_constants


class Problem(ipopt.problem):

    def __init__(self, N, n, q, obj, obj_grad, con, con_jac):
        """

        Parameters
        ----------
        num_discretization_points
        num_states
        num_free_model_parameters
        obj : function
            The objective function.
        obj_grad : function

        """

        num_free_variables = n * N + q
        num_constraints = n * (N-1)

        self.obj = obj
        self.obj_grad = obj_grad
        self.con = con
        self.con_jac = con_jac

        self.con_jac_rows, self.con_jac_cols, values = \
            con_jac(np.random.random(num_free_variables))

        con_bounds = np.zeros(num_constraints)

        super(Problem, self).__init__(n=num_free_variables,
                                      m=num_constraints,
                                      cl=con_bounds,
                                      cu=con_bounds)

        self.output_filename = 'ipopt_output.txt'
        #self.addOption('derivative_test', 'first-order')
        self.addOption('output_file', self.output_filename)
        self.addOption('print_timing_statistics', 'yes')
        self.addOption('linear_solver', 'ma57')

        self.obj_value = []


    def objective(self, free):
        return self.obj(free)

    def gradient(self, free):
        # This should return a column vector.
        return self.obj_grad(free)

    def constraints(self, free):
        # This should return a column vector.
        return self.con(free)

    def jacobianstructure(self):
        return (self.con_jac_rows, self.con_jac_cols)

    def jacobian(self, free):
        return self.con_jac(free)[2]

    def intermediate(self, *args):
        self.obj_value.append(args[2])


def input_force(typ, time):

    if typ == 'sine':
        lateral_force = 8.0 * np.sin(3.0 * 2.0 * np.pi * time)
    elif typ == 'random':
        lateral_force = 8.0 * np.random.random(len(time))
        lateral_force -= lateral_force.mean()
    elif typ == 'zero':
        lateral_force = np.zeros_like(time)
    elif typ == 'sumsines':
        # I took these frequencies from a sum of sines Ron designed for a
        # pilot control problem.
        nums = [7, 11, 16, 25, 38, 61, 103, 131, 151, 181, 313, 523]
        freq = 2 * np.pi * np.array(nums) / 240
        mags = 2.0 * np.ones(len(freq))
        lateral_force = sum_of_sines(mags, freq, time)
    else:
        raise ValueError('{} is not a valid force type.'.format(typ))

    return lateral_force


class Identifier():

    def __init__(self, num_links, duration, sample_rate, init_type,
                 sensor_noise, do_plot, do_animate):

        self.num_links = num_links
        self.duration = duration
        self.sample_rate = sample_rate
        self.init_type = init_type
        self.sensor_noise = sensor_noise
        self.do_plot = do_plot
        self.do_animate = do_animate

    def compute_discretization(self):

        self.num_time_steps = int(self.duration * self.sample_rate) + 1
        self.discretization_interval = 1.0 / self.sample_rate
        self.time = np.linspace(0.0, self.duration, num=self.num_time_steps)

    def generate_eoms(self):
        # Generate the symbolic equations of motion for the two link pendulum on
        # a cart.
        print("Generating equations of motion.")
        self.system = n_link_pendulum_on_cart(self.num_links,
                                              cart_force=True,
                                              joint_torques=True,
                                              spring_damper=True)

        self.mass_matrix = self.system[0]
        self.forcing_vector = self.system[1]
        self.constants_syms = self.system[2]
        self.coordinates_syms = self.system[3]
        self.speeds_syms = self.system[4]
        self.specified_inputs_syms = self.system[5]  # last entry is lateral force

        self.states_syms = self.coordinates_syms + self.speeds_syms

        self.num_states = len(self.states_syms)

    def find_optimal_gains(self):
        # Find some optimal gains for stablizing the pendulum on the cart.
        print('Finding the optimal gains.')
        self.gains = compute_controller_gains(self.num_links)

    def simulate(self):
        # Generate some "measured" data from the simulation.
        print('Simulating the system.')

        self.lateral_force = input_force('sumsines', self.time)

        set_point = np.zeros(self.num_states)

        self.initial_conditions = np.zeros(self.num_states)
        offset = 10.0 * np.random.random((self.num_states / 2) - 1)
        self.initial_conditions[1:self.num_states / 2] = np.deg2rad(offset)

        rhs, args = closed_loop_ode_func(self.system, self.time, set_point,
                                         self.gains, self.lateral_force)

        start = time.clock()
        self.x = odeint(rhs, self.initial_conditions, self.time, args=(args,))
        msg = 'Simulation of {} real time seconds took {} CPU seconds to compute.'
        print(msg.format(self.duration, time.clock() - start))

        self.x_noise = self.x + np.deg2rad(0.25) * np.random.randn(*self.x.shape)
        self.y = output_equations(self.x)
        self.y_noise = output_equations(self.x_noise)
        self.u = self.lateral_force

    def generate_constraint_funcs(self):

        print('Forming the constraint function.')
        # Generate the expressions for creating the closed loop equations of
        # motion.
        control_dict, gain_syms, equil_syms = \
            create_symbolic_controller(self.states_syms, self.specified_inputs_syms[:-1])

        self.num_gains = len(gain_syms)

        eq_dict = dict(zip(equil_syms, self.num_states * [0]))

        # This is the symbolic closed loop continuous system.
        closed = symbolic_constraints(self.mass_matrix, self.forcing_vector,
                                      self.states_syms, control_dict, eq_dict)

        # This is the discretized (backward euler) version of the closed loop
        # system.
        dclosed = discretize(closed, self.states_syms, self.specified_inputs_syms)

        # Now generate a function which evaluates the N-1 constraints.
        start = time.clock()
        gen_con_func = general_constraint(dclosed, self.states_syms,
                                          [self.specified_inputs_syms[-1]],
                                          self.constants_syms + gain_syms)
        msg = 'Compilation of constraint function took {} CPU seconds.'
        print(msg.format(time.clock() - start))

        self.con_func = wrap_constraint(gen_con_func,
                                        self.num_time_steps,
                                        self.num_states,
                                        self.discretization_interval,
                                        self.constants_syms + gain_syms,
                                        [self.specified_inputs_syms[-1]],
                                        constants_dict(self.constants_syms),
                                        {self.specified_inputs_syms[-1]: self.u})

        start = time.clock()
        gen_con_jac_func = general_constraint_jacobian(dclosed,
                                                       self.states_syms,
                                                       [self.specified_inputs_syms[-1]],
                                                       self.constants_syms + gain_syms,
                                                       gain_syms)
        msg = 'Compilation of constraint Jacobian function took {} CPU seconds.'
        print(msg.format(time.clock() - start))

        self.con_jac_func = wrap_constraint(gen_con_jac_func,
                                            self.num_time_steps,
                                            self.num_states,
                                            self.discretization_interval,
                                            self.constants_syms + gain_syms,
                                            [self.specified_inputs_syms[-1]],
                                            constants_dict(self.constants_syms),
                                            {self.specified_inputs_syms[-1]: self.u})

    def generate_objective_funcs(self):
        print('Forming the objective function.')

        self.obj_func = wrap_objective(objective_function,
                                       self.num_time_steps,
                                       self.num_states,
                                       self.discretization_interval,
                                       self.time,
                                       self.y_noise if self.sensor_noise else self.y)

        self.obj_grad_func = wrap_objective(objective_function_gradient,
                                            self.num_time_steps,
                                            self.num_states,
                                            self.discretization_interval,
                                            self.time,
                                            self.y_noise if self.sensor_noise else self.y)

    def optimize(self):

        print('Solving optimization problem.')

        self.prob = Problem(self.num_time_steps,
                            self.num_states,
                            self.num_gains,
                            self.obj_func,
                            self.obj_grad_func,
                            self.con_func,
                            self.con_jac_func)

        init_states, init_specified, init_constants = \
            parse_free(self.initial_guess, self.num_states, 0, self.num_time_steps)
        init_gains = init_constants.reshape(self.gains.shape)

        self.solution, info = self.prob.solve(self.initial_guess)

        self.sol_states, sol_specified, sol_constants = \
            parse_free(self.solution, self.num_states, 0, self.num_time_steps)
        sol_gains = sol_constants.reshape(self.gains.shape)

        print("Initial gain guess: {}".format(init_gains))
        print("Known gains: {}".format(self.gains))
        print("Identified gains: {}".format(sol_gains))

    def plot(self):

        plot_sim_results(self.y_noise if self.sensor_noise else self.y,
                         self.u)
        plot_constraints(self.con_func(self.initial_guess),
                         self.num_states,
                         self.num_time_steps,
                         self.states_syms)
        plot_constraints(self.con_func(self.solution),
                         self.num_states,
                         self.num_time_steps,
                         self.states_syms)
        plot_identified_state_trajectory(self.sol_states,
                                         self.x.T,
                                         self.states_syms)

    def animate(self, filename=None):

        animate_pendulum(self.time, self.x, 1.0, filename)

    def store_results(self):

        results = parse_ipopt_output(self.prob.output_filename)

        results["datetime"] = int((datetime.datetime.now() -
                                   datetime.datetime(1970, 1, 1)).total_seconds())

        results["num_links"] = self.num_links
        results["sim_duration"] = self.duration
        results["sample_rate"] = self.sample_rate
        results["sensor_noise"] = self.sensor_noise
        results["init_type"] = self.init_type

        hasher = hashlib.sha1()
        string = ''.join([str(v) for v in results.values()])
        hasher.update(string)
        results["run_id"] = hasher.hexdigest()

        known_solution = choose_initial_conditions('known', self.x, self.gains)
        results['initial_guess'] = self.initial_guess
        results['known_solution'] = known_solution
        results['optimal_solution'] = self.solution

        results['initial_guess_constraints'] = self.con_func(self.initial_guess)
        results['known_solution_constraints'] = self.con_func(known_solution)
        results['optimal_solution_constraints'] = self.con_func(self.solution)

        results['initial_conditions'] = self.initial_conditions
        results['lateral_force'] = self.lateral_force

        file_name = 'inverted_pendulum_direct_collocation_results.h5'

        add_results(file_name, results)

    def cleanup(self):

        os.system('rm multibody_system*')

    def identify(self):
        msg = """Running an identification for a {} link inverted pendulum with a {} second simulation discretized at {} hz."""
        msg = msg.format(self.num_links, self.duration, self.sample_rate)
        print('+' * len(msg))
        print(msg)
        print('+' * len(msg))

        self.compute_discretization()
        self.generate_eoms()
        self.find_optimal_gains()
        self.simulate()
        self.generate_constraint_funcs()
        self.generate_objective_funcs()
        self.initial_guess = \
            choose_initial_conditions(self.init_type,
                                      self.x_noise if self.sensor_noise else self.x,
                                      self.gains)
        self.optimize()
        self.store_results()

        if self.do_plot:
            self.plot()

        if self.do_animate:
            self.animate()

        self.cleanup()


def parse_ipopt_output(file_name):
    """Returns a dictionary with the IPOPT summary results.

    Notes
    -----

    This is an example of the summary at the end of the file:

    Number of Iterations....: 1013

                                       (scaled)                 (unscaled)
    Objective...............:   2.8983286604029537e-04    2.8983286604029537e-04
    Dual infeasibility......:   4.7997817057236348e-09    4.7997817057236348e-09
    Constraint violation....:   9.4542809291867735e-09    9.8205754639479892e-09
    Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
    Overall NLP error.......:   9.4542809291867735e-09    9.8205754639479892e-09


    Number of objective function evaluations             = 6881
    Number of objective gradient evaluations             = 1014
    Number of equality constraint evaluations            = 6900
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 1014
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 0
    Total CPU secs in IPOPT (w/o function evaluations)   =     89.023
    Total CPU secs in NLP function evaluations           =    457.114

    """

    with open(file_name, 'r') as f:
        output = f.readlines()

    results = {}

    lines_of_interest = output[-50:]
    for line in lines_of_interest:
        if 'Number of Iterations' in line and 'Maximum' not in line:
            results['num_iterations'] = int(line.split(':')[1].strip())

        elif 'Number of objective function evaluations' in line:
            results['num_obj_evals'] = int(line.split('=')[1].strip())

        elif 'Number of objective gradient evaluations' in line:
            results['num_obj_grad_evals'] = int(line.split('=')[1].strip())

        elif 'Number of equality constraint evaluations' in line:
            results['num_con_evals'] = int(line.split('=')[1].strip())

        elif 'Number of equality constraint Jacobian evaluations' in line:
            results['num_con_jac_evals'] = int(line.split('=')[1].strip())

        elif 'Total CPU secs in IPOPT (w/o function evaluations)' in line:
            results['time_ipopt'] = float(line.split('=')[1].strip())

        elif 'Total CPU secs in NLP function evaluations' in line:
            results['time_func_evals'] = float(line.split('=')[1].strip())

    return results


def create_database(file_name):
    """Creates an empty optimization results database on disk if it doesn't
    exist."""

    class RunTable(tables.IsDescription):
        run_id = tables.StringCol(40)  # sha1 hashes are 40 char long
        init_type = tables.StringCol(10)
        datetime = tables.Time32Col()
        num_links = tables.Int32Col()
        sim_duration = tables.Float32Col()
        sample_rate = tables.Float32Col()
        sensor_noise = tables.BoolCol()
        num_iterations = tables.Int32Col()
        num_obj_evals = tables.Int32Col()
        num_obj_grad_evals = tables.Int32Col()
        num_con_evals = tables.Int32Col()
        num_con_jac_evals = tables.Int32Col()
        time_ipopt = tables.Float32Col()
        time_func_evals = tables.Float32Col()

    if not os.path.isfile(file_name):
        h5file = tables.open_file(file_name,
                                  mode='w',
                                  title='Inverted Pendulum Direct Collocation Results')
        h5file.create_table('/', 'results', RunTable, 'Optimization Results Table')
        h5file.create_group('/', 'arrays', 'Optimization Parameter Arrays')

        h5file.close()


def add_results(file_name, results):

    if not os.path.isfile(file_name):
        create_database(file_name)

    h5file = tables.open_file(file_name, mode='a')

    print('Adding run {} to the database.'.format(results['run_id']))

    run_array_dir = h5file.create_group(h5file.root.arrays,
                                        results['run_id'],
                                        'Optimization Run #{}'.format(results['run_id']))
    arrays = ['initial_guess',
              'known_solution',
              'optimal_solution',
              'initial_guess_constraints',
              'known_solution_constraints',
              'optimal_solution_constraints',
              'initial_conditions',
              'lateral_force']

    for k in arrays:
        v = results.pop(k)
        h5file.create_array(run_array_dir, k, v)

    table = h5file.root.results
    opt_row = table.row

    for k, v in results.items():
        opt_row[k] = v

    opt_row.append()

    table.flush()

    h5file.close()


def choose_initial_conditions(typ, x, gains):

    free_states = x.T.flatten()
    free_gains = gains.flatten()

    if typ == 'known':
        initial_guess = np.hstack((free_states, free_gains))
    elif typ == 'zero':
        initial_guess = np.hstack((free_states, 0.1 * np.ones_like(free_gains)))
    elif typ == 'ones':
        initial_guess = np.hstack((free_states, np.ones_like(free_gains)))
    elif typ == 'close':
        gain_mod = 0.5 * np.abs(free_gains) * np.random.randn(len(free_gains))
        initial_guess = np.hstack((free_states, free_gains + gain_mod))
    elif typ == 'random':
        initial_guess = np.hstack((x.T.flatten(),
                                   100.0 * np.random.randn(len(free_gains))))

    return initial_guess


def load_results_table(filename):

    handle = tables.openFile(filename, 'r')
    df = pandas.DataFrame.from_records(handle.root.results[:])
    handle.close()
    return df


def load_run(filename, run_id):
    handle = tables.openFile(filename, 'r')
    group = getattr(handle.root.arrays, run_id)
    d = {}
    for array_name in group.__members__:
        d[array_name] = getattr(group, array_name)[:]
    handle.close()
    return d

def compute_gain_error(filename):
    # root mean square of gain error
    df = load_results_table(filename)
    rms = []
    for run_id, sim_dur, sample_rate in zip(df['run_id'],
                                            df['sim_duration'],
                                            df['sample_rate']):
        run_dict = load_run(filename, run_id)
        num_states = len(run_dict['initial_conditions'])
        num_time_steps = int(sim_dur * sample_rate)
        __, __, known_gains = parse_free(run_dict['known_solution'],
                                         num_states, 0, num_time_steps)
        __, __, optimal_gains = parse_free(run_dict['optimal_solution'],
                                           num_states, 0, num_time_steps)
        rms.append(np.sqrt(np.sum((known_gains - optimal_gains)**2)))
    df['RMS of Gains'] = np.asarray(rms)
    return df


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run N-Line System ID")

    parser.add_argument('-n', '--numlinks', type=int,
        help="The number of links in the pendulum.", default=1)

    parser.add_argument('-d', '--duration', type=float,
        help="The duration of the simulation in seconds.", default=1.0)

    parser.add_argument('-s', '--samplerate', type=float,
        help="The sample rate of the discretization.", default=50.0)

    parser.add_argument('-i', '--initialconditions', type=str,
        help="The type of initial conditions.", default='random')

    parser.add_argument('-r', '--sensornoise', action="store_true",
        help="Add noise to sensor data.",)

    parser.add_argument('-a', '--animate', action="store_true",
        help="Show the pendulum animation.",)

    parser.add_argument('-p', '--plot', action="store_true",
        help="Show result plots.")

    args = parser.parse_args()

    identifier = Identifier(args.numlinks, args.duration,
                            args.samplerate, args.initialconditions,
                            args.sensornoise, args.plot, args.animate)
    identifier.identify()
