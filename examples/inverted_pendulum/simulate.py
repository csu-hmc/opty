#!/usr/bin/env/python

# standard lib
from collections import OrderedDict

# external
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import solve_continuous_are
from pydy.codegen.ode_function_generators import generate_ode_function
from opty.utils import controllable, sum_of_sines

# local
from model import n_link_pendulum_on_cart


def constants_dict(constants):
    """Returns an ordered dictionary which maps the system constant symbols
    to numerical values. The cart spring is set to 10.0 N/m, the cart damper
    to 5.0 Ns/m and gravity is set to 9.81 m/s and the masses and lengths of
    the pendulums are all set to make the human sized."""
    return OrderedDict(zip(constants, [10.0, 5.0, 9.81] +
                           (len(constants) - 1) * [1.0]))


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


def input_force(typ, time):

    magnitude = 8.0  # Newtons

    if typ == 'sine':
        lateral_force = magnitude * np.sin(3.0 * 2.0 * np.pi * time)
    elif typ == 'random':
        lateral_force = 2.0 * magnitude * np.random.random(len(time))
        lateral_force -= lateral_force.mean()
    elif typ == 'zero':
        lateral_force = np.zeros_like(time)
    elif typ == 'sumsines':
        # I took these frequencies from a sum of sines Ron designed for a
        # pilot control problem.
        nums = [7, 11, 16, 25, 38, 61, 103, 131, 151, 181, 313, 523]
        freq = 2.0 * np.pi * np.array(nums, dtype=float) / 240.0
        lateral_force = sum_of_sines(magnitude, freq, time)[0]
    else:
        raise ValueError('{} is not a valid force type.'.format(typ))

    return lateral_force


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

    mass_matrix = system[0]
    forcing = system[1]
    constants = system[2]
    coordinates = system[3]
    speeds = system[4]
    specifieds = system[5]

    rhs = generate_ode_function(forcing,
                                coordinates,
                                speeds,
                                constants,
                                mass_matrix=mass_matrix,
                                specifieds=specifieds,
                                generator='cython')

    args = (controller, constants_dict(constants))

    return rhs, args
