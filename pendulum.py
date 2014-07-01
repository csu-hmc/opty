#!/usr/bin/env python

"""This script demonstrates an attempt at identifying the controller for a
two link inverted pendulum on a cart by direct collocation. I collect
"measured" data from the system by simulating it with a known optimal
controller under the influence of random lateral force perturbations. I then
form the optimization problem such that we minimize the error in the model's
simulated outputs with respect to the measured outputs. The optimizer
searches for the best set of controller gains (which are unknown) that
reproduce the motion and ensure the dynamics are valid.

Dependencies this runs with:

    numpy 1.8.1
    scipy 0.14.1
    sympy 0.7.5
    matplotlib 1.3.1
    pydy 0.2.1

"""

# standard lib
from collections import OrderedDict

# external
import numpy as np
import sympy as sym
from scipy.interpolate import interp1d
from scipy.linalg import solve_continuous_are
from scipy.integrate import odeint
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from pydy.codegen.code import generate_ode_function
from pydy.codegen.tests.models import \
    generate_n_link_pendulum_on_cart_equations_of_motion as n_link_pendulum_on_cart


def constants_dict(constants):
    """Returns an ordered dictionary which maps the system constant symbols
    to numerical values. Gravity is set to 9.81 m/s and the masses and
    lengths of the pendulums are all set to 1.0 kg and m, respectively."""
    return OrderedDict(zip(constants, [9.81] + (len(constants) - 1) * [1.0]))


def state_derivatives(states):
    """Returns functions of time which represent the time derivatives of the
    states."""
    return [state.diff() for state in states]


def build_constraint(mass_matrix, forcing_vector, states):
    """Returns Fr + Fr* from the mass_matrix and forcing vector."""

    xdot = state_derivatives(states)

    return mass_matrix * xdot - forcing_vector


def compute_controller_gains():
    """Returns a numerical gain matrix that can be multiplied by the error
    in the states to generate the joint torques needed to stabilize the
    pendulum.

    u(t) = K * [x_eq - x(t)]

    Returns
    -------
    K : ndarray, shape(2, 6)
        The gains needed to compute joint torques.

    """

    res = n_link_pendulum_on_cart(2, cart_force=False, joint_torques=True)

    mass_matrix = res[0]
    forcing_vector = res[1]
    constants = res[2]
    coordinates = res[3]
    speeds = res[4]
    specified = res[5]

    states = coordinates + speeds

    # all angles at pi/2 for the pendulum to be inverted
    equilibrium_point = np.hstack((0.0,
                                   np.pi / 2.0 * np.ones(len(coordinates) - 1),
                                   np.zeros(len(speeds)) ))
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

    xeq = sym.Matrix(num_states, 1, lambda i, j:
                     sym.Symbol('xeq_{}'.format(i)))

    K = sym.Matrix(num_inputs, num_states, lambda i, j:
                   sym.Symbol('k_{}{}'.format(i, j)))

    x = sym.Matrix(states).reshape(num_states, 1)
    T = sym.Matrix(inputs).reshape(num_inputs, 1)

    gain_symbols = [k for k in K]

    controller_dict = sym.solve(T + K * (xeq - x), inputs)

    return controller_dict, gain_symbols, xeq


def output_equations(x):
    """Returns an array of the generalized coordinates.

    Parameters
    ----------
    x : ndarray, shape(n, 6)
        The trajectories of the system states.

    Returns
    -------
    y : ndarray, shape(n, 3)
        The trajectories of the generalized coordinates.

    """

    return x[:, :3]


def simulate_system(system, duration, num_steps, controller_gain_matrix):
    """This simulates the closed loop system under later force
    perturbations.

    Parameters
    ----------
    system : tuple, len(6)
        The output of the symbolic EoM generator.
    duration : float
        The duration of the simulation.
    num_steps : integer
        The number of time steps.
    controller_gain_matrix : ndarray, shape(2, 6)
        The gain matrix that computes the optimal joint torques given the
        system state.

    Returns
    -------
    y : ndarray, shape(n, 3)
        The trajectories of the generalized coordinates.
    x : ndarray, shape(n, 6)
        The trajectories of the states.
    lateral_force : ndarray, shape(n,)
        The applied lateral force.

    """

    time = np.linspace(0.0, duration, num=num_steps)

    constants = system[2]
    coordinates = system[3]
    speeds = system[4]
    specified = system[5]

    states = coordinates + speeds

    lateral_force = 8.0 * np.random.random(len(time))
    lateral_force -= lateral_force.mean()
    interp_func = interp1d(time, lateral_force)

    equilibrium_point = np.hstack((0.0,
                                   np.pi / 2.0 * np.ones(len(coordinates) - 1),
                                   np.zeros(len(speeds))))

    def specified(x, t):
        joint_torques = np.dot(controller_gain_matrix, equilibrium_point - x)
        if t > time[-1]:
            lateral_force = interp_func(time[-1])
        else:
            lateral_force = interp_func(t)
        return np.hstack((joint_torques, lateral_force))

    rhs = generate_ode_function(*system)

    args = {'constants': constants_dict(constants).values(),
            'specified': specified}

    initial_conditions = np.zeros(len(states))
    initial_conditions[1:len(states) / 2] = np.pi / 2.0

    x = odeint(rhs, initial_conditions, time, args=(args,))
    y = output_equations(x)

    return y, x, lateral_force


def create_dynamics_constraints(controller_dict, gain_symbols,
                                equilibrium_symbols,
                                lateral_force_trajectory):
    """Return a sequence of dictionaries where each dictionary defines a
    constraint equation and it's Jacobian.

    The SLSQP solver in SciPy wants all of the constraints in this form:

        ({'type': 'eq', 'fun': f, 'jac': g, args=(...)},
         ...
         {'type': 'eq', 'fun': f, 'jac': g, args=(...)}

    The constraint functions and the Jacobians take the arguments:

        fun(free, i, lateral_force_trajectory)
        jac(free, i, lateral_force_trajectory)

    where:

        free : ndarray, shape(n * 6 + 12,)
            The free parameters in the optimization problem, i.e. the system
            states at each time step plus the controller gains.
        i : integer
            The ith time step at which this constraint should be valid.
        lateral_force_trajectory : ndarray, shape(n,)
            The known applied lateral force.

    """

    res = n_link_pendulum_on_cart(2, cart_force=True, joint_torques=True)

    mass_matrix = res[0]
    forcing_vector = res[1]
    constants = res[2]
    coordinates = res[3]
    speeds = res[4]
    lateral_force_symbol = res[5][-1]

    state_symbols = coordinates + speeds

    # Create a vector of x' symbols.
    xdot_symbols = state_derivatives(state_symbols)
    xdot_replacement_symbols = sym.symbols('xd:{}'.format(len(xdot_symbols)))
    xdot = sym.Matrix(xdot_replacement_symbols).reshape(len(state_symbols), 1)

    # Note for sympy issues: if you have x(t) and Derivative(x(t), t) in an
    # expression and try to lambdify with both as args, then
    # Derivative(x(t), t) will be replaced with Derivative(Dummy_3435, t).
    # Lamdify could set args from the top down instead of bottom up to
    # prevent this.

    # First, substitute controller expressions for the joint torques. This
    # creates a vector expression, f(x', x, u), shape(6, 1).
    constraint_eq = (mass_matrix * xdot - forcing_vector).subs(controller_dict)

    # Substitute all model constants numerical values.
    constraint_eq = constraint_eq.subs(constants_dict(constants))

    # Substitute the known equilibrium point numerical values.
    equilibrium_point = np.hstack((0.0,
                                   np.pi / 2.0 * np.ones(len(coordinates) - 1),
                                   np.zeros(len(speeds))))
    equilibrium_dict = dict(zip(equilibrium_symbols, equilibrium_point))
    constraint_eq = constraint_eq.subs(equilibrium_dict)

    # Compute the symbolic Jacobian matrix with respect to the free
    # parameters, shape(6, 18) or shape(18, 6).
    # TODO : Check whether Jacobian is computed across rows or columns.
    # TODO : It might be good to replace the x' terms with (x - xp) / h so
    # that these Jacobians reflect change of x' due to x.
    constraint_jacobian = constraint_eq.jacobian(state_symbols + gain_symbols)

    # Create a numerical functions that can be evaluated for any given xdot,
    # x, u, and system gains.

    args = list(xdot_replacement_symbols) + state_symbols + [lateral_force_symbol] + gain_symbols

    func = sym.lambdify(args, constraint_eq,
                        modules=({'ImmutableMatrix': np.array}, 'numpy'))

    jac = sym.lambdify(args, constraint_jacobian,
                       modules=({'ImmutableMatrix': np.array}, 'numpy'))

    def constraint_function(free, i, lateral_force_trajectory):
        """Evaluates the equation of motion constraint equation.

        Parameters
        ----------
        free : ndarray, shape(n * 6 + 2 * 6)
            The free optimization parameters, i.e. the 6 states at n time
            instances followed by the 12 controller gains.
        i : integer
            The discrete instance to form the constraint equation.
        lateral_force_trajectory : ndarray, shape(n,)
            The specified trajectory of the applied lateral force.

        Returns
        -------
        constraint_values : ndarray, shape(6,)
            The constraint values at the ith time instance.

        """
        n = len(lateral_force_trajectory)
        state_trajectory = free[:n * 6].reshape((n, 6))  # shape(n, 6)
        gains = free[n * 6:]  # shape(12,)

        xdot = compute_xdot(i, state_trajectory, 1.0/100.0)  # shape(6,)

        values = np.hstack((xdot, state_trajectory[i],
                            lateral_force_trajectory[i],
                            gains))  # shape(6 + 6 + 1 + 12)

        return func(*values)

    def constraint_jacobian_function(free, i, lateral_force_trajectory):
        """Evaluates the equation of motion contraint equation.

        Parameters
        ----------
        free : ndarray, shape(n * 6 + 2 * 6)
            The free optimization parameters, i.e. the 6 states at n time
            instances followed by the 12 controller gains.
        i : integer
            The discrete instance to form the constraint equation.
        lateral_force_trajectory : ndarray, shape(n,)
            The specified trajectory of the applied lateral force.

        Returns
        -------
        constraint_jacobian_values : ndarray, shape(6,)
            The Jacobian of the constraints at the ith time instance.

        """
        n = len(lateral_force_trajectory)
        state_trajectory = free[:n * 6].reshape((n, 6))  # shape(n, 6)
        gains = free[n * 6:]  # shape(12,)

        xdot = compute_xdot(i, state_trajectory, 1.0 / 100.0)  # shape(6,)

        values = np.hstack((xdot, state_trajectory[i], lateral_force_trajectory[i],
                            gains))  # shape(6 + 6 + 1 + 12)

        # TODO : The full Jacobian should be 6 x (6 * n + 12), but most
        # entries are zeros, i.e. all of the entries not associated with the
        # ith state or the gains.

        return jac(*values)

    def constraint_function_vectorized(free, lateral_force_trajectory):
        """Evaluates the equation of motion contraint equation.

        Parameters
        ----------
        free : ndarray, shape(n + 2 * 6)
            The n states followed by the 12 controller gains.

        Returns
        -------
        constraint_values : ndarray, shape(n,)
            The values of all of the constraint at each time instance.

        """
        # This function maybe able to be used directly in fmin_slsqp for the
        # arg f_eqcons.

        state_trajectory = free[:n]  # shape(n, 6)
        gains = np.tile(free[n:], (10, 1))  # shape(n, 12)

        xdot = compute_xdot(state_trajectory)  # shape(n, 6)

        values = np.hstack((xdot, state_trajectory,
                            lateral_force_trajectory, gains))  # shape(n, 6 + 6 + 1 + 12)

        return np.transpose(func(*values.T), (2, 0, 1))  # shape(n, 6)

    # Now for each time instance, define a constraint dictionary.
    n = len(lateral_force_trajectory)
    constraints = []
    for i in range(n):
        constraints.append({'type': 'eq',
                            'fun': constraint_function,
                            'jac': constraint_jacobian_function,
                            'args': (i, lateral_force_trajectory)})

    return constraints


def objective_function(free, y_measured):
    """Create and objective function which minimizes the error in the
    outputs with respect to the measured values.

    The arguments to the objective function should be a vector with the
    states at each time point, x, lateral force at each time point, u, and
    the controller gains.

    Parameters
    ----------
    free : ndarray, shape(N x M + 2 x 6,)
        The flattened state array with M states at N time points and the 2
        x 6 controller gains: [x11, ..., x1N, ..., xM1, xMN, k1, ..., k12]
    y_measured : ndarray, shape(N, 3)
        The measured trajectories of the output variables (generalized
        coordinates for now) at each time instance (this should current
        correspond to the time steps in the direct collocation but
        interpolation should eventually be used to separate the two).

    Returns
    -------
    cost : float
        The cost value.

    """
    n = len(y_measured)
    # TODO : Make sure this reshape does the right thing.
    state_trajectory = free[:n * 6].reshape((n, 6))  # shape(n, 6)

    return np.sum(((output_equations(state_trajectory) - y_measured)**2))


def objective_function_jacobian(free, y_measured):

    n = len(y_measured)
    state_trajectory = free[:n * 6].reshape((n, 6))  # shape(n, 6)

    dobj_dfree = np.zeros(len(free))
    dobj_dfree[:n*6] = 2.0 * (state_trajectory[:, :3] -
                              y_measured).reshape(n * 6)

    return dobj_dfree


def compute_xdot(i, x, h):
    """Returns a local approximation of the derivative of x at i using time
    step h.

    Parameters
    ----------
    i : integer
    x : ndarray, shape(n, 6)
    h : float

    Returns
    dxdt : ndarra, shape(6,)
    """
    try:
        return (x[i] - x[i - 1]) / h
    except IndexError:
        return np.zeros(6)


def animate_pendulum(t, states, length, filename=None):
    """Animates the n-pendulum and optionally saves it to file.

    Parameters
    ----------
    t : ndarray, shape(m)
        Time array.
    states: ndarray, shape(m,p)
        State time history.
    length: float
        The length of the pendulum links.
    filename: string or None, optional
        If true a movie file will be saved of the animation. This may take
        some time.

    """
    # the number of pendulum bobs
    numpoints = states.shape[1] / 2

    # first set up the figure, the axis, and the plot elements we want to
    # animate
    fig = plt.figure()

    # some dimesions
    cart_width = 0.4
    cart_height = 0.2

    # set the limits based on the motion
    xmin = np.around(states[:, 0].min() - cart_width / 2.0, 1)
    xmax = np.around(states[:, 0].max() + cart_width / 2.0, 1)

    # create the axes
    ax = plt.axes(xlim=(xmin, xmax), ylim=(-1.1, 2.1), aspect='equal')

    # display the current time
    time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes)

    # create a rectangular cart
    rect = Rectangle([states[0, 0] - cart_width / 2.0, -cart_height / 2],
        cart_width, cart_height, fill=True, color='red', ec='black')
    ax.add_patch(rect)

    # blank line for the pendulum
    line, = ax.plot([], [], lw=2, marker='o', markersize=6)

    # initialization function: plot the background of each frame
    def init():
        time_text.set_text('')
        rect.set_xy((states[0, 0] - cart_width / 2.0,
                     -cart_height / 2.0))
        line.set_data([], [])
        return time_text, rect, line,

    # animation function: update the objects
    def animate(i):
        time_text.set_text('time = {:2.2f}'.format(t[i]))
        rect.set_xy((states[i, 0] - cart_width / 2.0, -cart_height / 2))
        x = np.hstack((states[i, 0], np.zeros((numpoints - 1))))
        y = np.zeros((numpoints))
        for j in np.arange(1, numpoints):
            x[j] = x[j - 1] + length * np.cos(states[i, j])
            y[j] = y[j - 1] + length * np.sin(states[i, j])
        line.set_data(x, y)
        return time_text, rect, line,

    # call the animator function
    anim = animation.FuncAnimation(fig, animate, frames=len(t),
                                   init_func=init,
                                   interval=t[-1] / len(t) * 1000,
                                   blit=False, repeat=False)
    plt.show()

    # save the animation if a filename is given
    if filename is not None:
        anim.save(filename, fps=30, codec='libx264')


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
    h = sym.Symbol('h')

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

    # Note : The Derivative's must be substituted before the symbols.
    eoms = eoms.subs(dict(zip([s.diff() for s in states], euler_formula)))

    eoms = eoms.subs(dict(zip(states + specified, xi + si)))

    return eoms


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
    xi_syms, xp_syms, si_syms, h = discrete_symbols(state_syms, specified_syms)

    args = [x for x in xi_syms] + [x for x in xp_syms]
    args += [s for s in si_syms] + constant_syms + [h]

    modules = ({'ImmutableMatrix': np.array}, 'numpy')
    f = sym.lambdify(args, eom_vector, modules=modules)

    def constraints(state_values, specified_values, constant_values,
                    interval_value):
        """Returns a vector of constraint values give all of the
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

        x_current = state_values[:, 1:]
        x_previous = state_values[:, :-1]

        args = [x for x in x_current] + [x for x in x_previous]

        if len(specified_values.shape) == 2:
            si = specified_values[:, 1:]
            args += [s for s in si]
        else:
            si = specified_values[1:]
            args += [si]

        args += list(constant_values)
        args += [interval_value]

        lam_eval = np.squeeze(f(*args))

        return lam_eval.reshape(x_current.shape[0] * x_current.shape[1])

    return constraints


def constraint_func(func, num_time_steps, num_states,
                    interval_value, constant_syms, specified_syms,
                    fixed_constants, fixed_specified):
    """Returns a function that evaluates all of the constraints.

    Parameters
    ----------
    func : function
        A function that takes the full parameter set an evaulates the
        constraint functions or the gradient of the contraint functions.
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
    optimization_func : function
        A function which returns constraint values given the system's free
        parameters.

    """

    num_free_specified = len(specified_syms) - len(fixed_specified)

    len_states = num_states * num_time_steps
    len_specified = num_free_specified * num_time_steps

    def build_fixed_free(syms, fixed, free):

        all = []
        n = 0
        for i, s in enumerate(syms):
            if s in fixed.keys():
                all.append(fixed[s])
            else:
                all.append(free[n])
                n += 1
        return np.array(all)

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

        free_states = free[:len_states].reshape((num_states, num_time_steps))

        free_specified = free[len_states:len_states + len_specified]
        free_specified = free_specified.reshape((num_free_specified,
                                                 num_time_steps))
        all_specified = build_fixed_free(specified_syms, fixed_specified,
                                         free_specified)

        free_constants = free[len_states + len_specified:]

        all_constants = build_fixed_free(constant_syms, fixed_constants,
                                         free_constants)

        return func(free_states, all_specified, all_constants, interval_value)

    return constraints


def general_gradient(eom_vector, state_syms, specified_syms,
                     constant_syms, free_constants):
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
    free_constants : list of sympy.Symbols
        The p symbols which are a subset of constant_syms that will be free
        to vary in the optimization.

    Returns
    -------
    constraints : function
        A function which returns the numerical values of the constraints at
        time points 2,...,N.

    """
    xi_syms, xp_syms, si_syms, h = discrete_symbols(state_syms, specified_syms)

    # The free parameters are always the n * (N - 1) state values and the
    # user's specified unknown model constants, so the base Jacobian needs
    # to be taken with respect to the ith, and ith - 1 states, and the free
    # model constants.
    partials = xi_syms + xp_syms + free_constants

    # The arguments to the Jacobian function are all free symbols in the
    # matrix expression.
    args = xi_syms + xp_syms + si_syms + constant_syms + [h]

    modules = ({'ImmutableMatrix': np.array}, 'numpy')

    jac = sym.lambdify(args, eom_vector.jacobian(partials), modules=modules)

    # jac is now a function that takes arguments that are made up of all the
    # variables in the discretized equations of motion. It will be used to
    # build the sparse constraint gradient matrix. This Jacobian function
    # returns the non-zero elements needed to build the sparse constraint
    # gradient.

    num_free_constants = len(free_constants)

    def constraints_gradient(state_values, specified_values,
                             constant_values, interval_value):
        """Returns a sparse matrix of constraint gradient given all of the
        unknowns in the equations of motion over the 2, ..., N time steps.

        Parameters
        ----------
        states : ndarray, shape(n, N)
            The array of n states through N time steps.
        specified_values : ndarray, shape(m, N) or shape(N,)
            The array of m specified inputs through N time steps.
        constant_values : ndarray, shape(b,)
            The array of b constants.
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

        gradient_matrix = lil_matrix((2 * (num_time_steps - 1),
                                      num_states * num_time_steps +
                                      num_free_constants))

        # Now loop through the N - 1 constraints to compute the non-zero
        # entries to the gradient matrix (the partials for n states will be
        # computed at each iteration).

        for i in range(num_time_steps - 1):
            # n: num_states
            # m: num_specified
            # p: num_free_constants

            xi = x_current[:, i]  # len(n)
            xp = x_previous[:, i]  # len(n)
            if len(specified_values.shape) < 2:
                si = specified_values[i]  # len(m)
            else:
                si = specified_values[:, i]  # len(m)

            args = np.hstack((xi, xp, si, constant_values, interval_value))

            non_zero_derivatives = jac(*args)  # n x (2*n+p), p is len(free_constants)

            # the states repeat every N - 1 constraints
            # row_idxs = [0 * (N - 1), 1 * (N - 1),  2 * (N - 1),  n * (N - 1)]

            row_idxs = [j * (num_time_steps - 1) + i
                        for j in range(num_states)]

            # The derivative columns are in this order:
            # [x1i, x2i, ..., xni, x1p, x2p, ..., xnp, p1, ..., pp]
            # So we need to map them to the correct column.

            # first row, the columns indices mapping is:
            # [1, N + 1, ..., N - 1] : [x1p, x1i, 0, ..., 0]
            # [0, N, ..., 2 * (N - 1)] : [x2p, x2i, 0, ..., 0]
            # [-p:] : p1,..., pp  the free constants

            # i=0: [1, N + 1, 0, N + 0, n * N:n * N + p]
            # i=1: [2, N + 2, 1, N + 1, n * N:n * N + p]
            # i=2: [3, N + 3, 2, N + 2, n * N:n * N + p]

            col_idxs = [i + 1, i + num_time_steps + 1, i, i + num_time_steps]
            col_idxs += [num_states * num_time_steps + j
                         for j in range(num_free_constants)]

            substitute_matrix(gradient_matrix, row_idxs, col_idxs,
                              non_zero_derivatives)

        return gradient_matrix.tocsr()

    return constraints_gradient


def substitute_matrix(matrix, row_idxs, col_idxs, sub_matrix):
    """Returns the matrix with the values given by the row and column
    indices with those in the sub-matrix.

    Parameters
    ----------
    matrix : ndarray, shape(n, m)
        A matrix (i.e. 2D array).
    row_idxs : array_like, shape(p<=n,)
        The row indices which designate which entries should be replaced by
        the sub matrix entries.
    col_idxs : array_like, shape(q<=m,)
        The column indices which designate which entries should be replaced
        by the sub matrix entries.
    sub_matrix : ndarray, shape(p, q)
        A matrix of values to substitute into the specified rows and
        columns.

    Notes
    -----
    This makes a copy of the sub_matrix, so if it is large it may be slower
    than a more optimal implementation.

    Examples
    --------

    >>> a = np.zeros((3, 4))
    >>> sub = np.arange(4).reshape((2, 2))
    >>> substitute_matrix(a, [1, 2], [0, 2], sub)
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 2.,  0.,  3.,  0.]])

    """

    assert sub_matrix.shape == (len(row_idxs), len(col_idxs))

    row_idx_permutations = np.repeat(row_idxs, len(col_idxs))
    col_idx_permutations = np.array(list(col_idxs) * len(row_idxs))

    matrix[row_idx_permutations, col_idx_permutations] = sub_matrix.flatten()

    return matrix


if __name__ == "__main__":

    # Specify the number of time steps and duration of the data to fit.
    sample_rate = 100.0
    duration = 10.0  # seconds
    num_time_steps = int(duration * sample_rate) + 1

    # Generate the symbolic equations of motion for the two link pendulum on
    # a cart.
    system = n_link_pendulum_on_cart(2, cart_force=True, joint_torques=True)

    # Find some optimal gains for stablizing the pendulum on the cart.
    gains = compute_controller_gains()

    # Generate some "measured" data from the simulation.
    y, x, u = simulate_system(system, duration, num_time_steps, gains)

    # Generate the expressions for creating the closed loop equations of
    # motion.
    controller_dict, gain_symbols, equilibrium_symbols = \
        create_symbolic_controller(system[3] + system[4], system[5][:-1])

    # Generate the n constraint equations.
    constraints = create_dynamics_constraints(controller_dict, gain_symbols,
                                              equilibrium_symbols, u)

    # Give an initial guess for the optimizer. Use the known state
    # trajectories and zeros for the gains.
    initial_guess = np.hstack((x.flatten(), np.zeros(12)))


    # Now try to find the optimal solution using a Sequential Least Squares
    # Programming Method. This may work for this problem, but I will switch
    # out to IPOPT (and a python wrapper) if it doesn't.
    #result = minimize(objective_function, initial_guess, args=(y,),
                      #method='SLSQP', jac=objective_function_jacobian,
                      #constraints=constraints)

    # Plot the simulation results and animate the pendulum.
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(u)
    axes[1].plot(y[:, 0])
    axes[2].plot(np.rad2deg(y[:, 1:]))

    plt.show()

    animate_pendulum(np.linspace(0.0, duration, num_time_steps), x, 1.0)
