#!/usr/bin/env python

"""This script demonstrates an attempt at identifying the controller for a
two link inverted pendulum on a cart by direct collocation. I collect
"measured" data from the system by simulating it with a known opimtal
controller under the influence of random lateral force perturbations. I then
form the optimization problem such that we minimize the error in the model's
simulated outputs with the measured outputs. The optimizer searches for the
best set of gains (which are unknown) that reproduce the motion and ensure
the dynamics are valid.

Dependencies this runs with:

    numpy 1.8.1
    scipy 0.14.1
    sympy 0.7.5
    matplotlib 1.3.1
    pydy HEAD of master

"""

from collections import OrderedDict

import numpy as np
import sympy as sym
from scipy.interpolate import interp1d
from scipy.linalg import solve_continuous_are
from scipy.integrate import odeint
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
    outputs with respect to the measured values and minimizes jouint
    torque

    The arguments to the objective function should be a vector with the
    states at each time point, x, lateral force at each time point, u, and
    the controller gains.


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
        If true a movie file will be saved of the animation. This may take some time.

    """
    # the number of pendulum bobs
    numpoints = states.shape[1] / 2

    # first set up the figure, the axis, and the plot elements we want to animate
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
