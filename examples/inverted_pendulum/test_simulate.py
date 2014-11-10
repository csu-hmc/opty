#!/usr/bin/env python

import numpy as np
from scipy.integrate import odeint
import sympy as sym
import matplotlib.pyplot as plt
from pydy.codegen.code import generate_ode_function
from opty import direct_collocation as dc

import model
import simulate


def test_num_diff(sample_rate=100.0):

    num_links = 1

    # Generate the symbolic equations of motion for the two link pendulum on
    # a cart.
    system = model.n_link_pendulum_on_cart(num_links, cart_force=True,
                                           joint_torques=True,
                                           spring_damper=True)

    mass_matrix = system[0]
    forcing_vector = system[1]
    constants_syms = system[2]
    coordinates_syms = system[3]
    speeds_syms = system[4]
    specified_inputs_syms = system[5]  # last entry is lateral force

    states_syms = coordinates_syms + speeds_syms

    num_states = len(states_syms)

    gains = simulate.compute_controller_gains(num_links)

    # Specify the number of time steps and duration of the measurements.
    num_time_steps = 51
    duration = (num_time_steps - 1) / sample_rate
    discretization_interval = 1.0 / sample_rate

    msg = 'Integrating over {} seconds with {} time steps spaced at {} seconds apart.'
    print(msg.format(duration, num_time_steps, discretization_interval))

    # Integrate the equations of motion.
    time = np.linspace(0.0, duration, num=num_time_steps)

    lateral_force = simulate.input_force('sumsines', time)

    set_point = np.zeros(num_states)

    initial_conditions = np.zeros(num_states)
    offset = 10.0 * np.random.random((num_states / 2) - 1)
    initial_conditions[1:num_states / 2] = np.deg2rad(offset)

    rhs, args = simulate.closed_loop_ode_func(system, time, set_point,
                                              gains, lateral_force)

    x = odeint(rhs, initial_conditions, time, args=(args,))

    # Numerically differentiate the states with x2' = (x2 - x1) / h
    xdot = np.vstack((np.zeros(num_states),
                      np.diff(x, axis=0) / discretization_interval))

    # Create a symbolic function for the continious constraints.
    control_dict, gain_syms, equil_syms = \
        model.create_symbolic_controller(states_syms,
                                         specified_inputs_syms[:-1])
    eq_dict = dict(zip(equil_syms, num_states * [0]))
    closed = model.symbolic_constraints_solved(mass_matrix, forcing_vector,
                                               states_syms, control_dict,
                                               eq_dict)

    # Evaluate the constraint equation for each of the time steps.
    # This loop is really slow, could speed it up with lambdify or
    # something.
    closed_eval = np.zeros_like(x)

    for i in range(len(time) - 1):
        print('Eval {}'.format(i))
        current_x = x[i + 1, :]
        current_xdot = xdot[i + 1, :]

        val_map = dict(zip(model.state_derivatives(states_syms), current_xdot))
        val_map.update(dict(zip(states_syms, current_x)))
        val_map.update(simulate.constants_dict(constants_syms))
        val_map.update(dict(zip(gain_syms, gains.flatten())))
        val_map[specified_inputs_syms[-1]] = lateral_force[i + 1]
        evald_closed = closed.subs(val_map).evalf()
        closed_eval[i] = np.array(evald_closed).squeeze().astype(float)

    fig = plt.figure()
    plt.plot(closed_eval)
    plt.legend([str(s) for s in states_syms])
    fig.savefig('constraint_violations_{}hz.png'.format(sample_rate))


def test_sim_discrete_equate():
    """This ensures that the rhs function evaluates the same as the symbolic
    closed loop form."""

    num_links = 1

    system = model.n_link_pendulum_on_cart(num_links, cart_force=True,
                                           joint_torques=True,
                                           spring_damper=True)

    mass_matrix = system[0]
    forcing_vector = system[1]
    constants_syms = system[2]
    coordinates_syms = system[3]
    speeds_syms = system[4]
    specified_inputs_syms = system[5]  # last entry is lateral force
    states_syms = coordinates_syms + speeds_syms
    state_derivs_syms = model.state_derivatives(states_syms)

    gains = simulate.compute_controller_gains(num_links)

    equilibrium_point = np.zeros(len(states_syms))

    lateral_force = np.random.random(1)

    def specified(x, t):
        joint_torques = np.dot(gains, equilibrium_point - x)
        return np.hstack((joint_torques, lateral_force))

    rhs = generate_ode_function(*system)

    args = {'constants': simulate.constants_dict(constants_syms).values(),
            'specified': specified}

    state_values = np.random.random(len(states_syms))

    state_deriv_values = rhs(state_values, 0.0, args)

    control_dict, gain_syms, equil_syms = \
        model.create_symbolic_controller(states_syms,
                                         specified_inputs_syms[:-1])

    eq_dict = dict(zip(equil_syms, len(states_syms) * [0]))

    closed = model.symbolic_constraints(mass_matrix, forcing_vector,
                                        states_syms, control_dict, eq_dict)

    xdot_expr = sym.solve(closed, state_derivs_syms)
    xdot_expr = sym.Matrix([xdot_expr[xd] for xd in state_derivs_syms])

    val_map = dict(zip(states_syms, state_values))
    val_map.update(simulate.constants_dict(constants_syms))
    val_map.update(dict(zip(gain_syms, gains.flatten())))
    val_map[specified_inputs_syms[-1]] = lateral_force

    sym_sol = np.array([x for x in xdot_expr.subs(val_map).evalf()], dtype=float)

    np.testing.assert_allclose(state_deriv_values, sym_sol)

    # Now let's see if the discretized version gives a simliar answer if h
    # is small enough.
    collocator = dc.ConstraintCollocator(closed, states_syms, 10, 0.01)
    dclosed = collocator.discrete_eom
    xi = collocator.current_discrete_state_symbols
    xp = collocator.previous_discrete_state_symbols
    si = collocator.current_discrete_specified_symbols
    h = collocator.time_interval_symbol

    euler_formula = [(i - p) / h for i, p in zip(xi, xp)]

    xdot_expr = sym.solve(dclosed, euler_formula)
    xdot_expr = sym.Matrix([xdot_expr[xd] for xd in euler_formula])

    val_map = dict(zip(xi, state_values))
    val_map.update(simulate.constants_dict(constants_syms))
    val_map.update(dict(zip(gain_syms, gains.flatten())))
    val_map[si[0]] = lateral_force

    sym_sol = np.array([x for x in xdot_expr.subs(val_map).evalf()], dtype=float)

    np.testing.assert_allclose(state_deriv_values, sym_sol)

    # now how do i check that (xi - xp) / h equals sym_sol?
    # If I integrate the continous eoms and get a state trajectory, x, then
    # compute x' by backward euler given some h


def test_output_equations():

    # four states (cols), and 5 time steps (rows)
    x = np.array([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [9.0, 10.0, 11.0, 12.0],
                  [13.0, 14.0, 15.0, 16.0],
                  [17.0, 18.0, 19.0, 20.0]])

    y = simulate.output_equations(x)

    expected_y = np.array([[1.0, 2.0],
                           [5.0, 6.0],
                           [9.0, 10.0],
                           [13.0, 14.0],
                           [17.0, 18.0]])

    np.testing.assert_allclose(y, expected_y)
