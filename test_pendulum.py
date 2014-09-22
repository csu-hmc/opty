#!/usr/bin/env python

import numpy as np
from scipy import sparse
import sympy as sym
from sympy import symbols, Function, Matrix, simplify
import matplotlib.pyplot as plt

import pendulum
from direct_collocation import ConstraintCollocator


def test_num_diff(sample_rate=100.0):

    num_links = 1

    # Generate the symbolic equations of motion for the two link pendulum on
    # a cart.
    system = pendulum.n_link_pendulum_on_cart(num_links, cart_force=True,
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

    gains = pendulum.compute_controller_gains(num_links)

    # Specify the number of time steps and duration of the measurements.
    num_time_steps = 51
    duration = (num_time_steps - 1) / sample_rate
    discretization_interval = 1.0 / sample_rate

    msg = 'Integrating over {} seconds with {} time steps spaced at {} seconds apart.'
    print(msg.format(duration, num_time_steps, discretization_interval))

    # Integrate the equations of motion.
    time = np.linspace(0.0, duration, num=num_time_steps)

    lateral_force = pendulum.input_force('sumsines', time)

    set_point = np.zeros(num_states)

    initial_conditions = np.zeros(num_states)
    offset = 10.0 * np.random.random((num_states / 2) - 1)
    initial_conditions[1:num_states / 2] = np.deg2rad(offset)

    rhs, args = pendulum.closed_loop_ode_func(system, time, set_point,
                                              gains, lateral_force)

    x = pendulum.odeint(rhs, initial_conditions, time, args=(args,))

    # Numerically differentiate the states with x2' = (x2 - x1) / h
    xdot = np.vstack((np.zeros(num_states),
                      np.diff(x, axis=0) / discretization_interval))

    # Create a symbolic function for the continious constraints.
    control_dict, gain_syms, equil_syms = \
        pendulum.create_symbolic_controller(states_syms,
                                            specified_inputs_syms[:-1])
    eq_dict = dict(zip(equil_syms, num_states * [0]))
    closed = pendulum.symbolic_constraints_solved(mass_matrix, forcing_vector,
                                            states_syms, control_dict, eq_dict)

    # Evaluate the constraint equation for each of the time steps.
    # This loop is really slow, could speed it up with lambdify or
    # something.
    closed_eval = np.zeros_like(x)

    for i in range(len(time) - 1):
        print('Eval {}'.format(i))
        current_x = x[i + 1, :]
        current_xdot = xdot[i + 1, :]

        val_map = dict(zip(pendulum.state_derivatives(states_syms), current_xdot))
        val_map.update(dict(zip(states_syms, current_x)))
        val_map.update(pendulum.constants_dict(constants_syms))
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

    system = pendulum.n_link_pendulum_on_cart(num_links,
                                              cart_force=True,
                                              joint_torques=True,
                                              spring_damper=True)

    mass_matrix = system[0]
    forcing_vector = system[1]
    constants_syms = system[2]
    coordinates_syms = system[3]
    speeds_syms = system[4]
    specified_inputs_syms = system[5]  # last entry is lateral force
    states_syms = coordinates_syms + speeds_syms
    state_derivs_syms = pendulum.state_derivatives(states_syms)

    gains = pendulum.compute_controller_gains(num_links)

    equilibrium_point = np.zeros(len(states_syms))

    lateral_force = np.random.random(1)

    def specified(x, t):
        joint_torques = np.dot(gains, equilibrium_point - x)
        return np.hstack((joint_torques, lateral_force))

    rhs = pendulum.generate_ode_function(*system)

    args = {'constants': pendulum.constants_dict(constants_syms).values(),
            'specified': specified}

    state_values = np.random.random(len(states_syms))

    state_deriv_values = rhs(state_values, 0.0, args)

    control_dict, gain_syms, equil_syms = \
        pendulum.create_symbolic_controller(states_syms,
                                            specified_inputs_syms[:-1])

    eq_dict = dict(zip(equil_syms, len(states_syms) * [0]))

    closed = pendulum.symbolic_constraints(mass_matrix,
                                           forcing_vector,
                                           states_syms,
                                           control_dict,
                                           eq_dict)

    xdot_expr = sym.solve(closed, state_derivs_syms)
    xdot_expr = sym.Matrix([xdot_expr[xd] for xd in state_derivs_syms])

    val_map = dict(zip(states_syms, state_values))
    val_map.update(pendulum.constants_dict(constants_syms))
    val_map.update(dict(zip(gain_syms, gains.flatten())))
    val_map[specified_inputs_syms[-1]] = lateral_force

    sym_sol = np.array([x for x in xdot_expr.subs(val_map).evalf()], dtype=float)

    np.testing.assert_allclose(state_deriv_values, sym_sol)

    # Now let's see if the discretized version gives a simliar answer if h
    # is small enough.
    collocator = ConstraintCollocator(closed, states_syms, 10, 0.01)
    dclosed = collocator.discrete_eom
    xi = collocator.current_discrete_state_symbols
    xp = collocator.previous_discrete_state_symbols
    si = collocator.current_discrete_specified_symbols
    h = collocator.time_interval_symbol

    euler_formula = [(i - p) / h for i, p in zip(xi, xp)]

    xdot_expr = sym.solve(dclosed, euler_formula)
    xdot_expr = sym.Matrix([xdot_expr[xd] for xd in euler_formula])

    val_map = dict(zip(xi, state_values))
    val_map.update(pendulum.constants_dict(constants_syms))
    val_map.update(dict(zip(gain_syms, gains.flatten())))
    val_map[si[0]] = lateral_force

    sym_sol = np.array([x for x in xdot_expr.subs(val_map).evalf()], dtype=float)

    np.testing.assert_allclose(state_deriv_values, sym_sol)

    # now how do i check that (xi - xp) / h equals sym_sol?
    # If I integrate the continous eoms and get a state trajectory, x, then
    # compute x' by backward euler given some h


def test_state_derivatives():

    t = symbols('t')
    x, v = symbols('x, v', cls=Function)

    x = x(t)
    v = v(t)

    derivs = pendulum.state_derivatives([x, v])

    assert derivs == [x.diff(t), v.diff(t)]


def test_create_symbolic_controller():

    states = symbols('q1, q2, u1, u2', cls=Function)
    inputs = symbols('T1, T2', cls=Function)
    t = symbols('t')

    states = [s(t) for s in states]
    inputs = [i(t) for i in inputs]

    q1, q2, u1, u2 = states
    T1, T2 = inputs

    k00, k01, k02, k03 = symbols('k_0(:4)')
    k10, k11, k12, k13 = symbols('k_1(:4)')

    eq = symbols('q1_eq, q2_eq, u1_eq, u2_eq')
    q1_eq, q2_eq, u1_eq, u2_eq = eq

    # T = K * (x_eq - x)
    # [T1] = [k00 k01 k02 k03] * [q1_eq - q1]
    # [T2]   [k10 k11 k12 k13]   [q2_eq - q2]
    #                            [u1_eq - u1]
    #                            [u2_eq - u2]

    expected_controller_dict = \
        {T1: k00 * (q1_eq - q1) + k01 * (q2_eq - q2) +
             k02 * (u1_eq - u1) + k03 * (u2_eq - u2),
         T2: k10 * (q1_eq - q1) + k11 * (q2_eq - q2) +
             k12 * (u1_eq - u1) + k13 * (u2_eq - u2)}

    expected_gain_syms = [k00, k01, k02, k03, k10, k11, k12, k13]

    expected_xeq = Matrix([q1_eq, q2_eq, u1_eq, u2_eq])

    controller_dict, gain_syms, xeq = pendulum.create_symbolic_controller(states, inputs)

    for k, v in controller_dict.items():
        assert simplify(v - expected_controller_dict[k]) == 0
    assert gain_syms == expected_gain_syms
    assert xeq == expected_xeq


def test_symbolic_constraints():

    states = symbols('q0, q1, u0, u1', cls=Function)
    inputs = symbols('T0, T1', cls=Function)
    t = symbols('t')

    states = [s(t) for s in states]
    inputs = [i(t) for i in inputs]

    q0, q1, u0, u1 = states
    T0, T1 = inputs

    k00, k01, k02, k03 = symbols('k_0(:4)')
    k10, k11, k12, k13 = symbols('k_1(:4)')

    eq = symbols('q0_eq, q1_eq, u0_eq, u1_eq')
    q0_eq, q1_eq, u0_eq, u1_eq = eq

    # T = K * (x_eq - x)
    # [T1] = [k00 k01 k02 k03] * [q1_eq - q1]
    # [T2]   [k10 k11 k12 k13]   [q2_eq - q2]
    #                            [u1_eq - u1]
    #                            [u2_eq - u2]

    control_dict = \
        {T0: k00 * (q0_eq - q0) + k01 * (q1_eq - q1) +
             k02 * (u0_eq - u0) + k03 * (u1_eq - u1),
         T1: k10 * (q0_eq - q0) + k11 * (q1_eq - q1) +
             k12 * (u0_eq - u0) + k13 * (u1_eq - u1)}

    m20, m21, m22, m23 = symbols('m_2(:4)')
    m30, m31, m32, m33 = symbols('m_3(:4)')
    f2, f3 = symbols('f2, f3')

    mass_matrix = Matrix([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, m22, m23],
                          [0, 0, m32, m33]])

    forcing_vector = Matrix([u0,
                             u1,
                             f2 + T0,
                             f3 + T1])

    expected_closed = \
        Matrix([q0.diff(t) - u0,
                q1.diff(t) - u1,
                m22 * u0.diff(t) + m23 * u1.diff(t) - f2 - control_dict[T0],
                m32 * u0.diff(t) + m33 * u1.diff(t) - f3 - control_dict[T1]])

    closed = pendulum.symbolic_constraints(mass_matrix,
                                           forcing_vector,
                                           states,
                                           control_dict)

    assert simplify(expected_closed - closed) == Matrix([0, 0, 0, 0])

    eq_dict = {k: 0 for k in eq}

    eq_control_dict = \
        {T0: -k00 * q0 - k01 * q1 +
             -k02 * u0 - k03 * u1,
         T1: -k10 * q0 - k11 * q1 +
             -k12 * u0 - k13 * u1}

    expected_closed = \
        Matrix([q0.diff(t) - u0,
                q1.diff(t) - u1,
                m22 * u0.diff(t) + m23 * u1.diff(t) - f2 - eq_control_dict[T0],
                m32 * u0.diff(t) + m33 * u1.diff(t) - f3 - eq_control_dict[T1]])

    closed = pendulum.symbolic_constraints(mass_matrix,
                                           forcing_vector,
                                           states,
                                           control_dict,
                                           eq_dict)

    assert simplify(expected_closed - closed) == Matrix([0, 0, 0, 0])


def test_output_equations():

    # four states (cols), and 5 time steps (rows)
    x = np.array([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [9.0, 10.0, 11.0, 12.0],
                  [13.0, 14.0, 15.0, 16.0],
                  [17.0, 18.0, 19.0, 20.0]])

    y = pendulum.output_equations(x)

    expected_y = np.array([[1.0, 2.0],
                           [5.0, 6.0],
                           [9.0, 10.0],
                           [13.0, 14.0],
                           [17.0, 18.0]])

    np.testing.assert_allclose(y, expected_y)


def test_objective_function():

    M = 5
    o = 2
    n = 2 * o
    q = 3
    h = 0.01

    time = np.linspace(0.0, (M - 1) * h, num=M)

    y_measured = np.random.random((M, o))  # measured coordinates

    x_model = np.hstack((y_measured, np.random.random((M, o))))

    free = np.hstack((x_model.T.flatten(), np.random.random(q)))

    cost = pendulum.objective_function(free, M, n, h, time, y_measured)

    np.testing.assert_allclose(cost, 0.0, atol=1e-15)


def test_objective_function_gradient():

    M = 5
    o = 2
    n = 2 * o
    q = 3
    h = 0.01

    time = np.linspace(0.0, (M - 1) * h, num=M)
    y_measured = np.random.random((M, o))  # measured coordinates
    x_model = np.random.random((M, n))
    free = np.hstack((x_model.T.flatten(), np.random.random(q)))

    cost = pendulum.objective_function(free, M, n, h, time, y_measured)
    grad = pendulum.objective_function_gradient(free, M, n, h, time,
                                                y_measured)

    expected_grad = np.zeros_like(free)
    delta = 1e-8
    for i in range(len(free)):
        free_copy = free.copy()
        free_copy[i] = free_copy[i] + delta
        perturbed = pendulum.objective_function(free_copy, M, n, h, time,
                                                y_measured)
        expected_grad[i] = (perturbed - cost) / delta

    np.testing.assert_allclose(grad, expected_grad, atol=1e-8)


def test_f_minus_ma():

    t = symbols('t')
    x, v = symbols('x, v', cls=Function)
    m, c, k = symbols('m, c, k')
    f = symbols('f', cls=Function)

    x = x(t)
    v = v(t)
    f = f(t)

    states = [x, v]

    mass_matrix = Matrix([[1, 0], [0, m]])
    forcing_vector = Matrix([v, -c * v - k * x + f])

    constraint = pendulum.f_minus_ma(mass_matrix, forcing_vector, states)

    expected = Matrix([x.diff() - v,
                       m * v.diff() + c * v + k * x - f])

    assert simplify(constraint - expected) == Matrix([0, 0])
