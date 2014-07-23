#!/usr/bin/env python

import numpy as np
from scipy import sparse
import sympy as sym
from sympy import symbols, Function, Matrix, simplify

import pendulum


def test_sim_discrete_equate():
    """This ensures that the rhs function evaluates the same as the symbolic
    closed loop form."""

    num_links = 1

    system = pendulum.n_link_pendulum_on_cart(num_links,
                                              cart_force=True,
                                              joint_torques=True)

    mass_matrix = system[0]
    forcing_vector = system[1]
    constants_syms = system[2]
    coordinates_syms = system[3]
    speeds_syms = system[4]
    specified_inputs_syms = system[5]  # last entry is lateral force
    states_syms = coordinates_syms + speeds_syms
    state_derivs_syms = pendulum.state_derivatives(states_syms)

    gains = pendulum.compute_controller_gains(num_links)

    equilibrium_point = np.hstack((0.0,
                                   np.pi / 2.0 * np.ones(len(coordinates_syms) - 1),
                                   np.zeros(len(speeds_syms))))

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

    eq_values = [0] + [sym.pi / 2] * (len(coordinates_syms) - 1) + [0] * len(speeds_syms)

    eq_dict = dict(zip(equil_syms, eq_values))

    closed = pendulum.symbolic_closed_loop(mass_matrix,
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
    dclosed = pendulum.discretize(closed, states_syms, specified_inputs_syms)

    xi, xp, si, h = pendulum.discrete_symbols(states_syms,
                                              specified_inputs_syms)

    euler_formula = [(i - p) / h for i, p in zip(xi, xp)]

    xdot_expr = sym.solve(dclosed, euler_formula)
    xdot_expr = sym.Matrix([xdot_expr[xd] for xd in euler_formula])

    val_map = dict(zip(xi, state_values))
    #val_map[h] = 0.001
    #state_values_previous = state_values - val_map[h]
    #val_map.update(dict(zip(xp, state_values_previous)))
    val_map.update(pendulum.constants_dict(constants_syms))
    val_map.update(dict(zip(gain_syms, gains.flatten())))
    val_map[si[1]] = lateral_force

    sym_sol = np.array([x for x in xdot_expr.subs(val_map).evalf()], dtype=float)

    np.testing.assert_allclose(state_deriv_values, sym_sol)

    # now how do i check that (xi - xp) / h equals sym_sol?


def test_state_derivatives():

    t = symbols('t')
    x, v = symbols('x, v', cls=Function)

    x = x(t)
    v = v(t)

    derivs = pendulum.state_derivatives([x, v])

    assert derivs == [x.diff(t), v.diff(t)]


def test_parse_free():

    q = 2  # two free model constants
    n = 4  # four states
    r = 2  # two free specified inputs
    N = 10  # ten time steps

    expected_constants = np.random.random(q)
    expected_state_traj = np.random.random((n, N))
    expected_input_traj = np.random.random((r, N))

    free = np.hstack((expected_state_traj.flatten(),
                      expected_input_traj.flatten(),
                      expected_constants))

    state_traj, input_traj, constants = pendulum.parse_free(free, n, r, N)

    np.testing.assert_allclose(expected_constants, constants)
    np.testing.assert_allclose(expected_state_traj, state_traj)
    np.testing.assert_allclose(expected_input_traj, input_traj)


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


def test_symbolic_closed_loop():

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

    closed = pendulum.symbolic_closed_loop(mass_matrix,
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

    closed = pendulum.symbolic_closed_loop(mass_matrix,
                                           forcing_vector,
                                           states,
                                           control_dict,
                                           eq_dict)

    assert simplify(expected_closed - closed) == Matrix([0, 0, 0, 0])


def test_output_equations():

    # four states (cols), and 5 time steps (rows)
    x = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16],
                  [17, 18, 19, 20]])

    y = pendulum.output_equations(x)

    expected_y = np.array([[1, 2],
                           [5, 6],
                           [9, 10],
                           [13, 14],
                           [17, 18]])

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
    grad = pendulum.objective_function_gradient(free, M, n, h, time, y_measured)

    expected_grad = np.zeros_like(free)
    delta = 1e-8
    for i in range(len(free)):
        free_copy = free.copy()
        free_copy[i] = free_copy[i] + delta
        perturbed = pendulum.objective_function(free_copy, M, n, h, time, y_measured)
        expected_grad[i] = (perturbed - cost) / delta

    np.testing.assert_allclose(grad, expected_grad, atol=1e-8)


def test_substitute_matrix():

    A = np.arange(1, 13, dtype=float).reshape(3, 4)
    sub = np.array([[21, 22], [23, 24]])
    new_A = pendulum.substitute_matrix(A, [1, 2], [0, 2], sub)
    expected = np.array([[1, 2, 3, 4],
                         [21, 6, 22, 8],
                         [23, 10, 24, 12]], dtype=float)

    np.testing.assert_allclose(new_A, expected)

    A = sparse.lil_matrix(np.zeros((3, 4)))
    sub = np.array([[21, 22], [23, 24]])
    new_A = pendulum.substitute_matrix(A, [1, 2], [0, 2], sub)
    expected = np.array([[0, 0, 0, 0],
                         [21, 0, 22, 0],
                         [23, 0, 24, 0]], dtype=float)

    np.testing.assert_allclose(new_A.todense(), expected)


def test_discrete_symbols():
    t = symbols('t')
    v, x = symbols('v, x', cls=Function)
    f = symbols('f', cls=Function)

    v = v(t)
    x = x(t)
    f = f(t)

    states = [v, x]
    specified = [f]

    current_states, previous_states, current_specified, interval = \
        pendulum.discrete_symbols(states, specified)

    xi, vi, xp, vp, fi, h = symbols('xi, vi, xp, vp, fi, h')

    assert current_states[0] is vi
    assert current_states[1] is xi
    assert previous_states[0] is vp
    assert previous_states[1] is xp
    assert current_specified[0] is fi
    assert interval is h


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


def test_discretize():

    t, h = symbols('t, h')
    x, v = symbols('x, v', cls=Function)
    m, c, k = symbols('m, c, k')
    f = symbols('f', cls=Function)

    x = x(t)
    v = v(t)
    f = f(t)

    states = [x, v]
    specified = [f]

    eoms = Matrix([x.diff() - v,
                   m * v.diff() + c * v + k * x - f])

    discrete_eoms = pendulum.discretize(eoms, states, specified, h)

    xi, vi, xp, vp, fi = symbols('xi, vi, xp, vp, fi')

    expected = Matrix([(xi - xp) / h - vi,
                       m * (vi - vp) / h + c * vi + k * xi - fi])

    assert simplify(discrete_eoms - expected) == Matrix([0, 0])


def test_general_constraint():

    t, h = symbols('t, h')
    x, v = symbols('x, v', cls=Function)
    m, c, k = symbols('m, c, k')
    f = symbols('f', cls=Function)

    x = x(t)
    v = v(t)
    f = f(t)

    states = [x, v]
    specified = [f]

    xi, vi, xp, vp, fi = symbols('xi, vi, xp, vp, fi')

    eom_vector = Matrix([(xi - xp) / h - vi,
                         m * (vi - vp) / h + c * vi + k * xi - fi])

    constrain = pendulum.general_constraint(eom_vector, states, specified,
                                            [m, c, k])

    state_values = np.array([[1, 2, 3, 4],
                             [5, 6, 7, 8]])
    specified_values = np.array([2, 2, 2, 2])
    constant_values = np.array([1.0, 2.0, 3.0])
    m, c, k = constant_values
    h = 0.01

    result = constrain(state_values, specified_values, constant_values, h)

    expected_dynamic = np.zeros(3)
    expected_kinematic = np.zeros(3)

    for i in [1, 2, 3]:

        xi, vi = state_values[:, i]
        xp, vp = state_values[:, i - 1]
        fi = specified_values[i]

        expected_dynamic[i - 1] = m * (vi - vp) / h + c * vi + k * xi - fi
        expected_kinematic[i - 1] = (xi - xp) / h - vi

    expected = np.hstack((expected_kinematic, expected_dynamic))

    np.testing.assert_allclose(result, expected)


def test_general_constraint_jacobian():

    t, h = symbols('t, h')
    x, v = symbols('x, v', cls=Function)
    m, c, k = symbols('m, c, k')
    f = symbols('f', cls=Function)

    x = x(t)
    v = v(t)
    f = f(t)

    states = [x, v]
    specified = [f]
    constants = [m, c, k]
    free_constants = [k]

    xi, vi, xp, vp, fi = symbols('xi, vi, xp, vp, fi')

    eom_vector = Matrix([(xi - xp) / h - vi,
                         m * (vi - vp) / h + c * vi + k * xi - fi])

    jacobian = pendulum.general_constraint_jacobian(eom_vector, states,
                                                    specified, constants,
                                                    free_constants)

    state_values = np.array([[1, 2, 3, 4],   # x
                             [5, 6, 7, 8]])  # v
    specified_values = np.array([2, 2, 2, 2])
    constant_values = np.array([1.0, 2.0, 3.0])

    x = state_values[0]
    m, c, k = constant_values
    h = 0.01

    result = jacobian(state_values, specified_values, constant_values, h)

    # jacobian of eom_vector wrt vi, xi, xp, vp, k
    #    [     vi,  xi,   vp,   xp,  k]
    # x: [     -1, 1/h,    0, -1/h,  0]
    # v: [c + m/h,   k, -m/h,    0, xi]

    expected_jacobian = np.array(
        #     x1,     x2,     x3,    x4,     v1,        v2,         v3,        v4,    k
        [[-1 / h,  1 / h,      0,     0,      0,        -1,          0,         0,    0],
         [     0, -1 / h,  1 / h,     0,      0,         0,         -1,         0,    0],
         [     0,      0, -1 / h, 1 / h,      0,         0,          0,        -1,    0],
         [     0,      k,      0,     0, -m / h, c + m / h,          0,         0, x[1]],
         [     0,      0,      k,     0,      0,    -m / h,  c + m / h,         0, x[2]],
         [     0,      0,      0,     k,      0,         0,      -m /h, c + m / h, x[3]]])

    np.testing.assert_allclose(result.todense(), expected_jacobian)


def test_wrap_constraint():

    t, h = symbols('t, h')
    x, v = symbols('x, v', cls=Function)
    m, c, k = symbols('m, c, k')
    f = symbols('f', cls=Function)

    x = x(t)
    v = v(t)
    f = f(t)

    states = [x, v]
    specified = [f]
    constants = [m, c, k]
    free_constants = [k]

    xi, vi, xp, vp, fi = symbols('xi, vi, xp, vp, fi')

    eom_vector = Matrix([(xi - xp) / h - vi,
                         m * (vi - vp) / h + c * vi + k * xi - fi])

    gen_con = pendulum.general_constraint(eom_vector, states, specified,
                                          constants)

    gen_jac = pendulum.general_constraint_jacobian(eom_vector, states,
                                                   specified, constants,
                                                   free_constants)

    num_time_steps = 4
    num_states = 2
    interval_value = 0.01
    fixed_constants = {m: 1.0, c: 2.0}
    fixed_specified = {fi: np.array([2, 2, 2, 2])}

    specified_syms = [fi]

    constrain = pendulum.wrap_constraint(gen_con, num_time_steps,
                                         num_states, interval_value,
                                         constants, specified_syms,
                                         fixed_constants, fixed_specified)

    free = np.array([1, 2, 3, 4, 5, 6, 7, 8, 3.0])

    result = constrain(free)

    expected_dynamic = np.zeros(3)
    expected_kinematic = np.zeros(3)

    state_values = np.array([[1, 2, 3, 4],
                             [5, 6, 7, 8]])
    specified_values = np.array([2, 2, 2, 2])
    constant_values = np.array([1.0, 2.0, 3.0])
    m, c, k = constant_values
    h = interval_value

    for i in [1, 2, 3]:

        xi, vi = state_values[:, i]
        xp, vp = state_values[:, i - 1]
        fi = specified_values[i]

        expected_kinematic[i - 1] = (xi - xp) / h - vi
        expected_dynamic[i - 1] = m * (vi - vp) / h + c * vi + k * xi - fi

    expected = np.hstack((expected_kinematic, expected_dynamic))

    np.testing.assert_allclose(result, expected)

    jacobian = pendulum.wrap_constraint(gen_jac, num_time_steps, num_states,
                                        interval_value, constants,
                                        specified_syms, fixed_constants,
                                        fixed_specified)

    result = jacobian(free)

    x = state_values[0]

    expected_jacobian = np.array(
        #     x1,     x2,     x3,    x4,     v1,        v2,         v3,        v4,    k
        [[-1 / h,  1 / h,      0,     0,      0,        -1,          0,         0,    0],
         [     0, -1 / h,  1 / h,     0,      0,         0,         -1,         0,    0],
         [     0,      0, -1 / h, 1 / h,      0,         0,          0,        -1,    0],
         [     0,      k,      0,     0, -m / h, c + m / h,          0,         0, x[1]],
         [     0,      0,      k,     0,      0,    -m / h,  c + m / h,         0, x[2]],
         [     0,      0,      0,     k,      0,         0,      -m /h, c + m / h, x[3]]])

    np.testing.assert_allclose(result.todense(), expected_jacobian)
