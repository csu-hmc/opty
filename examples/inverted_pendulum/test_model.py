#!/usr/bin/env python

import sympy as sym

import model


def test_state_derivatives():

    t = sym.symbols('t')
    x, v = sym.symbols('x, v', cls=sym.Function)

    x = x(t)
    v = v(t)

    derivs = model.state_derivatives([x, v])

    assert derivs == [x.diff(t), v.diff(t)]


def test_f_minus_ma():

    t = sym.symbols('t')
    x, v = sym.symbols('x, v', cls=sym.Function)
    m, c, k = sym.symbols('m, c, k')
    f = sym.symbols('f', cls=sym.Function)

    x = x(t)
    v = v(t)
    f = f(t)

    states = [x, v]

    mass_matrix = sym.Matrix([[1, 0], [0, m]])
    forcing_vector = sym.Matrix([v, -c * v - k * x + f])

    constraint = model.f_minus_ma(mass_matrix, forcing_vector, states)

    expected = sym.Matrix([x.diff() - v,
                           m * v.diff() + c * v + k * x - f])

    assert sym.simplify(constraint - expected) == sym.Matrix([0, 0])


def test_symbolic_constraints():

    states = sym.symbols('q0, q1, u0, u1', cls=sym.Function)
    inputs = sym.symbols('T0, T1', cls=sym.Function)
    t = sym.symbols('t')

    states = [s(t) for s in states]
    inputs = [i(t) for i in inputs]

    q0, q1, u0, u1 = states
    T0, T1 = inputs

    k00, k01, k02, k03 = sym.symbols('k_0(:4)')
    k10, k11, k12, k13 = sym.symbols('k_1(:4)')

    eq = sym.symbols('q0_eq, q1_eq, u0_eq, u1_eq')
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

    m20, m21, m22, m23 = sym.symbols('m_2(:4)')
    m30, m31, m32, m33 = sym.symbols('m_3(:4)')
    f2, f3 = sym.symbols('f2, f3')

    mass_matrix = sym.Matrix([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, m22, m23],
                              [0, 0, m32, m33]])

    forcing_vector = sym.Matrix([u0,
                                 u1,
                                 f2 + T0,
                                 f3 + T1])

    expected_closed = \
        sym.Matrix([q0.diff(t) - u0,
                    q1.diff(t) - u1,
                    m22 * u0.diff(t) + m23 * u1.diff(t) - f2 - control_dict[T0],
                    m32 * u0.diff(t) + m33 * u1.diff(t) - f3 - control_dict[T1]])

    closed = model.symbolic_constraints(mass_matrix,
                                        forcing_vector,
                                        states,
                                        control_dict)

    assert sym.simplify(expected_closed - closed) == sym.Matrix([0, 0, 0, 0])

    eq_dict = {k: 0 for k in eq}

    eq_control_dict = \
        {T0: -k00 * q0 - k01 * q1 +
             -k02 * u0 - k03 * u1,
         T1: -k10 * q0 - k11 * q1 +
             -k12 * u0 - k13 * u1}

    expected_closed = \
       sym.Matrix([q0.diff(t) - u0,
                   q1.diff(t) - u1,
                   m22 * u0.diff(t) + m23 * u1.diff(t) - f2 - eq_control_dict[T0],
                   m32 * u0.diff(t) + m33 * u1.diff(t) - f3 - eq_control_dict[T1]])

    closed = model.symbolic_constraints(mass_matrix,
                                        forcing_vector,
                                        states,
                                        control_dict,
                                        eq_dict)

    assert sym.simplify(expected_closed - closed) == sym.Matrix([0, 0, 0, 0])


def test_create_symbolic_controller():

    states = sym.symbols('q1, q2, u1, u2', cls=sym.Function)
    inputs = sym.symbols('T1, T2', cls=sym.Function)
    t = sym.symbols('t')

    states = [s(t) for s in states]
    inputs = [i(t) for i in inputs]

    q1, q2, u1, u2 = states
    T1, T2 = inputs

    k00, k01, k02, k03 = sym.symbols('k_0(:4)')
    k10, k11, k12, k13 = sym.symbols('k_1(:4)')

    eq = sym.symbols('q1_eq, q2_eq, u1_eq, u2_eq')
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

    expected_xeq = sym.Matrix([q1_eq, q2_eq, u1_eq, u2_eq])

    controller_dict, gain_syms, xeq = model.create_symbolic_controller(states, inputs)

    for k, v in controller_dict.items():
        assert sym.simplify(v - expected_controller_dict[k]) == 0
    assert gain_syms == expected_gain_syms
    assert xeq == expected_xeq
