#!/usr/bin/env python

import numpy as np
from numpy import testing
import sympy as sym
from scipy import sparse

from .. import utils


def test_state_derivatives():

    t = sym.symbols('t')
    x, v = sym.symbols('x, v', cls=sym.Function)

    x = x(t)
    v = v(t)

    derivs = utils.state_derivatives([x, v])

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

    constraint = utils.f_minus_ma(mass_matrix, forcing_vector, states)

    expected = sym.Matrix([x.diff() - v,
                           m * v.diff() + c * v + k * x - f])

    assert sym.simplify(constraint - expected) == sym.Matrix([0, 0])


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

    state_traj, input_traj, constants = utils.parse_free(free, n, r, N)

    np.testing.assert_allclose(expected_constants, constants)
    np.testing.assert_allclose(expected_state_traj, state_traj)
    np.testing.assert_allclose(expected_input_traj, input_traj)


def test_ufuncify_matrix():

    a, b, c = sym.symbols('a, b, if')

    expr_00 = a**2 * sym.cos(b)**c
    expr_01 = sym.tan(b) / sym.sin(a + b) + c**4
    expr_10 = a**2 + b**2 - sym.sqrt(c)
    expr_11 = ((a + b + c) * (a + b)) / a * sym.sin(b)

    sym_mat = sym.Matrix([[expr_00, expr_01],
                          [expr_10, expr_11]])

    # These simply set up some large one dimensional arrays that will be
    # used in the expression evaluation.

    n = 10000

    a_vals = np.random.random(n)
    b_vals = np.random.random(n)
    c_vals = np.random.random(n)
    c_val = np.random.random(1)[0]

    def eval_matrix_loop_numpy(a_vals, b_vals, c_vals):
        """Since the number of matrix elements are typically much smaller
        than the number of evaluations, NumPy can be used to compute each of
        the Matrix expressions. This is equivalent to the individual
        lambdified expressions above."""

        result = np.empty((n, 2, 2))

        result[:, 0, 0] = a_vals**2 * np.cos(b_vals)**c_vals
        result[:, 0, 1] = np.tan(b_vals) / np.sin(a_vals + b_vals) + c_vals**4
        result[:, 1, 0] = a_vals**2 + b_vals**2 - np.sqrt(c_vals)
        result[:, 1, 1] = (((a_vals + b_vals + c_vals) * (a_vals + b_vals)) /
                           a_vals * np.sin(b_vals))

        return result

    f = utils.ufuncify_matrix((a, b, c), sym_mat)

    result = np.empty((n, 4))

    testing.assert_allclose(f(result, a_vals, b_vals, c_vals),
                            eval_matrix_loop_numpy(a_vals, b_vals, c_vals))

    f = utils.ufuncify_matrix((a, b, c), sym_mat, const=(c,))

    result = np.empty((n, 4))

    testing.assert_allclose(f(result, a_vals, b_vals, c_val),
                            eval_matrix_loop_numpy(a_vals, b_vals, c_val))

    f = utils.ufuncify_matrix((a, b, c), sym_mat, const=(c,), parallel=True)

    result = np.empty((n, 4))

    testing.assert_allclose(f(result, a_vals, b_vals, c_val),
                            eval_matrix_loop_numpy(a_vals, b_vals, c_val))


def test_substitute_matrix():

    A = np.arange(1, 13, dtype=float).reshape(3, 4)
    sub = np.array([[21, 22], [23, 24]])
    new_A = utils.substitute_matrix(A, [1, 2], [0, 2], sub)
    expected = np.array([[1, 2, 3, 4],
                         [21, 6, 22, 8],
                         [23, 10, 24, 12]], dtype=float)

    np.testing.assert_allclose(new_A, expected)

    A = sparse.lil_matrix(np.zeros((3, 4)))
    sub = np.array([[21, 22], [23, 24]])
    new_A = utils.substitute_matrix(A, [1, 2], [0, 2], sub)
    expected = np.array([[0, 0, 0, 0],
                         [21, 0, 22, 0],
                         [23, 0, 24, 0]], dtype=float)

    np.testing.assert_allclose(new_A.todense(), expected)
