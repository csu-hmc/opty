#!/usr/bin/env python

import pytest
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


class TestCreateObjectiveFunction(object):
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.t = sym.symbols('t')
        self.x, self.v = sym.symbols('x, v', cls=sym.Function)
        self.m, self.c, self.k = sym.symbols('m, c, k')
        self.f1, self.f2 = sym.symbols('f1:3', cls=sym.Function)

        self.x = self.x(self.t)
        self.v = self.v(self.t)
        self.f1, self.f2 = self.f1(self.t), self.f2(self.t)

        self.state_symbols = [self.x, self.v]
        self.input_symbols = [self.f2, self.f1]  # Should be sorted
        self.unknown_symbols = [self.m, self.c, self.k]  # Should be sorted
        self.n = len(self.state_symbols)
        self.q = len(self.input_symbols)
        self.r = len(self.unknown_symbols)
        self.N = 20

        self.x_vals = np.random.random(self.N)
        self.v_vals = np.random.random(self.N)
        self.f1_vals = np.random.random(self.N)
        self.f2_vals = np.random.random(self.N)
        self.m_val, self.c_val, self.k_val = np.random.random(3)
        self.free = np.hstack((self.x_vals, self.v_vals, self.f1_vals,
                               self.f2_vals, self.c_val, self.k_val,
                               self.m_val))

    def test_backward_single_state(self):
        obj_expr = sym.Integral(self.x ** 2, self.t)
        obj, obj_grad = utils.create_objective_function(
            obj_expr, self.state_symbols, self.input_symbols,
            self.unknown_symbols, self.N, 0.5)
        np.testing.assert_allclose(obj(self.free),
                                   0.5 * (self.x_vals[1:] ** 2).sum())
        np.testing.assert_allclose(obj_grad(self.free), np.hstack((
            0, 0.5 * 2 * self.x_vals[1:],
            np.zeros(self.N * (1 + self.q) + self.r))))

    def test_backward_single_input(self):
        obj_expr = sym.Integral(self.f1 ** 2, self.t)
        obj, obj_grad = utils.create_objective_function(
            obj_expr, self.state_symbols, self.input_symbols,
            self.unknown_symbols, self.N, 1)
        np.testing.assert_allclose(obj(self.free), (self.f1_vals[1:] ** 2).sum())
        np.testing.assert_allclose(obj_grad(self.free), np.hstack((
            np.zeros(self.N * self.n + 1), 2 * self.f1_vals[1:],
            np.zeros(self.N * 1 + self.r))))

    def test_backward_single_unknown(self):
        obj_expr = self.m ** 2
        obj, obj_grad = utils.create_objective_function(
            obj_expr, self.state_symbols, self.input_symbols,
            self.unknown_symbols, self.N, 0.3)
        np.testing.assert_allclose(obj(self.free), self.m_val ** 2)
        np.testing.assert_allclose(obj_grad(self.free), np.hstack((
            np.zeros(self.N * (self.n + self.q) + 2), 2 * self.m_val)))

    def test_backward_all(self):
        obj_expr = (
            sym.Integral(self.x ** 2 + self.m ** 2, self.t) +
            sym.Integral(self.c ** 2 * self.f2 ** 2 , self.t) +
            sym.sin(self.k) ** 2
        )
        obj, obj_grad = utils.create_objective_function(
            obj_expr, self.state_symbols, self.input_symbols,
            self.unknown_symbols, self.N, 0.3)
        np.testing.assert_allclose(
            obj(self.free),
            0.3 * ((self.x_vals[1:] ** 2).sum() +
                   (self.N - 1) * self.m_val ** 2 +
                   (self.c_val ** 2 * self.f2_vals[1:] ** 2).sum()) +
            np.sin(self.k_val) ** 2
        )
        np.testing.assert_allclose(obj_grad(self.free), np.hstack((
            0, 0.3 * 2 * self.x_vals[1:], np.zeros(self.N * 2 + 1),
            0.3 * 2 * self.c_val ** 2 * self.f2_vals[1:],
            0.3 * 2 * self.c_val * (self.f2_vals[1:] ** 2).sum(),
            2 * np.sin(self.k_val) * np.cos(self.k_val),
            0.3 * (self.N - 1) * 2 * self.m_val
            )))

    def test_no_states(self):
        free = self.free[self.n * self.N:]
        obj_expr = sym.Integral(self.f1 ** 2, self.t)
        obj, obj_grad = utils.create_objective_function(
            obj_expr, [], self.input_symbols, self.unknown_symbols, self.N, 1)
        np.testing.assert_allclose(obj(free), (self.f1_vals[1:] ** 2).sum())
        np.testing.assert_allclose(obj_grad(free), np.hstack((
            0, 2 * self.f1_vals[1:], np.zeros(self.N + self.r)))
        )

    def test_no_inputs(self):
        free = np.hstack((self.free[:self.n * self.N], self.free[-self.r:]))
        obj_expr = sym.Integral(self.x ** 2, self.t)
        obj, obj_grad = utils.create_objective_function(
            obj_expr, self.state_symbols, [], self.unknown_symbols, self.N, 1)
        np.testing.assert_allclose(obj(free), (self.x_vals[1:] ** 2).sum())
        np.testing.assert_allclose(obj_grad(free), np.hstack((
            0, 2 * self.x_vals[1:], np.zeros(self.N + self.r)))
        )

    def test_no_unknowns(self):
        free = self.free[:-self.r]
        obj_expr = sym.Integral(self.x ** 2, self.t)
        obj, obj_grad = utils.create_objective_function(
            obj_expr, self.state_symbols, self.input_symbols, [], self.N, 1)
        np.testing.assert_allclose(obj(free), (self.x_vals[1:] ** 2).sum())
        np.testing.assert_allclose(obj_grad(free), np.hstack((
            0, 2 * self.x_vals[1:], np.zeros(self.N * (self.n - 1 + self.q)))
        ))

    def test_midpoint_all(self):
        obj_expr = (
            sym.Integral(self.x ** 2 + self.m ** 2, self.t) +
            sym.Integral(self.c ** 2 * self.f2 ** 2 , self.t) +
            sym.sin(self.k) ** 2
        )
        x_mid = (self.x_vals[1:] + self.x_vals[:-1]) / 2
        f2_mid = (self.f2_vals[1:] + self.f2_vals[:-1]) / 2
        obj, obj_grad = utils.create_objective_function(
            obj_expr, self.state_symbols, self.input_symbols,
            self.unknown_symbols, self.N, 0.3, integration_method='midpoint')
        np.testing.assert_allclose(
            obj(self.free),
            0.3 * ((x_mid ** 2).sum() + (self.N - 1) * self.m_val ** 2 +
                   (self.c_val ** 2 * f2_mid ** 2).sum()) +
            np.sin(self.k_val) ** 2
        )
        np.testing.assert_allclose(obj_grad(self.free), np.hstack((
            0.3 * self.x_vals[0], 0.3 * 2 * self.x_vals[1:-1],
            0.3 * self.x_vals[-1], np.zeros(self.N * 2),
            0.3 * self.c_val ** 2 * self.f2_vals[0],
            0.3 * 2 * self.c_val ** 2 * self.f2_vals[1:-1],
            0.3 * self.c_val ** 2 * self.f2_vals[-1],
            0.3 * 2 * self.c_val * (f2_mid ** 2).sum(),
            2 * np.sin(self.k_val) * np.cos(self.k_val),
            0.3 * (self.N - 1) * 2 * self.m_val
            )))

    def test_not_existing_method(self):
        with pytest.raises(NotImplementedError):
            utils.create_objective_function(
                sym.Integral(self.x ** 2, self.t), self.state_symbols,
                self.input_symbols, self.unknown_symbols, self.N, 1,
                integration_method='not_existing_method')

    def test_invalid_integration_limits(self):
        with pytest.raises(NotImplementedError):
            obj, obj_grad = utils.create_objective_function(
                sym.Integral(self.x ** 2, (self.t, 0, 1)), self.state_symbols,
                self.input_symbols, self.unknown_symbols, self.N, 1)


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

    # NOTE: `if` is a reserved C keyword, this should not cause a compile
    # error.
    a, b, c, d = sym.symbols('a, b, if, d_{badsym}')

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

    # NOTE : Will not compile due to d_{badsym} being an invalid C variable
    # name.
    with pytest.raises(ImportError) as error:
        utils.ufuncify_matrix((a, b, d), sym_mat.xreplace({c: d}))

    assert error.match("double d_{badsym}")


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
