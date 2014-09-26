#!/usr/bin/env python

from collections import OrderedDict

import numpy as np
import sympy as sym
from scipy import sparse

from direct_collocation import (ConstraintCollocator, objective_function,
                                objective_function_gradient)


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

    cost = objective_function(free, M, n, h, time, y_measured)

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

    cost = objective_function(free, M, n, h, time, y_measured)
    grad = objective_function_gradient(free, M, n, h, time, y_measured)

    expected_grad = np.zeros_like(free)
    delta = 1e-8
    for i in range(len(free)):
        free_copy = free.copy()
        free_copy[i] = free_copy[i] + delta
        perturbed = objective_function(free_copy, M, n, h, time, y_measured)
        expected_grad[i] = (perturbed - cost) / delta

    np.testing.assert_allclose(grad, expected_grad, atol=1e-8)


class TestConstraintCollocator():

    def setup(self):

        m, c, k, t = sym.symbols('m, c, k, t')
        x, v, f = [s(t) for s in sym.symbols('x, v, f', cls=sym.Function)]

        self.state_symbols = (x, v)
        self.constant_symbols = (m, c, k)
        self.specified_symbols = (f,)
        self.discrete_symbols = sym.symbols('xi, vi, xp, vp, fi')

        self.state_values = np.array([[1.0, 2.0, 3.0, 4.0],
                                      [5.0, 6.0, 7.0, 8.0]])
        self.specified_values = np.array([2.0, 2.0, 2.0, 2.0])
        self.constant_values = np.array([1.0, 2.0, 3.0])
        self.interval_value = 0.01
        self.free = np.array([1.0, 2.0, 3.0, 4.0,  # x
                              5.0, 6.0, 7.0, 8.0,  # v
                              3.0])  # k

        self.eom = sym.Matrix([x.diff() - v,
                               m * v.diff() + c * v + k * x - f])

        par_map = OrderedDict(zip(self.constant_symbols[:2],
                                  self.constant_values[:2]))
        traj_map = OrderedDict(zip([f], [self.specified_values]))

        self.collocator = \
            ConstraintCollocator(equations_of_motion=self.eom,
                                 state_symbols=self.state_symbols,
                                 num_collocation_nodes=4,
                                 node_time_interval=self.interval_value,
                                 known_parameter_map=par_map,
                                 known_trajectory_map=traj_map)

    def test_init(self):

        assert self.collocator.state_symbols == self.state_symbols
        assert self.collocator.state_derivative_symbols == \
            tuple([s.diff() for s in self.state_symbols])
        assert self.collocator.num_states == 2

    def test_sort_parameters(self):

        # TODO : Added checks for the other cases.

        self.collocator._sort_parameters()

        m, c, k = self.constant_symbols

        assert self.collocator.known_parameters == (m, c)
        assert self.collocator.num_known_parameters == 2

        assert self.collocator.unknown_parameters == (k,)
        assert self.collocator.num_unknown_parameters == 1

    def test_sort_trajectories(self):

        # TODO : Added checks for the other cases.

        self.collocator._sort_trajectories()

        assert self.collocator.known_input_trajectories == self.specified_symbols
        assert self.collocator.num_known_input_trajectories == 1

        assert self.collocator.unknown_input_trajectories == tuple()
        assert self.collocator.num_unknown_input_trajectories == 0

    def test_discrete_symbols(self):

        self.collocator._discrete_symbols()

        xi, vi, xp, vp, fi = self.discrete_symbols

        assert self.collocator.current_discrete_state_symbols == (xi, vi)
        assert self.collocator.previous_discrete_state_symbols == (xp, vp)
        assert self.collocator.current_discrete_specified_symbols[0] is fi

    def test_discretize_eom(self):

        m, c, k = self.constant_symbols
        xi, vi, xp, vp, fi = self.discrete_symbols
        h = self.collocator.time_interval_symbol

        expected = sym.Matrix([(xi - xp) / h - vi,
                               m * (vi - vp) / h + c * vi + k * xi - fi])

        self.collocator._discretize_eom()

        zero = sym.simplify(self.collocator.discrete_eom - expected)

        assert zero == sym.Matrix([0, 0])

    def test_gen_multi_arg_con_func(self):

        self.collocator._gen_multi_arg_con_func()

        # Make sure the parameters are in the correct order.
        constant_values = \
            np.array([self.constant_values[self.constant_symbols.index(c)]
                      for c in self.collocator.parameters])

        # TODO : Once there are more than one specified, they will need to
        # be put in the correct order too.

        result = self.collocator._multi_arg_con_func(self.state_values,
                                                     self.specified_values,
                                                     constant_values,
                                                     self.interval_value)

        m, c, k = self.constant_values
        h = self.interval_value

        expected_dynamic = np.zeros(3)
        expected_kinematic = np.zeros(3)

        for i in [1, 2, 3]:

            xi, vi = self.state_values[:, i]
            xp, vp = self.state_values[:, i - 1]
            fi = self.specified_values[i]

            expected_dynamic[i - 1] = m * (vi - vp) / h + c * vi + k * xi - fi
            expected_kinematic[i - 1] = (xi - xp) / h - vi

        expected = np.hstack((expected_kinematic, expected_dynamic))

        np.testing.assert_allclose(result, expected)

    def test_gen_multi_arg_con_jac_func(self):

        self.collocator._gen_multi_arg_con_jac_func()

        # Make sure the parameters are in the correct order.
        constant_values = \
            np.array([self.constant_values[self.constant_symbols.index(c)]
                      for c in self.collocator.parameters])

        # TODO : Once there are more than one specified, they will need to
        # be put in the correct order too.

        jac_vals = self.collocator._multi_arg_con_jac_func(self.state_values,
                                                           self.specified_values,
                                                           constant_values,
                                                           self.interval_value)

        row_idxs, col_idxs = self.collocator.jacobian_indices()

        jacobian_matrix = sparse.coo_matrix((jac_vals, (row_idxs, col_idxs)))

        # jacobian of eom_vector wrt vi, xi, xp, vp, k
        #    [     vi,  xi,   vp,   xp,  k]
        # x: [     -1, 1/h,    0, -1/h,  0]
        # v: [c + m/h,   k, -m/h,    0, xi]

        x = self.state_values[0]
        m, c, k = self.constant_values
        h = self.interval_value

        expected_jacobian = np.array(
            #     x1,     x2,     x3,    x4,     v1,        v2,         v3,        v4,    k
            [[-1 / h,  1 / h,      0,     0,      0,        -1,          0,         0,    0],
             [     0, -1 / h,  1 / h,     0,      0,         0,         -1,         0,    0],
             [     0,      0, -1 / h, 1 / h,      0,         0,          0,        -1,    0],
             [     0,      k,      0,     0, -m / h, c + m / h,          0,         0, x[1]],
             [     0,      0,      k,     0,      0,    -m / h,  c + m / h,         0, x[2]],
             [     0,      0,      0,     k,      0,         0,      -m /h, c + m / h, x[3]]],
            dtype=float)

        np.testing.assert_allclose(jacobian_matrix.todense(), expected_jacobian)

    def test_generate_constraint_function(self):

        constrain = self.collocator.generate_constraint_function()

        result = constrain(self.free)

        expected_dynamic = np.zeros(3)
        expected_kinematic = np.zeros(3)

        m, c, k = self.constant_values
        h = self.interval_value

        for i in [1, 2, 3]:

            xi, vi = self.state_values[:, i]
            xp, vp = self.state_values[:, i - 1]
            fi = self.specified_values[i]

            expected_kinematic[i - 1] = (xi - xp) / h - vi
            expected_dynamic[i - 1] = m * (vi - vp) / h + c * vi + k * xi - fi

        expected = np.hstack((expected_kinematic, expected_dynamic))

        np.testing.assert_allclose(result, expected)

    def test_generate_jacobian_function(self):

        jacobian = self.collocator.generate_jacobian_function()

        jac_vals = jacobian(self.free)

        row_idxs, col_idxs = self.collocator.jacobian_indices()

        jacobian_matrix = sparse.coo_matrix((jac_vals, (row_idxs, col_idxs)))

        m, c, k = self.constant_values
        h = self.interval_value
        x = self.state_values[0]

        expected_jacobian = np.array(
            #     x1,     x2,     x3,    x4,     v1,        v2,         v3,        v4,    k
            [[-1 / h,  1 / h,      0,     0,      0,        -1,          0,         0,    0],
            [     0, -1 / h,  1 / h,     0,      0,         0,         -1,         0,    0],
            [     0,      0, -1 / h, 1 / h,      0,         0,          0,        -1,    0],
            [     0,      k,      0,     0, -m / h, c + m / h,          0,         0, x[1]],
            [     0,      0,      k,     0,      0,    -m / h,  c + m / h,         0, x[2]],
            [     0,      0,      0,     k,      0,         0,      -m /h, c + m / h, x[3]]],
            dtype=float)

        np.testing.assert_allclose(jacobian_matrix.todense(), expected_jacobian)


class TestConstraintCollocatorUnknownTrajectories():

    def setup(self):

        # constant parameters
        m, c, t = sym.symbols('m, c, t')
        # time varying parameters
        x, v, f, k = [s(t) for s in sym.symbols('x, v, f, k',
                                                cls=sym.Function)]

        self.state_symbols = (x, v)
        self.constant_symbols = (m, c)
        self.specified_symbols = (f, k)
        self.discrete_symbols = sym.symbols('xi, vi, xp, vp, fi, ki')

        # The state order should match the eom order and state symbol order.
        self.state_values = np.array([[1.0, 2.0, 3.0, 4.0],  # x
                                      [5.0, 6.0, 7.0, 8.0]])  # v
        # This must be ordered as the known first then the unknown.
        self.specified_values = np.array([[9.0, 10.0, 11.0, 12.0],  # f
                                          [13.0, 14.0, 15.0, 16.0]])  # k
        # This must be ordered as the known first then the unknown.
        self.constant_values = np.array([1.0,  # m
                                         2.0])  # c
        self.interval_value = 0.01
        # The free optimization variable array.
        self.free = np.array([1.0, 2.0, 3.0, 4.0,  # x
                              5.0, 6.0, 7.0, 8.0,  # v
                              13.0, 14.0, 15.0, 16.0,  # k
                              2.0])  # c

        self.eom = sym.Matrix([x.diff() - v,
                               m * v.diff() + c * v + k * x - f])

        par_map = OrderedDict(zip([m], [1.0]))
        traj_map = OrderedDict(zip([f], [self.specified_values[0]]))

        self.collocator = \
            ConstraintCollocator(equations_of_motion=self.eom,
                                 state_symbols=self.state_symbols,
                                 num_collocation_nodes=4,
                                 node_time_interval=self.interval_value,
                                 known_parameter_map=par_map,
                                 known_trajectory_map=traj_map)

    def test_init(self):

        assert self.collocator.state_symbols == self.state_symbols
        assert self.collocator.state_derivative_symbols == \
            tuple([s.diff() for s in self.state_symbols])
        assert self.collocator.num_states == 2

    def test_sort_parameters(self):

        # TODO : Added checks for the other cases.

        self.collocator._sort_parameters()

        m, c = self.constant_symbols

        assert self.collocator.known_parameters == (m,)
        assert self.collocator.num_known_parameters == 1

        assert self.collocator.unknown_parameters == (c,)
        assert self.collocator.num_unknown_parameters == 1

    def test_sort_trajectories(self):

        # TODO : Added checks for the other cases.

        self.collocator._sort_trajectories()

        f, k = self.specified_symbols

        assert self.collocator.known_input_trajectories == (f,)
        assert self.collocator.num_known_input_trajectories == 1

        assert self.collocator.unknown_input_trajectories == (k,)
        assert self.collocator.num_unknown_input_trajectories == 1

    def test_discrete_symbols(self):

        self.collocator._discrete_symbols()

        xi, vi, xp, vp, fi, ki = self.discrete_symbols

        assert self.collocator.current_discrete_state_symbols == (xi, vi)
        assert self.collocator.previous_discrete_state_symbols == (xp, vp)
        # The following should be in order of known and unknown.
        assert self.collocator.current_discrete_specified_symbols == (fi, ki)

    def test_discretize_eom(self):

        m, c = self.constant_symbols
        xi, vi, xp, vp, fi, ki = self.discrete_symbols
        h = self.collocator.time_interval_symbol

        expected = sym.Matrix([(xi - xp) / h - vi,
                               m * (vi - vp) / h + c * vi + ki * xi - fi])

        self.collocator._discretize_eom()

        zero = sym.simplify(self.collocator.discrete_eom - expected)

        assert zero == sym.Matrix([0, 0])

    def test_gen_multi_arg_con_func(self):

        self.collocator._gen_multi_arg_con_func()

        # Make sure the parameters are in the correct order.
        constant_values = \
            np.array([self.constant_values[self.constant_symbols.index(c)]
                      for c in self.collocator.parameters])

        # TODO : Once there are more than one specified, they will need to
        # be put in the correct order too.

        result = self.collocator._multi_arg_con_func(self.state_values,
                                                     self.specified_values,
                                                     constant_values,
                                                     self.interval_value)

        m, c = self.constant_values
        h = self.interval_value

        expected_dynamic = np.zeros(3)
        expected_kinematic = np.zeros(3)

        for i in [1, 2, 3]:

            xi, vi = self.state_values[:, i]
            xp, vp = self.state_values[:, i - 1]
            fi, ki = self.specified_values[:, i]

            expected_dynamic[i - 1] = m * (vi - vp) / h + c * vi + ki * xi - fi
            expected_kinematic[i - 1] = (xi - xp) / h - vi

        expected = np.hstack((expected_kinematic, expected_dynamic))

        np.testing.assert_allclose(result, expected)

    def test_jacobian_indices(self):

        row_idxs, col_idxs = self.collocator.jacobian_indices()

        expected_row_idxs = np.array([0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 1,
                                      1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 2, 2,
                                      2, 2, 2, 2, 5, 5, 5, 5, 5, 5])

        expected_col_idxs = np.array([1, 5, 0, 4, 9, 12, 1, 5, 0, 4, 9, 12,
                                      2, 6, 1, 5, 10, 12, 2, 6, 1, 5, 10,
                                      12, 3, 7, 2, 6, 11, 12, 3, 7, 2, 6,
                                      11, 12])

        np.testing.assert_allclose(row_idxs, expected_row_idxs)
        np.testing.assert_allclose(col_idxs, expected_col_idxs)

    def test_gen_multi_arg_con_jac_func(self):

        self.collocator._gen_multi_arg_con_jac_func()

        # Make sure the parameters are in the correct order.
        constant_values = \
            np.array([self.constant_values[self.constant_symbols.index(c)]
                      for c in self.collocator.parameters])

        # TODO : Once there are more than one specified, they will need to
        # be put in the correct order too.

        jac_vals = self.collocator._multi_arg_con_jac_func(self.state_values,
                                                           self.specified_values,
                                                           constant_values,
                                                           self.interval_value)

        row_idxs, col_idxs = self.collocator.jacobian_indices()

        jacobian_matrix = sparse.coo_matrix((jac_vals, (row_idxs, col_idxs)))

        # jacobian of eom_vector wrt vi, xi, xp, vp, k
        #    [     vi,  xi,   vp,   xp,  k]
        # x: [     -1, 1/h,    0, -1/h,  0]
        # v: [c + m/h,   k, -m/h,    0, xi]

        x, v = self.state_values
        m, c = self.constant_values
        f, k = self.specified_values
        h = self.interval_value

        expected_jacobian = np.array(
            #     x0,     x1,     x2,    x3,     v0,        v1,         v2,        v3,   k0,   k1,   k2,   k3,    c    con
            [[-1 / h,  1 / h,      0,     0,      0,        -1,          0,         0,    0,    0,    0,    0,    0],  # 1
             [     0, -1 / h,  1 / h,     0,      0,         0,         -1,         0,    0,    0,    0,    0,    0],  # 2
             [     0,      0, -1 / h, 1 / h,      0,         0,          0,        -1,    0,    0,    0,    0,    0],  # 3
             [     0,   k[1],      0,     0, -m / h, c + m / h,          0,         0,    0, x[1],    0,    0, v[1]],  # 1
             [     0,      0,   k[2],     0,      0,    -m / h,  c + m / h,         0,    0,    0, x[2],    0, v[2]],  # 2
             [     0,      0,      0,  k[3],      0,         0,      -m /h, c + m / h,    0,    0,    0, x[3], v[3]]], # 3
            dtype=float)

        np.testing.assert_allclose(jacobian_matrix.todense(), expected_jacobian)

    def test_generate_constraint_function(self):

        constrain = self.collocator.generate_constraint_function()

        result = constrain(self.free)

        m, c = self.constant_values
        h = self.interval_value

        expected_dynamic = np.zeros(3)
        expected_kinematic = np.zeros(3)

        for i in [1, 2, 3]:

            xi, vi = self.state_values[:, i]
            xp, vp = self.state_values[:, i - 1]
            fi, ki = self.specified_values[:, i]

            expected_dynamic[i - 1] = m * (vi - vp) / h + c * vi + ki * xi - fi
            expected_kinematic[i - 1] = (xi - xp) / h - vi

        expected = np.hstack((expected_kinematic, expected_dynamic))

        np.testing.assert_allclose(result, expected)

    def test_generate_jacobian_function(self):

        jacobian = self.collocator.generate_jacobian_function()

        jac_vals = jacobian(self.free)

        row_idxs, col_idxs = self.collocator.jacobian_indices()

        jacobian_matrix = sparse.coo_matrix((jac_vals, (row_idxs, col_idxs)))

        x, v = self.state_values
        m, c = self.constant_values
        f, k = self.specified_values
        h = self.interval_value

        expected_jacobian = np.array(
            #     x0,     x1,     x2,    x3,     v0,        v1,         v2,        v3,   k0,   k1,   k2,   k3,    c    con
            [[-1 / h,  1 / h,      0,     0,      0,        -1,          0,         0,    0,    0,    0,    0,    0],  # 1
             [     0, -1 / h,  1 / h,     0,      0,         0,         -1,         0,    0,    0,    0,    0,    0],  # 2
             [     0,      0, -1 / h, 1 / h,      0,         0,          0,        -1,    0,    0,    0,    0,    0],  # 3
             [     0,   k[1],      0,     0, -m / h, c + m / h,          0,         0,    0, x[1],    0,    0, v[1]],  # 1
             [     0,      0,   k[2],     0,      0,    -m / h,  c + m / h,         0,    0,    0, x[2],    0, v[2]],  # 2
             [     0,      0,      0,  k[3],      0,         0,      -m /h, c + m / h,    0,    0,    0, x[3], v[3]]], # 3
            dtype=float)

        np.testing.assert_allclose(jacobian_matrix.todense(), expected_jacobian)


def test_merge_fixed_free_parameters():

    m, c, k = sym.symbols('m, c, k')
    all_syms = m, c, k
    known = {m: 1.0, c: 2.0}
    unknown = np.array([3.0])

    merged = ConstraintCollocator._merge_fixed_free(all_syms, known,
                                                    unknown, 'par')

    expected = np.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(merged, expected)

    a, b, c, d = sym.symbols('a, b, c, d')
    all_syms = a, b, c, d
    known = {a: 1.0, b: 2.0}
    unknown = np.array([3.0, 4.0])

    merged = ConstraintCollocator._merge_fixed_free(all_syms, known,
                                                    unknown, 'par')

    expected = np.array([1.0, 2.0, 3.0, 4.0])

    np.testing.assert_allclose(merged, expected)


def test_merge_fixed_free_trajectories():

    t = sym.symbols('t')
    f, k = sym.symbols('f, k', cls=sym.Function)
    f = f(t)
    k = k(t)
    all_syms = f, k
    known = {f: np.array([1.0, 2.0])}
    unknown = np.array([3.0, 4.0])

    merged = ConstraintCollocator._merge_fixed_free(all_syms, known,
                                                    unknown, 'traj')

    expected = np.array([[1.0, 2.0],
                         [3.0, 4.0]])

    np.testing.assert_allclose(merged, expected)

    t = sym.symbols('t')
    all_syms = [f(t) for f in sym.symbols('a, b, c, d', cls=sym.Function)]
    a, b, c, d = all_syms
    known = {a: np.array([1.0, 2.0]),
             b: np.array([3.0, 4.0])}
    unknown = np.array([[5.0, 6.0],
                        [7.0, 8.0]])

    merged = ConstraintCollocator._merge_fixed_free(all_syms, known,
                                                    unknown, 'traj')

    expected = np.array([[1.0, 2.0],
                         [3.0, 4.0],
                         [5.0, 6.0],
                         [7.0, 8.0]])

    np.testing.assert_allclose(merged, expected)
