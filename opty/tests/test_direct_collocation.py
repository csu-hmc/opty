#!/usr/bin/env python

from collections import OrderedDict

import numpy as np
import sympy as sym
from scipy import sparse
from nose.tools import raises

from ..direct_collocation import Problem, ConstraintCollocator


def test_Problem():

    m, c, k, t = sym.symbols('m, c, k, t')
    x, v, f = [s(t) for s in sym.symbols('x, v, f', cls=sym.Function)]

    state_symbols = (x, v)

    interval_value = 0.01

    eom = sym.Matrix([x.diff() - v,
                      m * v.diff() + c * v + k * x - f])

    prob = Problem(lambda x: 1.0,
                   lambda x: x,
                   eom,
                   state_symbols,
                   2,
                   interval_value,
                   bounds={x: (-10.0, 10.0),
                           f: (-8.0, 8.0),
                           m: (-1.0, 1.0),
                           c: (-0.5, 0.5)})

    INF = 10e19
    expected_lower = np.array([-10.0, -10.0,
                               -INF, -INF,
                               -8.0, -8.0,
                               -0.5, -INF, -1.0])
    np.testing.assert_allclose(prob.lower_bound, expected_lower)
    expected_upper = np.array([10.0, 10.0,
                               INF, INF,
                               8.0, 8.0,
                               0.5, INF, 1.0])
    np.testing.assert_allclose(prob.upper_bound, expected_upper)


class TestConstraintCollocator():

    def setup(self):

        m, c, k, t = sym.symbols('m, c, k, t')
        x, v, f = [s(t) for s in sym.symbols('x, v, f', cls=sym.Function)]

        self.state_symbols = (x, v)
        self.constant_symbols = (m, c, k)
        self.specified_symbols = (f,)
        self.discrete_symbols = sym.symbols('xi, vi, xp, vp, xn, vn, fi, fn',
                                            real=True)

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
        assert self.collocator.num_collocation_nodes == 4

    @raises(ValueError)
    def test_integration_method(self):
        self.collocator.integration_method = 'booger'

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

        assert self.collocator.known_input_trajectories == \
            self.specified_symbols
        assert self.collocator.num_known_input_trajectories == 1

        assert self.collocator.unknown_input_trajectories == tuple()
        assert self.collocator.num_unknown_input_trajectories == 0

    def test_discrete_symbols(self):

        self.collocator._discrete_symbols()

        xi, vi, xp, vp, xn, vn, fi, fn = self.discrete_symbols

        assert self.collocator.previous_discrete_state_symbols == (xp, vp)
        assert self.collocator.current_discrete_state_symbols == (xi, vi)
        assert self.collocator.next_discrete_state_symbols == (xn, vn)

        assert self.collocator.current_discrete_specified_symbols == (fi, )
        assert self.collocator.next_discrete_specified_symbols == (fn, )

    def test_discretize_eom_backward_euler(self):

        m, c, k = self.constant_symbols
        xi, vi, xp, vp, xn, vn, fi, fn = self.discrete_symbols
        h = self.collocator.time_interval_symbol

        expected = sym.Matrix([(xi - xp) / h - vi,
                               m * (vi - vp) / h + c * vi + k * xi - fi])

        self.collocator._discretize_eom()

        zero = sym.simplify(self.collocator.discrete_eom - expected)

        assert zero == sym.Matrix([0, 0])

    def test_discretize_eom_midpoint(self):

        m, c, k = self.constant_symbols
        xi, vi, xp, vp, xn, vn, fi, fn = self.discrete_symbols

        h = self.collocator.time_interval_symbol

        expected = sym.Matrix([(xn - xi) / h - (vi + vn) / 2,
                               m * (vn - vi) / h + c * (vi + vn) / 2 +
                               k * (xi + xn) / 2 - (fi + fn) / 2])

        self.collocator.integration_method = 'midpoint'
        self.collocator._discretize_eom()

        zero = sym.simplify(self.collocator.discrete_eom - expected)

        assert zero == sym.Matrix([0, 0])

    def test_gen_multi_arg_con_func_backward_euler(self):

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

    def test_gen_multi_arg_con_func_midpoint(self):

        self.collocator.integration_method = 'midpoint'
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

        for i in [0, 1, 2]:

            xi, vi = self.state_values[:, i]
            xn, vn = self.state_values[:, i + 1]
            fi = self.specified_values[i:i + 1]
            fn = self.specified_values[i + 1:i + 2]

            expected_kinematic[i] = (xn - xi) / h - (vi + vn) / 2
            expected_dynamic[i] = (m * (vn - vi) / h + c * (vn + vi) / 2 + k
                                   * (xn + xi) / 2 - (fi + fn) / 2)

        expected = np.hstack((expected_kinematic, expected_dynamic))

        np.testing.assert_allclose(result, expected)

    def test_gen_multi_arg_con_jac_func_backward_euler(self):

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

    def test_gen_multi_arg_con_jac_func_midpoint(self):

        self.collocator.integration_method = 'midpoint'
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

        x = self.state_values[0]
        m, c, k = self.constant_values
        h = self.interval_value

        part1 = np.array(
            #        x0,       x1,       x2,      x3
            [[ -1.0 / h,  1.0 / h,        0,       0],
             [        0, -1.0 / h,  1.0 / h,       0],
             [        0,        0, -1.0 / h, 1.0 / h],
             [  k / 2.0,  k / 2.0,        0,       0],
             [        0,  k / 2.0,  k / 2.0,       0],
             [        0,        0,  k / 2.0, k / 2.0]],
            dtype=float)

        part2 = np.array(
            #                v0,               v1,               v2,              v3,                   k      i
            [[       -1.0 / 2.0,       -1.0 / 2.0,                0,               0,                   0],  # 0
             [                0,       -1.0 / 2.0,       -1.0 / 2.0,               0,                   0],  # 1
             [                0,                0,       -1.0 / 2.0,      -1.0 / 2.0,                   0],  # 2
             [ -m / h + c / 2.0,  m / h + c / 2.0,                0,               0, (x[1] + x[0]) / 2.0],  # 0
             [                0, -m / h + c / 2.0,  m / h + c / 2.0,               0, (x[2] + x[1]) / 2.0],  # 1
             [                0,                0, -m / h + c / 2.0, m / h + c / 2.0, (x[3] + x[2]) / 2.0]], # 2
            dtype=float)

        expected_jacobian = np.hstack((part1, part2))

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
            [     0,  -1 / h,  1 / h,     0,      0,         0,         -1,         0,    0],
            [     0,       0, -1 / h, 1 / h,      0,         0,          0,        -1,    0],
            [     0,       k,      0,     0, -m / h, c + m / h,          0,         0, x[1]],
            [     0,       0,      k,     0,      0,    -m / h,  c + m / h,         0, x[2]],
            [     0,       0,      0,     k,      0,         0,      -m /h, c + m / h, x[3]]],
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
        self.discrete_symbols = sym.symbols('xi, vi, xp, vp, fi, ki',
                                            real=True)

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

    def test_gen_multi_arg_con_jac_func_midpoint(self):

        self.collocator.integration_method = 'midpoint'
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

        x, v = self.state_values
        m, c = self.constant_values
        f, k = self.specified_values
        h = self.interval_value

        part1 = np.array(
            #                   x0,                  x1,                  x2,                  x3
            [[            -1.0 / h,             1.0 / h,                   0,                   0],
             [                   0,            -1.0 / h,             1.0 / h,                   0],
             [                   0,                   0,            -1.0 / h,             1.0 / h],
             [ (k[0] + k[1]) / 4.0, (k[0] + k[1]) / 4.0,                   0,                   0],
             [                   0, (k[1] + k[2]) / 4.0, (k[1] + k[2]) / 4.0,                   0],
             [                   0,                   0, (k[2] + k[3]) / 4.0, (k[2] + k[3]) / 4.0]],
            dtype=float)

        part2 = np.array(
            #              v0,             v1,             v2,            v3      i
            [[     -1.0 / 2.0,     -1.0 / 2.0,              0,             0],  # 0
             [              0,     -1.0 / 2.0,     -1.0 / 2.0,             0],  # 1
             [              0,              0,     -1.0 / 2.0,    -1.0 / 2.0],  # 2
             [ -m / h + c / 2,  m / h + c / 2,              0,             0],  # 0
             [              0, -m / h + c / 2,  m / h + c / 2,             0],  # 1
             [              0,              0, -m / h + c / 2, m / h + c / 2]], # 2
            dtype=float)

        part3 = np.array(
            #                   k0,                   k1,                 k2,                  k3,                 c      i
            [[                   0,                   0,                   0,                   0,                 0],  # 0
             [                   0,                   0,                   0,                   0,                 0],  # 1
             [                   0,                   0,                   0,                   0,                 0],  # 2
             [ (x[1] + x[0]) / 4.0, (x[1] + x[0]) / 4.0,                   0,                   0, (v[1] + v[0]) / 2],  # 0
             [                   0, (x[2] + x[1]) / 4.0, (x[2] + x[1]) / 4.0,                   0, (v[2] + v[1]) / 2],  # 1
             [                   0,                   0, (x[3] + x[2]) / 4.0, (x[3] + x[2]) / 4.0, (v[3] + v[2]) / 2]], # 2
            dtype=float)

        expected_jacobian = np.hstack((part1, part2, part3))

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
    all_syms = [g(t) for g in sym.symbols('a, b, c, d', cls=sym.Function)]
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


class TestConstraintCollocatorInstanceConstraints():

    def setup(self):

        I, m, g, d, t = sym.symbols('I, m, g, d, t')
        theta, omega, T = [f(t) for f in sym.symbols('theta, omega, T',
                                                     cls=sym.Function)]

        self.state_symbols = (theta, omega)
        self.constant_symbols = (I, m, g, d)
        self.specified_symbols = (T,)
        self.discrete_symbols = \
            sym.symbols('thetai, omegai, thetap, omegap, Ti', real=True)

        self.state_values = np.array([[1.0, 2.0, 3.0, 4.0],
                                      [5.0, 6.0, 7.0, 8.0]])
        self.specified_values = np.array([9.0, 10.0, 11.0, 12.0])
        self.constant_values = np.array([1.0, 1.0, 9.81, 1.0])
        self.interval_value = 0.01
        self.time = np.array([0.0, 0.01, 0.02, 0.03])
        self.free = np.array([1.0, 2.0, 3.0, 4.0,  # theta
                              5.0, 6.0, 7.0, 8.0,  # omega
                              9.0, 10.0, 11.0, 12.0,  # T
                              1.0,  # I
                              1.0,  # m
                              9.81,  # g
                              1.0])  # d

        self.eom = sym.Matrix([theta.diff() - omega,
                               I * omega.diff() + m * g * d * sym.sin(theta) - T])

        par_map = OrderedDict(zip(self.constant_symbols,
                                  self.constant_values))

        # Additional node equality contraints.
        theta, omega = sym.symbols('theta, omega', cls=sym.Function)
        instance_constraints = (2.0 * theta(0.0),
                                3.0 * theta(0.03) - sym.pi,
                                4.0 * omega(0.0),
                                5.0 * omega(0.03))

        self.collocator = \
            ConstraintCollocator(equations_of_motion=self.eom,
                                 state_symbols=self.state_symbols,
                                 num_collocation_nodes=4,
                                 node_time_interval=self.interval_value,
                                 known_parameter_map=par_map,
                                 instance_constraints=instance_constraints)

    def test_init(self):

        assert self.collocator.state_symbols == self.state_symbols
        assert self.collocator.state_derivative_symbols == \
            tuple([s.diff() for s in self.state_symbols])
        assert self.collocator.num_states == 2

    def test_sort_parameters(self):

        self.collocator._sort_parameters()

        assert self.collocator.known_parameters == self.constant_symbols
        assert self.collocator.num_known_parameters == 4

        assert self.collocator.unknown_parameters == tuple()
        assert self.collocator.num_unknown_parameters == 0

    def test_sort_trajectories(self):

        self.collocator._sort_trajectories()

        assert self.collocator.known_input_trajectories == tuple()
        assert self.collocator.num_known_input_trajectories == 0

        assert self.collocator.unknown_input_trajectories == self.specified_symbols
        assert self.collocator.num_unknown_input_trajectories == 1

    def test_discrete_symbols(self):

        self.collocator._discrete_symbols()

        thetai, omegai, thetap, omegap, Ti = self.discrete_symbols

        assert self.collocator.current_discrete_state_symbols == (thetai,
                                                                  omegai)
        assert self.collocator.previous_discrete_state_symbols == (thetap,
                                                                   omegap)
        assert self.collocator.current_discrete_specified_symbols == (Ti,)

    def test_discretize_eom(self):

        I, m, g, d = self.constant_symbols
        thetai, omegai, thetap, omegap, Ti = self.discrete_symbols
        h = self.collocator.time_interval_symbol

        expected = sym.Matrix([(thetai - thetap) / h - omegai,
                               I * (omegai - omegap) / h + m * g * d * sym.sin(thetai) - Ti])

        self.collocator._discretize_eom()

        zero = sym.simplify(self.collocator.discrete_eom - expected)

        assert zero == sym.Matrix([0, 0])

    def test_identify_function_in_instance_constraints(self):

        self.collocator._identify_functions_in_instance_constraints()

        theta, omega = sym.symbols('theta, omega', cls=sym.Function)

        expected = set((theta(0.0), theta(0.03), omega(0.0), omega(0.03)))

        assert self.collocator.instance_constraint_function_atoms == expected

    def test_find_free_index(self):

        theta, omega = sym.symbols('theta, omega', cls=sym.Function)

        self.collocator._identify_functions_in_instance_constraints()
        self.collocator._find_closest_free_index()

        expected = {theta(0.0): 0,
                    theta(0.03): 3,
                    omega(0.0): 4,
                    omega(0.03): 7}

        assert self.collocator.instance_constraints_free_index_map == expected

    def test_lambdify_instance_constraints(self):

        f = self.collocator._instance_constraints_func()

        extra_constraints = f(self.free)

        expected = np.array([2.0, 12.0 - np.pi, 20.0, 40.0])

        np.testing.assert_allclose(extra_constraints, expected)

    def test_instance_constraints_jacobian_indices(self):

        rows, cols = self.collocator._instance_constraints_jacobian_indices()

        np.testing.assert_allclose(rows, np.array([6, 7, 8, 9]))
        np.testing.assert_allclose(cols, np.array([0, 3, 4, 7]))

    def test_instance_constraints_jacobian_values(self):

        f = self.collocator._instance_constraints_jacobian_values_func()

        vals = f(self.free)

        np.testing.assert_allclose(vals, np.array([2.0, 3.0, 4.0, 5.0]))

    def test_gen_multi_arg_con_func(self):

        self.collocator._gen_multi_arg_con_func()

        # Make sure the parameters are in the correct order.
        constant_values = \
            np.array([self.constant_values[self.constant_symbols.index(c)]
                      for c in self.collocator.parameters])

        result = self.collocator._multi_arg_con_func(self.state_values,
                                                     self.specified_values,
                                                     constant_values,
                                                     self.interval_value)

        I, m, g, d = self.constant_values
        h = self.interval_value

        expected_dynamic = np.zeros(3)
        expected_kinematic = np.zeros(3)

        for i in [1, 2, 3]:

            thetai, omegai = self.state_values[:, i]
            thetap, omegap = self.state_values[:, i - 1]
            Ti = self.specified_values[i]

            expected_kinematic[i - 1] = (thetai - thetap) / h - omegai
            expected_dynamic[i - 1] = (I * (omegai - omegap) / h + m * g * d
                                       * sym.sin(thetai) - Ti)

        expected = np.hstack((expected_kinematic, expected_dynamic))

        np.testing.assert_allclose(result, expected)

    def test_jacobian_indices(self):

        row_idxs, col_idxs = self.collocator.jacobian_indices()

        expected_row_idxs = np.array([0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 1, 1, 1,
                                      1, 1, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 5,
                                      5, 5, 5, 5, 6, 7, 8, 9])

        expected_col_idxs = np.array([1, 5, 0, 4, 9, 1, 5, 0, 4, 9, 2, 6, 1,
                                      5, 10, 2, 6, 1, 5, 10, 3, 7, 2, 6, 11,
                                      3, 7, 2, 6, 11, 0, 3, 4, 7])

        np.testing.assert_allclose(row_idxs, expected_row_idxs)
        np.testing.assert_allclose(col_idxs, expected_col_idxs)

    def test_gen_multi_arg_con_jac_func(self):

        self.collocator._gen_multi_arg_con_jac_func()

        # Make sure the parameters are in the correct order.
        constant_values = \
            np.array([self.constant_values[self.constant_symbols.index(c)]
                      for c in self.collocator.parameters])

        jac_vals = self.collocator._multi_arg_con_jac_func(self.state_values,
                                                           self.specified_values,
                                                           constant_values,
                                                           self.interval_value)

        row_idxs, col_idxs = self.collocator.jacobian_indices()
        row_idxs = row_idxs[:-4]
        col_idxs = col_idxs[:-4]

        jacobian_matrix = sparse.coo_matrix((jac_vals, (row_idxs, col_idxs)))

        #                   thetai, omegai, thetap, omegap, Ti
        # theta : [            1/h,     -1,   -1/h,      0,  0],
        # omega : [g*m*cos(thetai),    I/h,      0,   -I/h, -1]])

        theta = self.state_values[0]
        I, m, g, d = self.constant_values
        h = self.interval_value

        expected_jacobian = np.array(
            # theta0,                       theta1,                       theta2,                       theta3, omega0, omega1, omega2, omega3,   T0, T1, T2, T3
            [[-1 / h,                        1 / h,                            0,                            0,      0,     -1,      0,      0,    0,  0,  0,  0],  # 1
             [     0,                       -1 / h,                        1 / h,                            0,      0,      0,     -1,      0,    0,  0,  0,  0],  # 2
             [     0,                            0,                       -1 / h,                        1 / h,      0,      0,      0,     -1,    0,  0,  0,  0],  # 3
             [     0, d * g * m * np.cos(theta[1]),                            0,                            0, -I / h,  I / h,      0,      0,    0, -1,  0,  0],  # 1
             [     0,                            0, d * g * m * np.cos(theta[2]),                            0,      0, -I / h,  I / h,      0,    0,  0, -1,  0],  # 2
             [     0,                            0,                            0, d * g * m * np.cos(theta[3]),      0,      0, -I / h,  I / h,    0,  0,  0, -1]], # 3
            dtype=float)

        np.testing.assert_allclose(jacobian_matrix.todense(), expected_jacobian)

    def test_generate_constraint_function(self):

        constrain = self.collocator.generate_constraint_function()

        result = constrain(self.free)

        expected_dynamic = np.zeros(3)
        expected_kinematic = np.zeros(3)

        I, m, g, d = self.constant_values
        h = self.interval_value

        expected_dynamic = np.zeros(3)
        expected_kinematic = np.zeros(3)

        for i in [1, 2, 3]:

            thetai, omegai = self.state_values[:, i]
            thetap, omegap = self.state_values[:, i - 1]
            Ti = self.specified_values[i]

            expected_kinematic[i - 1] = (thetai - thetap) / h - omegai
            expected_dynamic[i - 1] = (I * (omegai - omegap) / h + m * g * d
                                       * sym.sin(thetai) - Ti)
        theta_values = self.state_values[0]
        omega_values = self.state_values[1]

        expected_node_constraints = np.array([2.0 * theta_values[0],
                                              3.0 * theta_values[3] - np.pi,
                                              4.0 * omega_values[0],
                                              5.0 * omega_values[3]])

        expected = np.hstack((expected_kinematic, expected_dynamic,
                              expected_node_constraints))

        np.testing.assert_allclose(result, expected)

    def test_generate_jacobian_function(self):

        jacobian = self.collocator.generate_jacobian_function()

        jac_vals = jacobian(self.free)

        row_idxs, col_idxs = self.collocator.jacobian_indices()

        jacobian_matrix = sparse.coo_matrix((jac_vals, (row_idxs, col_idxs)))

        theta = self.state_values[0]
        I, m, g, d = self.constant_values
        h = self.interval_value

        expected_jacobian = np.array(
            # theta0,                       theta1,                       theta2,                       theta3, omega0, omega1, omega2, omega3,   T0, T1, T2, T3
            [[-1 / h,                        1 / h,                            0,                            0,      0,     -1,      0,      0,    0,  0,  0,  0],  # 1
             [     0,                       -1 / h,                        1 / h,                            0,      0,      0,     -1,      0,    0,  0,  0,  0],  # 2
             [     0,                            0,                       -1 / h,                        1 / h,      0,      0,      0,     -1,    0,  0,  0,  0],  # 3
             [     0, d * g * m * np.cos(theta[1]),                            0,                            0, -I / h,  I / h,      0,      0,    0, -1,  0,  0],  # 1
             [     0,                            0, d * g * m * np.cos(theta[2]),                            0,      0, -I / h,  I / h,      0,    0,  0, -1,  0],  # 2
             [     0,                            0,                            0, d * g * m * np.cos(theta[3]),      0,      0, -I / h,  I / h,    0,  0,  0, -1],  # 3
             [   2.0,                            0,                            0,                            0,      0,      0,      0,      0,    0,  0,  0,  0],
             [     0,                            0,                            0,                          3.0,      0,      0,      0,      0,    0,  0,  0,  0],
             [     0,                            0,                            0,                            0,    4.0,      0,      0,      0,    0,  0,  0,  0],
             [     0,                            0,                            0,                            0,      0,      0,      0,    5.0,    0,  0,  0,  0]],
            dtype=float)

        np.testing.assert_allclose(jacobian_matrix.todense(), expected_jacobian)
