#!/usr/bin/env python

from collections import OrderedDict

import numpy as np
import sympy as sym
from scipy import sparse

from direct_collocation import ConstraintCollocator


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
        self.free = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0])

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
