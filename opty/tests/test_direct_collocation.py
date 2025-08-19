#!/usr/bin/env python

from collections import OrderedDict

import numpy as np
import sympy as sym
import sympy.physics.mechanics as mech
from sympy.physics.mechanics.models import n_link_pendulum_on_cart
from pytest import raises


from ..direct_collocation import Problem, ConstraintCollocator
from ..utils import (create_objective_function, sort_sympy, parse_free,
                     _coo_matrix)


def test_extra_algebraic(plot=False):
    """
    Chaplygin Sleigh example with a single nonholonomic constraint and the
    sleigh angle as an input.
    """

    m = sym.symbols('m', real=True)
    x, y, theta = mech.dynamicsymbols('x, y, theta', real=True)
    vx, vy = mech.dynamicsymbols('v_x, v_y', real=True)
    Fx, Fy = mech.dynamicsymbols('F_x, F_y')
    t = mech.dynamicsymbols._t

    states = (x, y, vx, vy)  # n states
    specifieds = (Fx, Fy, theta)

    # M equations of motion
    eom = sym.Matrix([
        m*vx.diff() - Fx,
        x.diff() - vx,
        m*vy.diff() - Fy,
        y.diff() - vy,
        -sym.sin(theta)*vx + sym.cos(theta)*vy,
        # NOTE : the following also works
        #-sym.sin(theta)*x.diff() + sym.cos(theta)*y.diff(),
    ])

    num_nodes = 100
    interval_value = 0.1
    dur = interval_value*(num_nodes - 1)

    obj_func = sym.Integral(Fx**2 + Fy**2 + theta**2, t)
    obj, obj_grad = create_objective_function(
        obj_func, states, specifieds, tuple(), num_nodes,
        interval_value, time_symbol=t)

    instance_constraints = (
        x.func(0.0),
        y.func(0.0),
        vx.func(0.0),
        vy.func(0.0),
        x.func(dur) - 1.0,
        y.func(dur) - 1.0,
        vx.func(dur),
        vy.func(dur),
    )

    par_map = {
        m: 1.0,
    }

    prob = Problem(
        obj,
        obj_grad,
        eom,
        states,
        num_nodes,
        interval_value,
        known_parameter_map=par_map,
        instance_constraints=instance_constraints,
        time_symbol=t,
        backend='numpy',
    )

    initial_guess = np.zeros(prob.num_free)
    solution, _ = prob.solve(initial_guess)

    if plot:
        prob.plot_trajectories(solution)


def test_pendulum():

    target_angle = np.pi
    duration = 10.0
    num_nodes = 51
    interval_value = duration / (num_nodes - 1)

    # Symbolic equations of motion
    # NOTE : h, real=True is used as a regression test for
    # https://github.com/csu-hmc/opty/issues/162
    I, m, g, h, t = sym.symbols('I, m, g, h, t', real=True)
    theta, omega, T = sym.symbols('theta, omega, T', cls=sym.Function)

    state_symbols = (theta(t), omega(t))
    specified_symbols = (T(t),)

    eom = sym.Matrix([theta(t).diff() - omega(t),
                      I*omega(t).diff() + m*g*h*sym.sin(theta(t)) - T(t)])

    # Specify the known system parameters.
    par_map = OrderedDict()
    par_map[I] = 1.0
    par_map[m] = 1.0
    par_map[g] = 9.81
    par_map[h] = 1.0

    # Specify the objective function and it's gradient.
    obj_func = sym.Integral(T(t)**2, t)
    obj, obj_grad = create_objective_function(
        obj_func, state_symbols, specified_symbols, tuple(), num_nodes,
        interval_value, time_symbol=t)

    # Specify the symbolic instance constraints, i.e. initial and end
    # conditions.
    instance_constraints = (theta(0.0),
                            theta(duration) - target_angle,
                            omega(0.0),
                            omega(duration))

    # This will test that a compilation works.
    prob = Problem(
        obj, obj_grad, eom, state_symbols, num_nodes, interval_value,
        known_parameter_map=par_map,
        instance_constraints=instance_constraints,
        time_symbol=t,
        bounds={T(t): (-2.0, 2.0)},
        show_compile_output=True,
        eom_bounds={0: (-20.0, 20.0), 1: (-10.0, 10.0)},
    )

    assert prob.collocator.num_instance_constraints == 4

    expected_low_con_bounds = np.zeros(prob.num_constraints)
    expected_low_con_bounds[:50] = -20.0
    expected_low_con_bounds[50:100] = -10.0
    np.testing.assert_allclose(prob._low_con_bounds, expected_low_con_bounds)

    expected_upp_con_bounds = np.zeros(prob.num_constraints)
    expected_upp_con_bounds[:50] = 20.0
    expected_upp_con_bounds[50:100] = 10.0
    np.testing.assert_allclose(prob._upp_con_bounds, expected_upp_con_bounds)


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
                   time_symbol=t,
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

    assert prob.collocator.num_instance_constraints == 0


class TestConstraintCollocator():

    def setup_method(self):

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
                                 known_trajectory_map=traj_map,
                                 time_symbol=t)

        self.numpy_collocator = \
            ConstraintCollocator(equations_of_motion=self.eom,
                                 state_symbols=self.state_symbols,
                                 num_collocation_nodes=4,
                                 node_time_interval=self.interval_value,
                                 known_parameter_map=par_map,
                                 known_trajectory_map=traj_map,
                                 time_symbol=t,
                                 backend='numpy')

    def test_init(self):

        assert self.collocator.state_symbols == self.state_symbols
        assert self.collocator.state_derivative_symbols == \
            tuple([s.diff() for s in self.state_symbols])
        assert self.collocator.num_states == 2
        assert self.collocator.num_collocation_nodes == 4

    def test_integration_method(self):
        with raises(ValueError):
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
            fi = self.specified_values[i:i + 1][0]
            fn = self.specified_values[i + 1:i + 2][0]

            expected_kinematic[i] = (xn - xi) / h - (vi + vn) / 2
            expected_dynamic[i] = (m * (vn - vi) / h + c * (vn + vi) / 2 + k
                                   * (xn + xi) / 2 - (fi + fn) / 2)

        expected = np.hstack((expected_kinematic, expected_dynamic))

        np.testing.assert_allclose(result, expected)

    def test_gen_multi_arg_con_jac_func_backward_euler(self):

        self.collocator._gen_multi_arg_con_jac_func()
        self.numpy_collocator._gen_multi_arg_con_jac_func()

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
        numpy_jac_vals = self.numpy_collocator._multi_arg_con_jac_func(
            self.state_values, self.specified_values, constant_values,
            self.interval_value)

        row_idxs, col_idxs = self.collocator.jacobian_indices()

        jacobian_matrix = _coo_matrix(jac_vals, row_idxs, col_idxs)

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

        np.testing.assert_allclose(jacobian_matrix, expected_jacobian)
        np.testing.assert_allclose(_coo_matrix(numpy_jac_vals, row_idxs,
                                               col_idxs), expected_jacobian)

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

        jacobian_matrix = _coo_matrix(jac_vals, row_idxs, col_idxs)

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

        np.testing.assert_allclose(jacobian_matrix, expected_jacobian)

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

        jacobian_matrix = _coo_matrix(jac_vals, row_idxs, col_idxs)

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

        np.testing.assert_allclose(jacobian_matrix, expected_jacobian)


class TestConstraintCollocatorUnknownTrajectories():

    def setup_method(self):

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

        jacobian_matrix = _coo_matrix(jac_vals, row_idxs, col_idxs)

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

        np.testing.assert_allclose(jacobian_matrix, expected_jacobian)

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

        jacobian_matrix = _coo_matrix(jac_vals, row_idxs, col_idxs)

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

        np.testing.assert_allclose(jacobian_matrix, expected_jacobian)

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

        jacobian_matrix = _coo_matrix(jac_vals, row_idxs, col_idxs)

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

        np.testing.assert_allclose(jacobian_matrix, expected_jacobian)


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

    def setup_method(self):

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
                                 instance_constraints=instance_constraints,
                                 time_symbol=t)

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

        jacobian_matrix = _coo_matrix(jac_vals, row_idxs, col_idxs)

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

        np.testing.assert_allclose(jacobian_matrix, expected_jacobian)

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

        jacobian_matrix = _coo_matrix(jac_vals, row_idxs, col_idxs)

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

        np.testing.assert_allclose(jacobian_matrix, expected_jacobian)


class TestConstraintCollocatorVariableDuration():

    def setup_method(self):

        m, g, d, t, h = sym.symbols('m, g, d, t, h')
        theta, omega, T = [f(t) for f in sym.symbols('theta, omega, T',
                                                     cls=sym.Function)]

        self.num_nodes = 4
        self.state_symbols = (theta, omega)
        self.constant_symbols = (m, g, d)
        self.specified_symbols = (T,)
        self.discrete_symbols = sym.symbols(
            'thetai, omegai, thetap, omegap, Ti', real=True)
        self.interval_symbol = h

        self.state_values = np.array([[1.0, 2.0, 3.0, 4.0],
                                      [5.0, 6.0, 7.0, 8.0]])
        self.specified_values = np.array([9.0, 10.0, 11.0, 12.0])
        self.constant_values = np.array([1.0, 9.81, 1.0])
        self.interval_value = 0.01
        self.time = np.array([0.0, 0.01, 0.02, 0.03])
        self.free = np.array([1.0, 2.0, 3.0, 4.0,  # theta
                              5.0, 6.0, 7.0, 8.0,  # omega
                              9.0, 10.0, 11.0, 12.0,  # T
                              1.0,  # m
                              9.81,  # g
                              1.0,  # d
                              0.01])  # h

        self.eom = sym.Matrix([theta.diff() - omega,
                               m*d**2*omega.diff() + m*g*d*sym.sin(theta) - T])

        par_map = OrderedDict(zip(self.constant_symbols,
                                  self.constant_values))

        # Additional node equality contraints.
        theta, omega = sym.symbols('theta, omega', cls=sym.Function)
        # If it is variable duration then use values 0 to N - 1 to specify
        # instance constraints instead of time.
        t0, tf = 0*h, (self.num_nodes - 1)*h
        instance_constraints = (theta(t0),
                                theta(tf) - sym.pi,
                                omega(t0),
                                omega(tf))

        self.collocator = ConstraintCollocator(
            equations_of_motion=self.eom,
            state_symbols=self.state_symbols,
            num_collocation_nodes=self.num_nodes,
            node_time_interval=self.interval_symbol,
            known_parameter_map=par_map,
            instance_constraints=instance_constraints,
            time_symbol=t)

    def test_init(self):

        assert self.collocator.state_symbols == self.state_symbols
        assert (self.collocator.state_derivative_symbols ==
                tuple([s.diff() for s in self.state_symbols]))
        assert self.collocator.num_states == 2

    def test_sort_parameters(self):

        self.collocator._sort_parameters()

        assert self.collocator.known_parameters == self.constant_symbols
        assert self.collocator.num_known_parameters == 3

        assert self.collocator.unknown_parameters == tuple()
        assert self.collocator.num_unknown_parameters == 0

    def test_sort_trajectories(self):

        self.collocator._sort_trajectories()

        assert self.collocator.known_input_trajectories == tuple()
        assert self.collocator.num_known_input_trajectories == 0

        assert (self.collocator.unknown_input_trajectories ==
                self.specified_symbols)
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

        m, g, d = self.constant_symbols
        thetai, omegai, thetap, omegap, Ti = self.discrete_symbols
        h = self.collocator.time_interval_symbol

        expected = sym.Matrix([
            (thetai - thetap)/h - omegai,
            m*d**2*(omegai - omegap)/h + m*g*d*sym.sin(thetai) - Ti])

        self.collocator._discretize_eom()

        zero = sym.simplify(self.collocator.discrete_eom - expected)

        assert zero == sym.Matrix([0, 0])

    def test_identify_function_in_instance_constraints(self):

        self.collocator._identify_functions_in_instance_constraints()

        theta, omega = sym.symbols('theta, omega', cls=sym.Function)

        expected = set((theta(0*self.interval_symbol),
                        theta((self.num_nodes - 1)*self.interval_symbol),
                        omega(0*self.interval_symbol),
                        omega((self.num_nodes - 1)*self.interval_symbol)))

        assert self.collocator.instance_constraint_function_atoms == expected

    def test_find_free_index(self):

        theta, omega = sym.symbols('theta, omega', cls=sym.Function)

        self.collocator._identify_functions_in_instance_constraints()
        self.collocator._find_closest_free_index()

        expected = {
            theta(0*self.interval_symbol): 0,
            theta((self.num_nodes - 1)*self.interval_symbol): 3,
            omega(0*self.interval_symbol): 4,
            omega((self.num_nodes - 1)*self.interval_symbol): 7,
        }

        assert self.collocator.instance_constraints_free_index_map == expected

    def test_lambdify_instance_constraints(self):

        f = self.collocator._instance_constraints_func()

        extra_constraints = f(self.free)

        expected = np.array([1.0, 4.0 - np.pi, 5.0, 8.0])

        np.testing.assert_allclose(extra_constraints, expected)

    def test_instance_constraints_jacobian_indices(self):

        rows, cols = self.collocator._instance_constraints_jacobian_indices()

        np.testing.assert_allclose(rows, np.array([6, 7, 8, 9]))
        np.testing.assert_allclose(cols, np.array([0, 3, 4, 7]))

    def test_instance_constraints_jacobian_values(self):

        f = self.collocator._instance_constraints_jacobian_values_func()

        vals = f(self.free)

        np.testing.assert_allclose(vals, np.array([1.0, 1.0, 1.0, 1.0]))

    def test_gen_multi_arg_con_func(self):

        self.collocator._gen_multi_arg_con_func()

        # Make sure the parameters are in the correct order.
        constant_values = np.array([
            self.constant_values[self.constant_symbols.index(c)]
            for c in self.collocator.parameters])

        result = self.collocator._multi_arg_con_func(self.state_values,
                                                     self.specified_values,
                                                     constant_values,
                                                     self.interval_value)

        m, g, d = self.constant_values
        h = self.interval_value

        expected_dynamic = np.zeros(3)
        expected_kinematic = np.zeros(3)

        for i in [1, 2, 3]:

            thetai, omegai = self.state_values[:, i]
            thetap, omegap = self.state_values[:, i - 1]
            Ti = self.specified_values[i]

            expected_kinematic[i - 1] = (thetai - thetap)/h - omegai
            expected_dynamic[i - 1] = (m*d**2*(omegai - omegap)/h +
                                       m*g*d*sym.sin(thetai) - Ti)

        expected = np.hstack((expected_kinematic, expected_dynamic))

        np.testing.assert_allclose(result, expected)

    def test_jacobian_indices(self):

        row_idxs, col_idxs = self.collocator.jacobian_indices()

        expected_row_idxs = np.array([0, 0, 0, 0, 0, 0,
                                      3, 3, 3, 3, 3, 3,
                                      1, 1, 1, 1, 1, 1,
                                      4, 4, 4, 4, 4, 4,
                                      2, 2, 2, 2, 2, 2,
                                      5, 5, 5, 5, 5, 5,
                                      6, 7, 8, 9])

        expected_col_idxs = np.array([1, 5, 0, 4, 9, 12,
                                      1, 5, 0, 4, 9, 12,
                                      2, 6, 1, 5, 10, 12,
                                      2, 6, 1, 5, 10, 12,
                                      3, 7, 2, 6, 11, 12,
                                      3, 7, 2, 6, 11, 12,
                                      0, 3, 4, 7])

        np.testing.assert_allclose(row_idxs, expected_row_idxs)
        np.testing.assert_allclose(col_idxs, expected_col_idxs)

    def test_gen_multi_arg_con_jac_func(self):

        self.collocator._gen_multi_arg_con_jac_func()

        # Make sure the parameters are in the correct order.
        constant_values = np.array([
            self.constant_values[self.constant_symbols.index(c)]
            for c in self.collocator.parameters])

        jac_vals = self.collocator._multi_arg_con_jac_func(
            self.state_values, self.specified_values, constant_values,
            self.interval_value)

        row_idxs, col_idxs = self.collocator.jacobian_indices()
        # skip instance constraints
        row_idxs = row_idxs[:-4]
        col_idxs = col_idxs[:-4]

        jacobian_matrix = _coo_matrix(jac_vals, row_idxs, col_idxs)

        theta = self.state_values[0]
        omega = self.state_values[1]
        m, g, d = self.constant_values
        h = self.interval_value

        expected_jacobian = np.array(
            # theta0,                 theta1,                 theta2,               # theta3,    omega0,    omega1,    omega2,   omega3, T0, T1, T2, T3, h
            [[  -1/h,                    1/h,                      0,                      0,         0,        -1,         0,        0,  0,  0,  0,  0, -(theta[1] - theta[0])/h**2],  # 1
             [     0,                   -1/h,                    1/h,                      0,         0,         0,        -1,        0,  0,  0,  0,  0, -(theta[2] - theta[1])/h**2],  # 2
             [     0,                      0,                   -1/h,                    1/h,         0,         0,         0,       -1,  0,  0,  0,  0, -(theta[3] - theta[2])/h**2],  # 3
             [     0, d*g*m*np.cos(theta[1]),                      0,                      0, -m*d**2/h,  m*d**2/h,         0,        0,  0, -1,  0,  0, -d**2*m*(omega[1] - omega[0])/h**2],  # 1
             [     0,                      0, d*g*m*np.cos(theta[2]),                      0,         0, -m*d**2/h,  m*d**2/h,        0,  0,  0, -1,  0, -d**2*m*(omega[2] - omega[1])/h**2],  # 2
             [     0,                      0,                      0, d*g*m*np.cos(theta[3]),         0,         0, -m*d**2/h, m*d**2/h,  0,  0,  0, -1, -d**2*m*(omega[3] - omega[2])/h**2]], # 3
            dtype=float)

        np.testing.assert_allclose(jacobian_matrix, expected_jacobian)

    def test_generate_constraint_function(self):

        constrain = self.collocator.generate_constraint_function()

        result = constrain(self.free)

        expected_dynamic = np.zeros(3)
        expected_kinematic = np.zeros(3)

        m, g, d = self.constant_values
        h = self.interval_value

        expected_dynamic = np.zeros(3)
        expected_kinematic = np.zeros(3)

        for i in [1, 2, 3]:

            thetai, omegai = self.state_values[:, i]
            thetap, omegap = self.state_values[:, i - 1]
            Ti = self.specified_values[i]

            expected_kinematic[i - 1] = (thetai - thetap)/h - omegai
            expected_dynamic[i - 1] = (m*d**2*(omegai - omegap)/h +
                                       m*g*d*sym.sin(thetai) - Ti)

        theta_values = self.state_values[0]
        omega_values = self.state_values[1]

        expected_node_constraints = np.array([theta_values[0],
                                              theta_values[3] - np.pi,
                                              omega_values[0],
                                              omega_values[3]])

        expected = np.hstack((expected_kinematic, expected_dynamic,
                              expected_node_constraints))

        np.testing.assert_allclose(result, expected)

    def test_generate_jacobian_function(self):

        jacobian = self.collocator.generate_jacobian_function()

        jac_vals = jacobian(self.free)

        row_idxs, col_idxs = self.collocator.jacobian_indices()

        jacobian_matrix = _coo_matrix(jac_vals, row_idxs, col_idxs)

        th = self.state_values[0]
        om = self.state_values[1]
        m, g, d = self.constant_values
        h = self.interval_value

        expected_jacobian = np.array(
            #  th0,                 th1,                 th2,                th3,        om0,       om1,       om2,      om3, T0, T1, T2, T3,                             h
            [[-1/h,                 1/h,                   0,                  0,          0,        -1,         0,        0,  0,  0,  0,  0,        -(th[1] - th[0])/h**2],  # 1
             [   0,                -1/h,                 1/h,                  0,          0,         0,        -1,        0,  0,  0,  0,  0,        -(th[2] - th[1])/h**2],  # 2
             [   0,                   0,                -1/h,                1/h,          0,         0,         0,       -1,  0,  0,  0,  0,        -(th[3] - th[2])/h**2],  # 3
             [   0, d*g*m*np.cos(th[1]),                   0,                   0, -m*d**2/h,  m*d**2/h,         0,        0,  0, -1,  0,  0, -d**2*m*(om[1] - om[0])/h**2],  # 1
             [   0,                   0, d*g*m*np.cos(th[2]),                   0,         0, -m*d**2/h,  m*d**2/h,        0,  0,  0, -1,  0, -d**2*m*(om[2] - om[1])/h**2],  # 2
             [   0,                   0,                   0, d*g*m*np.cos(th[3]),         0,         0, -m*d**2/h, m*d**2/h,  0,  0,  0, -1, -d**2*m*(om[3] - om[2])/h**2],  # 3
             [ 1.0,                   0,                   0,                   0,         0,         0,         0,        0,  0,  0,  0,  0,                            0],
             [   0,                   0,                   0,                 1.0,         0,         0,         0,        0,  0,  0,  0,  0,                            0],
             [   0,                   0,                   0,                   0,       1.0,         0,         0,        0,  0,  0,  0,  0,                            0],
             [   0,                   0,                   0,                   0,         0,         0,         0,      1.0,  0,  0,  0,  0,                            0]],
            dtype=float)

        np.testing.assert_allclose(jacobian_matrix, expected_jacobian)


def test_known_and_unknown_order():

    kane = n_link_pendulum_on_cart(n=3, cart_force=True, joint_torques=True)

    states = kane.q.col_join(kane.u)

    eom = kane.mass_matrix_full @ states.diff() - kane.forcing_full

    g, l0, l1, l2, m0, m1, m2, m3, t = sort_sympy(eom.free_symbols)

    # leave two parameters free and disorder the entries to the dictionary
    par_map = {}
    par_map[l1] = 1.5
    par_map[l0] = 1.0
    par_map[m3] = 2.5
    par_map[g] = 9.81
    par_map[m1] = 1.5

    (u1d, q2d, q3d, u3d, q0d, q1d, u0d, u2d, F, T1, T2, T3, q0, q1, q2, q3, u0,
     u1, u2, u3) = sort_sympy(mech.find_dynamicsymbols(eom))

    num_nodes = 51
    interval = 0.1

    # order in the dictionary should not match sort_sympy()
    traj_map = {
        T1: np.zeros(num_nodes),
        F: np.ones(num_nodes),
    }

    col = ConstraintCollocator(
        equations_of_motion=eom,
        state_symbols=states,
        num_collocation_nodes=num_nodes,
        node_time_interval=interval,
        known_parameter_map=par_map,
        known_trajectory_map=traj_map,
        time_symbol=t,
    )

    assert col.input_trajectories == (T1, F, T2, T3)
    assert col.known_input_trajectories == (T1, F)
    assert col.known_parameters == (l1, l0, m3, g, m1)
    assert col.unknown_input_trajectories == (T2, T3)
    assert col.unknown_parameters == (l2, m0, m2)
    assert col.parameters == (l1, l0, m3, g, m1, l2, m0, m2)
    assert col.state_symbols == (q0, q1, q2, q3, u0, u1, u2, u3)

def test_for_algebraic_eoms():
    """
    If algebraic equations of motion are given to Problem, a ValueError should
    be raised. This a a test for this
    """

    target_angle = np.pi
    duration = 10.0
    num_nodes = 500
    interval_value = duration / (num_nodes - 1)

    I, m, g, h, t = sym.symbols('I, m, g, h, t', real=True)
    theta, omega, T = sym.symbols('theta, omega, T', cls=sym.Function)

    state_symbols = (theta(t), omega(t))
    specified_symbols = (T(t),)

    # removed the .diff(t) from eom to get AEs
    eom = sym.Matrix([theta(t) - omega(t),
                     I*omega(t) + m*g*h*sym.sin(theta(t)) - T(t)])

    # Specify the known system parameters.
    par_map = {}
    par_map[I] = 1.0
    par_map[m] = 1.0
    par_map[g] = 9.81
    par_map[h] = 1.0

    # Specify the objective function and it's gradient.
    obj_func = sym.Integral(T(t)**2, t)
    obj, obj_grad = create_objective_function(
        obj_func, state_symbols, specified_symbols, tuple(), num_nodes,
        interval_value, time_symbol=t)

    # Specify the symbolic instance constraints, i.e. initial and end
    # conditions.
    instance_constraints = (theta(0.0),
                            theta(duration) - target_angle,
                            omega(0.0),
                            omega(duration))

    # This will test that a ValueError is raised.
    with raises(ValueError) as excinfo:
        prob = Problem(
            obj, obj_grad, eom, state_symbols, num_nodes, interval_value,
            known_parameter_map=par_map,
            instance_constraints=instance_constraints,
            time_symbol=t,
            bounds={T(t): (-2.0, 2.0)},
        )

    assert excinfo.type is ValueError


def test_prob_parse_free():
    """
    Test for parse_free method of Problem class.
    ===========================================

    This tests whether the parse_free method of the Problem class works as
    the parse_free in utils.
    It also tests that only 'numpy' and 'cython' backends are accepted ands
    raises a ValueError for any other backend.

    **States**

    - :math:`x_1, x_2, ux_1, ux_2` : state variables

    **Control**

    - :math:`u_1, u_2` : control variable

    """

    t = mech.dynamicsymbols._t

    x1, x2, ux1, ux2 = mech.dynamicsymbols('x1 x2 ux1 ux2')
    u1, u2 = mech.dynamicsymbols('u1 u2')
    h = sym.symbols('h')
    a, b = sym.symbols('a b')

    # equations of motion.
    # (No meaning, just for testing)
    eom = sym.Matrix([
            -x1.diff(t) + ux1,
            -x2.diff(t) + ux2,
            -ux1.diff(t) + a*u1,
            -ux2.diff(t) + b*u2,
    ])

    # Set up and Solve the Optimization Problem
    num_nodes = 11
    t0, tf = 0.0, 0.9
    state_symbols = (x1, x2, ux1, ux2)
    control_symbols = (u1, u2)

    interval_value = (tf - t0)/(num_nodes - 1)
    times = np.linspace(t0, tf, num_nodes)

    # Specify the symbolic instance constraints, as per the example.
    instance_constraints = (
        x1.func(t0) - 1.0,
        x2.func(t0) + 1.0,
    )

    # Specify the objective function and form the gradient.

    def obj(free):
        return sum([free[i]**2 for i in range(2*num_nodes)])

    def obj_grad(free):
        grad = np.zeros_like(free)
        grad[:2*num_nodes] = 2*free[:2*num_nodes]
        return grad

    # Create the optimization problem and set any options.
    prob = Problem(
            obj,
            obj_grad,
            eom,
            state_symbols,
            num_nodes,
            interval_value,
            instance_constraints=instance_constraints,
)

    # Give some estimates for the trajectory.
    initial_guess = np.random.rand(prob.num_free)
    initial_guess1 = initial_guess

    # check whether same results.
    statesu, controlsu, constantsu = parse_free(initial_guess1,
            len(state_symbols), len(control_symbols), num_nodes)

    states, controls, constants = prob.parse_free(initial_guess)
    np.testing.assert_allclose(states, statesu)
    np.testing.assert_allclose(controls, controlsu)
    np.testing.assert_allclose(constants, constantsu)

    # test with variable interval_value
    interval_value = h
    t0, tf = 0.0, (num_nodes - 1)*interval_value
    def obj(free):
        return sum([free[i]**2 for i in range(2*num_nodes)])

    def obj_grad(free):
        grad = np.zeros_like(free)
        grad[:2*num_nodes] = 2*free[:2*num_nodes]
        return grad

    # Create the optimization problem and set any options.
    prob = Problem(
            obj,
            obj_grad,
            eom,
            state_symbols,
            num_nodes,
            interval_value,
            instance_constraints=instance_constraints,
)

    # Give some estimates for the trajectory.
    initial_guess = np.random.rand(prob.num_free)
    initial_guess1 = initial_guess

    # check whether same results.
    statesu, controlsu, constantsu, timeu = parse_free(initial_guess1,
        len(state_symbols), len(control_symbols),
        num_nodes, variable_duration=True)

    states, controls, constants, times = prob.parse_free(initial_guess)
    np.testing.assert_allclose(states, statesu)
    np.testing.assert_allclose(controls, controlsu)
    np.testing.assert_allclose(constants, constantsu)
    np.testing.assert_allclose(timeu, times)

    # check that only 'numpy' and 'cython' backends are accepted as backend
    with raises(ValueError):
        Problem(
            obj,
            obj_grad,
            eom,
            state_symbols,
            num_nodes,
            interval_value,
            time_symbol=t,
            backend='nonsensical',
        )

def test_one_eom_only():
    """
    Only one differential equation should work. This tests for the corrrect
    shape of the constraints and jacobian.

    """
    # Equations of motion.
    t = mech.dynamicsymbols._t
    y, u = mech.dynamicsymbols('y u')

    eom = sym.Matrix([-y.diff(t) - y**3 + u])

    t0, tf = 0.0, 10.0
    num_nodes = 100
    interval_value = (tf - t0)/(num_nodes - 1)

    state_symbols = (y, )
    specified_symbols = (u,)

    # Specify the objective function and form the gradient.
    obj_func = sym.Integral(y**2 + u**2, t)
    obj, obj_grad = create_objective_function(
        obj_func,
        state_symbols,
        specified_symbols,
        tuple(),
        num_nodes,
        node_time_interval=interval_value
    )

    # Specify the symbolic instance constraints.
    instance_constraints = (
        y.func(t0) - 1,
        y.func(tf) - 1.5,
    )

        # Create the optimization problem and set any options.
    prob = Problem(
        obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        instance_constraints=instance_constraints,
        )

    initial_guess = np.zeros(prob.num_free)
    initial_guess[0] = 1.0
    initial_guess[num_nodes-1] = 1.5

    # assert that prob.constraints and prob.jacobian have the correct shape.
    length = 1*(num_nodes-1) + 2
    assert prob.constraints(initial_guess).shape == (length,)

    length = (2*1 + 1 + 0 + 0) * (1*(num_nodes-1)) + 2
    assert prob.jacobian(initial_guess).shape == (length,)

def test_duplicate_state_symbols():
    """Test for duplicate state symbols and for number of state_symbols is
    equal to the number of of eoms"""

    x, ux, z = mech.dynamicsymbols('x, ux, z')
    t = mech.dynamicsymbols._t
    m = sym.symbols('m')
    F = mech.dynamicsymbols('F')

    eom = sym.Matrix([
        -x.diff(t) + ux,
        -ux.diff(t) + F/m,
        ])

    par_map = {m: 1.0}

    num_nodes = 76

    t0, tf = 0.0, 1.0
    interval_value = tf/(num_nodes - 1)

    def obj(free):
        Fx = free[2*num_nodes:3*num_nodes]
        return interval_value*np.sum(Fx**2)

    def obj_grad(free):
        grad = np.zeros_like(free)
        l1 = 2*num_nodes
        l2 = 3*num_nodes
        grad[l1: l2] = 2.0*free[l1:l2]*interval_value
        return grad

    instance_constraints = (
        x.func(t0),
        ux.func(t0),
        x.func(tf) - 1.0,
        ux.func(tf),
    )

     # Test for duplicate state symbols
    state_symbols = (x, ux, x)
    with raises(ValueError):
        prob = Problem(
            obj,
            obj_grad,
            eom,
            state_symbols,
            num_nodes,
            interval_value,
            known_parameter_map=par_map,
            instance_constraints=instance_constraints,
        )

    # Test that No of state_symbols is equal to the No of time derivatives
    state_symbols = (x,)
    with raises(ValueError):
        prob = Problem(
            obj,
            obj_grad,
            eom,
            state_symbols,
            num_nodes,
            interval_value,
            known_parameter_map=par_map,
            instance_constraints=instance_constraints,
        )


def test_attributes_read_only():
    """
    Test to ensure the ConstraintCollocator attributes are read-only.
    """

    # random optimization problem
    x1, x2, x3 = mech.dynamicsymbols('x1 x2 x3')
    ux1, ux2, ux3 = mech.dynamicsymbols('ux1 ux2 ux3')
    u1, u2, u3 = mech.dynamicsymbols('u1 u2 u3')
    a1, a2, a3, a4, a5, a6 = sym.symbols('a1 a2 a3 a4 a5 a6')
    h = sym.symbols('h')

    t = mech.dynamicsymbols._t
    eom = sym.Matrix([
        -x1.diff(t) + ux1 + a3,
        -x2.diff(t) + ux2 + a5,
        -x3.diff(t) + ux3,
        -ux1.diff(t) + a6*x1 + a5*u1,
        -ux2.diff(t) + a4*x2 + a3*u2,
        -ux3.diff(t) + a2*x3 + a1*u3,
    ])

    # Set up the Optimization Problem

    state_symbols = [x1, x2, x3, ux1, ux2, ux3]

    num_nodes = 11
    h = sym.symbols('h')

    # Specify the known symbols.
    par_map = {}
    par_map[a2] = 1.0
    par_map[a4] = 2.0
    par_map[a6] = 3.0

    duration = (num_nodes - 1)*h
    t0, tf = 0.0, duration
    interval_value = h

    # Set up the instance constraints, the bounds and Problem.

    test = ConstraintCollocator(
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        known_parameter_map=par_map,
        time_symbol=t,
    )

    # Test if all these attributes are read-only
    for XX in [
        'current_discrete_state_symbols',
        'current_discrete_specified_symbols',
        'current_known_discrete_specified_symbols',
        'current_unknown_discrete_specified_symbols',
        'discrete_eom',
        'eom',
        'input_trajectories',
        'instance_constraints',
        'known_input_trajectories',
        'known_parameters',
        'known_parameter_map',
        'known_trajectory_map',
        'next_known_discrete_specified_symbols',
        'next_discrete_state_symbols',
        'next_unknown_discrete_specified_symbols',
        'node_time_interval',
        'num_collocation_nodes',
        'num_constraints',
        'num_free',
        'num_input_trajectories',
        'num_instance_constraints',
        'num_known_input_trajectories',
        'num_known_parameters',
        'num_parameters',
        'num_states',
        'num_unknown_input_trajectories',
        'num_unknown_parameters',
        'parameters',
        'parallel',
        'previous_discrete_state_symbols',
        'show_compile_output',
        'state_derivative_symbols',
        'state_symbols',
        'time_interval_symbol',
        'time_symbol',
        'tmp_dir',
        'unknown_input_trajectories',
        'unknown_parameters',
        ]:

        with raises(AttributeError):
            setattr(test, XX, 5)


def test_time_vector():
    """Test to ensure that time_vector retunrs the correct values.
    Many wrong time vectors and a few right ones are given, the method should
    give the correct time vector, and not any of the wrong ones.
    So, wrong_time_vector is mostly an incorrect one.
    """
    x, ux = mech.dynamicsymbols('x, ux')
    t = mech.dynamicsymbols._t

    # just random eoms, no physical meaning.
    eom = sym.Matrix([
        -x.diff(t) + ux,
        -ux.diff(t) + 2,
    ])

    # An example of arange() not giving the correct answer is:
    # step, n = 0.007, 25; len(np.arange(0.0, 0.0 + step*n, step)) -> 26!
    num_nodes = 25
    interval_value = 0.007

    prob = Problem(
        lambda x: 1.0,
        lambda x: x,
        eom,
        (x, ux),
        num_nodes,
        interval_value,
        time_symbol=t,
        backend='numpy'
    )

    # if arange was used it would add 0.175 as the 26th value to this array:
    expected_time = np.array([
        0.000, 0.007, 0.014, 0.021, 0.028, 0.035, 0.042, 0.049, 0.056,
        0.063, 0.070, 0.077, 0.084, 0.091, 0.098, 0.105, 0.112, 0.119,
        0.126, 0.133, 0.140, 0.147, 0.154, 0.161, 0.168,
    ])

    time = prob.time_vector()

    np.testing.assert_allclose(time, expected_time)

    # whether or not a solution is given should not change the result
    time = prob.time_vector(start_time=0.03901)
    np.testing.assert_allclose(time, expected_time + 0.03901)

    # whether or not a solution is given should not change the result
    solution = np.ones(prob.num_free)
    time = prob.time_vector(solution=solution)
    np.testing.assert_allclose(time, expected_time)

    # variable time interval
    h = sym.symbols('h')

    prob = Problem(
        lambda x: 1.0,
        lambda x: x,
        eom,
        (x, ux),
        num_nodes,
        h,
        time_symbol=t,
        backend='numpy'
    )

    # solution must be given
    with raises(ValueError):
        prob.time_vector()

    # solution must be given
    with raises(ValueError):
        prob.time_vector(start_time=12.0)

    # make sure passing solution works
    solution = np.ones(prob.num_free)
    solution[-1] = 0.007
    time = prob.time_vector(solution=solution)
    np.testing.assert_allclose(time, expected_time)

    # make sure start_time works
    solution = np.ones(prob.num_free)
    solution[-1] = 0.007
    time = prob.time_vector(solution=solution, start_time=0.014)
    np.testing.assert_allclose(time, expected_time + 0.014)

    # final time > initial time
    solution[-1] = 0.0001
    with raises(ValueError):
        prob.time_vector(solution, start_time=500.0)

    # interval_value must be greater than zero
    solution[-1] = 0.0
    with raises(ValueError):
        prob.time_vector(solution)


def test_check_bounds_conflict():
    """Test to ensure that the method of Problem, bounds_conflict_initial_guess
    raises a ValueError when the initial guesses violates the bounds.
    Then the test that the kwarg respect_bounds works as expected in solve.
    Test if invalid keys in the eom_bound are detected.
    """

    x, y, z, ux, uy, uz = mech.dynamicsymbols('x y z ux uy uz')
    u1, u2, u3 = mech.dynamicsymbols('u1 u2 u3')
    a1, a2, a3 = sym.symbols('a1 a2 a3')
    b1, b2, b3 = sym.symbols('b1 b2 b3')
    t = mech.dynamicsymbols._t

    # just random eoms, no physical meaning.
    eom = sym.Matrix([
        -x.diff(t) + ux,
        -ux.diff(t) + a1 * x + u1 + b1,
        -y.diff(t) + uy,
        -uy.diff(t) + a2 * y + u2 + b2,
        -z.diff(t) + uz,
        -uz.diff(t) + a3 * z + u3 + b3,
    ])

    state_symbols = (x, y, z, ux, uy, uz)
    par_map = {b1: 1.0,
               b2: 2.0,
               b3: 3.0
    }
    num_nodes = 3

    # A: constant time interval
    t0, tf = 0.0, 1.0
    interval_value = tf/(num_nodes - 1)

    def obj(free):
        Fx = free[2*num_nodes:3*num_nodes]
        return interval_value*np.sum(Fx**2)

    def obj_grad(free):
        grad = np.zeros_like(free)
        l1 = 2*num_nodes
        l2 = 3*num_nodes
        grad[l1: l2] = 2.0*free[l1:l2]*interval_value
        return grad

    bounds= {
        a1: (-1.0, 1.0),
        a2: (-1.0, 1.0),
        a3: (-1.0, 1.0),

        u1: (-1.0, 1.0),
        u2: (-1.0, 1.0),
        u3: (-1.0, 1.0),

        x: (-1.0, 1.0 ),
        ux: (-1.0, 1.0),
        y: (-1.0, 1.0),
        uy: (-np.inf, 1.0),
        z: (1.0, -1.0),
        uz: (1.0, -np.inf),
    }

    # check for wrong bounds
    prob = Problem(
        obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        known_parameter_map=par_map,
        bounds=bounds,
        time_symbol=t,
        backend='numpy'
    )

    initial_guess = np.zeros(prob.num_free)
    with raises(ValueError):
        prob.check_bounds_conflict(initial_guess)

    # check for reversed eom_bounds
    eom_bounds = {0: (-1.0, 1.0),
                  1: (1.0, -1.0),
                  }

    prob = Problem(
        obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        known_parameter_map=par_map,
        eom_bounds=eom_bounds,
        time_symbol=t,
        backend='numpy'
    )

    initial_guess = np.zeros(prob.num_free)
    with raises(ValueError):
        prob.check_bounds_conflict(initial_guess)

    # check for invalid keys in eom_bounds
    eom_bounds_bad = {0: (-1.0, 1.0) , 6: (0.0, 1.0), 'bad': (0.0, 1.0)}

    with raises(ValueError):
        prob = Problem(
            obj,
            obj_grad,
            eom,
            state_symbols,
            num_nodes,
            interval_value,
            known_parameter_map=par_map,
            eom_bounds=eom_bounds_bad,
            time_symbol=t,
            backend='numpy'
        )

    # check for values outside the bounds
    bounds[z] = (-1.0, 1.0)
    bounds[uz] = (-1.0, 1.0)
    initial_guess = np.zeros(prob.num_free)
    start_idx = np.random.randint(0, num_nodes - 1)
    for i in range(9):
        initial_guess[start_idx + i*num_nodes] = 10.0
    with raises(ValueError):
        prob.check_bounds_conflict(initial_guess)


    # B: variable time interval
    h = sym.symbols('h')
    interval_value = h
    t0, tf = 0.0, (num_nodes - 1)*h

    def obj(free):
        Fx = free[6*num_nodes:9*num_nodes]
        return free[-1]*np.sum(Fx**2)

    def obj_grad(free):
        grad = np.zeros_like(free)
        l1 = 6*num_nodes
        l2 = 9*num_nodes
        grad[l1: l2] = 2.0*free[l1:l2]*free[-1]
        grad[-1] = np.sum(free[l1:l2]**2)
        return grad

    instance_constraints = (
        x.func(t0),
        ux.func(t0),
        x.func(tf) - 1.0,
        ux.func(tf),
    )
    bounds= {
        a1: (-1.0, 1.0),
        a2: (-1.0, 1.0),
        a3: (-1.0, 1.0),

        u1: (-np.inf, 1.0),
        u2: (-1.0, 1.0),
        u3: (1.0, 1.0),

        x: (-np.inf, 1.0),
        ux: (-1.0, 1.0),
        y: (-1.0, 1.0),
        uy: (-1.0, 1.0),
        z: (-1.0, 1.0),
        uz: (-1.0, np.inf),
        h: (0.0, 0.4 )
    }

    prob = Problem(
        obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        known_parameter_map=par_map,
        instance_constraints=instance_constraints,
        bounds=bounds,
        time_symbol=t,
        backend='numpy'
    )

    initial_guess = np.zeros(prob.num_free)
    start_idx = np.random.randint(0, num_nodes - 1)
    # check for values outside the bounds
    for i in range(9):
        initial_guess[start_idx + i*num_nodes] = -10.0
    initial_guess[-1] = 0.5
    with raises(ValueError):
        prob.check_bounds_conflict(initial_guess)

    # check for wrong bounds
    bounds[a1] = (1.0, -1.0)
    bounds[x] = (np.inf, -1.0)
    bounds[uy] = (1.0, -np.inf)

    prob = Problem(
        obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        known_parameter_map=par_map,
        instance_constraints=instance_constraints,
        bounds=bounds,
        time_symbol=t,
        backend='numpy'
    )

    initial_guess = np.zeros(prob.num_free)
    with raises(ValueError):
        prob.check_bounds_conflict(initial_guess)


    # C: respect_bounds=True: initial guess must be within bounds, else a
    # ValueError is raised.
    with raises(ValueError):
        _, _ = prob.solve(initial_guess, respect_bounds=True)

    # D: respect_bounds=False. Bounds are ignored.
    _, _ = prob.solve(initial_guess)

    # E without bounds no ValueError is raised
    prob = Problem(
        obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        known_parameter_map=par_map,
        instance_constraints=instance_constraints,
        time_symbol=t,
        backend='numpy'
    )

    initial_guess = np.ones(prob.num_free) * 10.0
    prob.check_bounds_conflict(initial_guess)

    # F Initial guess within bounds, no ValueError should be raised
    bounds= {
        a1: (-np.inf, 1.0),
        a2: (-1.0, 1.0),
        a3: (-1.0, np.inf),

        u1: (-np.inf, 1.0),
        u2: (-1.0, 1.0),
        u3: (-1.0, np.inf),

        x: (-1.0, 1.0 ),
        ux: (-1.0, 1.0),
        y: (-1.0, 1.0),
        uy: (-np.inf, 1.0),
        z: (-1.0, 1.0),
        uz: (-1.0, np.inf),
    }

    prob = Problem(
        obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        known_parameter_map=par_map,
        bounds=bounds,
        time_symbol=t,
        backend='numpy'
    )

    initial_guess = np.zeros(prob.num_free)
    prob.check_bounds_conflict(initial_guess)
