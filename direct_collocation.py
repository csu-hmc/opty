#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
import sympy as sym
from sympy.physics import mechanics as me
import ipopt

from simulate import output_equations
from utils import ufuncify_matrix, parse_free


class Problem(ipopt.problem):

    def __init__(self, N, n, q, obj, obj_grad, con, con_jac,
                 con_jac_indices):
        """

        Parameters
        ----------
        N : integer
            Number of discretization points during the time range.
        n : integer
            Number of states in the system.
        q : integer
           Number of free model parameters, i.e. the number of model
           constants which are included in the free optimization parameters.
        obj : function
            Returns the value of the objective function.
        obj_grad : function
            Returns the gradient of the objective function.
        con : function
            Returns the value of the constraints.
        con_jac : function
            Returns the Jacobian of the constraints.
        con_jac_indices : function
            Returns the indices of the non-zero values in the Jacobian.

        """

        num_free_variables = n * N + q
        num_constraints = n * (N-1)

        self.obj = obj
        self.obj_grad = obj_grad
        self.con = con
        self.con_jac = con_jac

        # TODO : 2 * n + q is likely only valid if there are no free input
        # trajectories. I think it is 2 * n + q + r.
        self.con_jac_rows, self.con_jac_cols = con_jac_indices()

        con_bounds = np.zeros(num_constraints)

        super(Problem, self).__init__(n=num_free_variables,
                                      m=num_constraints,
                                      cl=con_bounds,
                                      cu=con_bounds)

        self.output_filename = 'ipopt_output.txt'
        #self.addOption('derivative_test', 'first-order')
        self.addOption('output_file', self.output_filename)
        self.addOption('print_timing_statistics', 'yes')
        self.addOption('linear_solver', 'ma57')

        self.obj_value = []

    def objective(self, free):
        return self.obj(free)

    def gradient(self, free):
        # This should return a column vector.
        return self.obj_grad(free)

    def constraints(self, free):
        # This should return a column vector.
        return self.con(free)

    def jacobianstructure(self):
        return (self.con_jac_rows, self.con_jac_cols)

    def jacobian(self, free):
        return self.con_jac(free)

    def intermediate(self, *args):
        self.obj_value.append(args[2])


def objective_function(free, num_dis_points, num_states, dis_period,
                       time_measured, y_measured):
    """Returns the norm of the difference in the measured and simulated
    output.

    Parameters
    ----------
    free : ndarray, shape(n * N + q,)
        The flattened state array with n states at N time points and the q
        free model constants.
    num_dis_points : integer
        The number of model discretization points.
    num_states : integer
        The number of system states.
    dis_period : float
        The discretization time interval.
    y_measured : ndarray, shape(M, o)
        The measured trajectories of the o output variables at each sampled
        time instance.

    Returns
    -------
    cost : float
        The cost value.

    Notes
    -----
    This assumes that the states are ordered:

    [coord1, ..., coordn, speed1, ..., speedn]

    y_measured is interpolated at the discretization time points and
    compared to the model output at the discretization time points.

    """
    M, o = y_measured.shape
    N, n = num_dis_points, num_states

    sample_rate = 1.0 / dis_period
    duration = (N - 1) / sample_rate

    model_time = np.linspace(0.0, duration, num=N)

    states, specified, constants = parse_free(free, n, 0, N)

    model_state_trajectory = states.T  # states is shape(n, N) so transpose
    model_outputs = output_equations(model_state_trajectory)

    func = interp1d(time_measured, y_measured, axis=0)

    return dis_period * np.sum((func(model_time).flatten() -
                                model_outputs.flatten())**2)


def objective_function_gradient(free, num_dis_points, num_states,
                                dis_period, time_measured, y_measured):
    """Returns the gradient of the objective function with respect to the
    free parameters.

    Parameters
    ----------
    free : ndarray, shape(N * n + q,)
        The flattened state array with n states at N time points and the q
        free model constants.
    num_dis_points : integer
        The number of model discretization points.
    num_states : integer
        The number of system states.
    dis_period : float
        The discretization time interval.
    y_measured : ndarray, shape(M, o)
        The measured trajectories of the o output variables at each sampled
        time instance.

    Returns
    -------
    gradient : ndarray, shape(N * n + q,)
        The gradient of the cost function with respect to the free
        parameters.

    Warning
    -------
    This is currently only valid if the model outputs (and measurements) are
    simply the states. The chain rule will be needed if the function
    output_equations() is more than a simple selection.

    """

    M, o = y_measured.shape
    N, n = num_dis_points, num_states

    sample_rate = 1.0 / dis_period
    duration = (N - 1) / sample_rate

    model_time = np.linspace(0.0, duration, num=N)

    states, specified, constants = parse_free(free, n, 0, N)

    model_state_trajectory = states.T  # states is shape(n, N)

    # coordinates
    model_outputs = output_equations(model_state_trajectory)  # shape(N, o)

    func = interp1d(time_measured, y_measured, axis=0)

    dobj_dfree = np.zeros_like(free)
    # Set the derivatives with respect to the coordinates, all else are
    # zero.
    # 2 * (xi - xim)
    dobj_dfree[:N * o] = 2.0 * dis_period * (model_outputs -
                                             func(model_time)).T.flatten()

    return dobj_dfree


def wrap_objective(obj_func, *args):
    def wrapped_func(free):
        return obj_func(free, *args)
    return wrapped_func


class ConstraintCollocator():
    """This class is responsible for generating the constraint function and
    the sparse Jacobian of the constraint function using direct collocation
    methods for a non-linear programming problem where the essential
    constraints are defined from the equations of motion of the system."""

    time_interval_symbol = sym.Symbol('h')

    def __init__(self, equations_of_motion, state_symbols,
                 num_collocation_nodes, node_time_interval,
                 known_parameter_map={}, known_trajectory_map={},
                 time_symbol='t', tmp_dir=None):
        """
        Parameters
        ----------
        equations_of_motion : sympy.Matrix, shape(n, 1)
            A column matrix of SymPy expressions defining the right hand
            side of the equations of motion when the left hand side is zero,
            e.g. 0 = x'(t) - f(x(t), u(t), p) or 0 = f(x'(t), x(t), u(t),
            p). These should be in first order form.
        state_symbols : iterable
            An iterable containing all of the SymPy functions of time which
            represent the states in the equations of motion.
        num_collocation_nodes : integer
            The number of collocation nodes, N. All known trajectory arrays
            should be of this length.
        node_time_interval : float
            The time interval between collocation nodes.
        known_parameter_map : dictionary, optional
            A dictionary that maps the SymPy symbols representing the known
            constant parameters to floats. Any parameters in the equations
            of motion not provided in this dictionary will become free
            optimization variables.
        known_trajectory_map : dictionary, optional
            A dictionary that maps the non-state SymPy functions of time to
            ndarrays of floats of shape(N,). Any time varying parameters in
            the equations of motion not provided in this dictionary will
            become free trajectories optimization variables.
        time_symbol : string, optional
            The string representation of the SymPy Symbol which represents
            time in the equations of motion.
        tmp_dir : string, optional
            If you want to see the generated Cython and C code for the
            constraint and constraint Jacobian evaluations, pass in a path
            to a directory here.

        """
        self.eom = equations_of_motion

        self.time_symbol = sym.Symbol(time_symbol)
        me.dynamicsymbols._t = self.time_symbol

        self.state_symbols = tuple(state_symbols)
        self.state_derivative_symbols = tuple([s.diff(self.time_symbol) for
                                               s in state_symbols])
        self.num_states = len(self.state_symbols)

        self.num_collocation_nodes = num_collocation_nodes
        self.node_time_interval = node_time_interval

        self.known_parameter_map = known_parameter_map
        self.known_trajectory_map = known_trajectory_map

        self.tmp_dir = tmp_dir

        self._sort_parameters()
        self._check_known_trajectories()
        self._sort_trajectories()
        self._discrete_symbols()
        self._discretize_eom()

    @staticmethod
    def _parse_inputs(all_syms, known_syms):
        """Returns sets of symbols and their counts, based on if the known
        symbols exist in the set of all symbols.

        Parameters
        ----------
        all_syms : sequence
            A set of SymPy symbols or functions.
        known_syms : sequence
            A set of SymPy symbols or functions.

        Returns
        -------
        known : tuple
            The set of known symbols.
        num_known : integer
            The number of known symbols.
        unknown : tuple
            The set of unknown symbols in all_syms.
        num_unknown :integer
            The number of unknown symbols.

        """
        all_syms = set(all_syms)
        known_syms = known_syms

        def sort_sympy(seq):
            seq = list(seq)
            try:  # symbols
                seq.sort(key=lambda x: x.name)
            except AttributeError:  # functions
                seq.sort(key=lambda x: x.__class__.__name__)
            return seq

        if not all_syms:  # if empty sequence
            if known_syms:
                msg = '{} are not in the provided equations of motion.'
                raise ValueError(msg.format(known_syms))
            else:
                known = tuple()
                num_known = 0
                unknown = tuple()
                num_unknown = 0
        else:
            if known_syms:
                known = tuple(known_syms)  # don't sort known syms
                num_known = len(known)
                unknown = tuple(sort_sympy(all_syms.difference(known)))
                num_unknown = len(unknown)
            else:
                known = tuple()
                num_known = 0
                unknown = tuple(sort_sympy(all_syms))
                num_unknown = len(unknown)

        return known, num_known, unknown, num_unknown

    def _sort_parameters(self):
        """Finds and counts all of the parameters in the equations of motion
        and categorizes them based on which parameters the user supplies.
        The unknown parameters are sorted by name."""

        parameters = self.eom.free_symbols.copy()
        parameters.remove(self.time_symbol)

        res = self._parse_inputs(parameters,
                                 self.known_parameter_map.keys())

        self.known_parameters = res[0]
        self.num_known_parameters = res[1]
        self.unknown_parameters = res[2]
        self.num_unknown_parameters = res[3]

        self.parameters = res[0] + res[2]
        self.num_parameters = len(self.parameters)

    def _check_known_trajectories(self):
        """Raises and error if the known tracjectories are not the correct
        length."""

        N = self.num_collocation_nodes

        for k, v in self.known_trajectory_map.items():
            if len(v) != N:
                msg = 'The known parameter {} is not length {}'
                raise ValueError(msg.format(k, N))

    def _sort_trajectories(self):
        """Finds and counts all of the non-state, time varying parameters in
        the equations of motion and categorizes them based on which
        parameters the user supplies. The unknown parameters are sorted by
        name."""

        states = set(self.state_symbols)
        states_derivatives = set(self.state_derivative_symbols)

        time_varying_symbols = me.find_dynamicsymbols(self.eom)
        state_related = states.union(states_derivatives)
        non_states = time_varying_symbols.difference(state_related)

        res = self._parse_inputs(non_states,
                                 self.known_trajectory_map.keys())

        self.known_input_trajectories = res[0]
        self.num_known_input_trajectories = res[1]
        self.unknown_input_trajectories = res[2]
        self.num_unknown_input_trajectories = res[3]

        self.input_trajectories = res[0] + res[2]
        self.num_input_trajectories = len(self.input_trajectories)

    def _discrete_symbols(self):
        """Instantiates discrete symbols for each time varying variable in
        the euqations of motion.

        Instantiates
        ------------
        current_discrete_state_symbols : tuple of sympy.Symbols
            The n symbols representing the system's ith states.
        previous_discrete_state_symbols : tuple of sympy.Symbols
            The n symbols representing the system's (ith - 1) states.
        current_discrete_specified_symbols : tuple of sympy.Symbols
            The m symbols representing the system's ith specified inputs.

        """

        specified = (self.known_input_trajectories +
                     self.unknown_input_trajectories)

        xi = [sym.Symbol(f.__class__.__name__ + 'i')
              for f in self.state_symbols]
        xp = [sym.Symbol(f.__class__.__name__ + 'p')
              for f in self.state_symbols]
        si = [sym.Symbol(f.__class__.__name__ + 'i') for f in specified]

        self.current_discrete_state_symbols = tuple(xi)
        self.previous_discrete_state_symbols = tuple(xp)
        self.current_discrete_specified_symbols = tuple(si)

    def _discretize_eom(self):
        """Instantiates the constraint equations in a discretized form using
        backward Euler discretization.

        Instantiates
        ------------
        discrete_eoms : sympy.Matrix, shape(n, 1)
            The column vector of the discretized equations of motion.

        """
        x = self.state_symbols
        xd = self.state_derivative_symbols
        u = self.input_trajectories

        xi = self.current_discrete_state_symbols
        xp = self.previous_discrete_state_symbols
        ui = self.current_discrete_specified_symbols

        h = self.time_interval_symbol

        deriv_sub = {d: (i - p) / h for d, i, p in zip(xd, xi, xp)}

        func_sub = dict(zip(x + u, xi + ui))

        self.discrete_eom = me.msubs(self.eom, deriv_sub, func_sub)

    def _gen_multi_arg_con_func(self):
        """Instantiates a function that evaluates the constraints given all
        of the arguments of the functions, i.e. not just the free
        optimization variables.

        Instantiates
        ------------
        _multi_arg_con_func : function
            A function which returns the numerical values of the constraints
            at collocation nodes 2,...,N.

        Notes
        -----
        args:
            all current states (x1i, ..., xni)
            all previous states (x1p, ... xnp)
            all current specifieds (s1i, ..., smi)
            parameters (c1, ..., cb)
            time interval (h)

            args: (x1i, ..., xni, x1p, ... xnp, s1i, ..., smi, c1, ..., cb, h)
            n: num states
            m: num specified
            b: num parameters

        The function should evaluate and return an array:

            [con_1_2, ..., con_1_N, con_2_2, ...,
             con_2_N, ..., con_n_2, ..., con_n_N]

        for n states and N-1 constraints at the time points.

        """
        xi_syms = self.current_discrete_state_symbols
        xp_syms = self.previous_discrete_state_symbols
        si_syms = self.current_discrete_specified_symbols
        h_sym = self.time_interval_symbol
        constant_syms = self.known_parameters + self.unknown_parameters

        args = [x for x in xi_syms] + [x for x in xp_syms]
        args += [s for s in si_syms] + list(constant_syms) + [h_sym]

        f = ufuncify_matrix(args, self.discrete_eom,
                            const=constant_syms + (h_sym,),
                            tmp_dir=self.tmp_dir)

        def constraints(state_values, specified_values, constant_values,
                        interval_value):
            """Returns a vector of constraint values given all of the
            unknowns in the equations of motion over the 2, ..., N time
            steps.

            Parameters
            ----------
            states : ndarray, shape(n, N)
                The array of n states through N time steps.
            specified_values : ndarray, shape(m, N) or shape(N,)
                The array of m specifieds through N time steps.
            constant_values : ndarray, shape(b,)
                The array of b parameters.
            interval_value : float
                The value of the discretization time interval.

            Returns
            -------
            constraints : ndarray, shape(N-1,)
                The array of constraints from t = 2, ..., N.
                [con_1_2, ..., con_1_N, con_2_2, ...,
                 con_2_N, ..., con_n_2, ..., con_n_N]

            """

            if state_values.shape[0] < 2:
                raise ValueError('There should always be at least two states.')

            assert state_values.shape == (self.num_states,
                                          self.num_collocation_nodes)

            x_current = state_values[:, 1:]  # n x N - 1
            x_previous = state_values[:, :-1]  # n x N - 1

            # 2n x N - 1
            args = [x for x in x_current] + [x for x in x_previous]

            # 2n + m x N - 1
            if len(specified_values.shape) == 2:
                assert specified_values.shape == \
                    (self.num_input_trajectories,
                     self.num_collocation_nodes)
                si = specified_values[:, 1:]
                args += [s for s in si]
            else:
                assert specified_values.shape == \
                    (self.num_collocation_nodes,)
                si = specified_values[1:]
                args += [si]

            args += [c for c in constant_values]
            args += [interval_value]

            num_constraints = state_values.shape[1] - 1

            # TODO : Move this to an attribute of the class so that it is
            # only initialized once and just reuse it on each evaluation of
            # this function.
            result = np.empty((num_constraints, state_values.shape[0]))

            return f(result, *args).T.flatten()

        self._multi_arg_con_func = constraints

    def jacobian_indices(self):
        """Returns the row and column indices for the non-zero values in the
        constraint Jacobian.

        Returns
        -------
        jac_row_idxs : ndarray, shape(2 * n + q + r,)
            The row indices for the non-zero values in the Jacobian.
        jac_col_idxs : ndarray, shape(n,)
            The column indices for the non-zero values in the Jacobian.

        """

        num_constraint_nodes = self.num_collocation_nodes - 1
        # TODO : Change to the following to support free specified.
        # num_states * (2 * num_states + num_free_constants +
        # num_free_specified)
        num_partials = self.num_states * (2 * self.num_states +
                                          self.num_unknown_parameters)
        num_non_zero_values = num_partials * num_constraint_nodes

        jac_row_idxs = np.empty(num_non_zero_values, dtype=int)
        jac_col_idxs = np.empty(num_non_zero_values, dtype=int)

        for i in range(num_constraint_nodes):
            # n: num_states
            # m: num_specified
            # p: num_free_constants

            # the states repeat every N - 1 constraints
            # row_idxs = [0 * (N - 1), 1 * (N - 1),  2 * (N - 1),  n * (N - 1)]

            row_idxs = [j * (num_constraint_nodes) + i
                        for j in range(self.num_states)]

            # The derivative columns are in this order:
            # [x1i, x2i, ..., xni, x1p, x2p, ..., xnp, p1, ..., pp]
            # So we need to map them to the correct column.

            # first row, the columns indices mapping is:
            # [1, N + 1, ..., N - 1] : [x1p, x1i, 0, ..., 0]
            # [0, N, ..., 2 * (N - 1)] : [x2p, x2i, 0, ..., 0]
            # [-p:] : p1,..., pp  the free constants

            # i=0: [1, ..., n * N + 1, 0, ..., n * N + 0, n * N:n * N + p]
            # i=1: [2, ..., n * N + 2, 1, ..., n * N + 1, n * N:n * N + p]
            # i=2: [3, ..., n * N + 3, 2, ..., n * N + 2, n * N:n * N + p]

            col_idxs = [j * self.num_collocation_nodes + i + 1
                        for j in range(self.num_states)]
            col_idxs += [j * self.num_collocation_nodes + i
                         for j in range(self.num_states)]
            col_idxs += [self.num_states * self.num_collocation_nodes + j
                         for j in range(self.num_unknown_parameters)]

            row_idx_permutations = np.repeat(row_idxs, len(col_idxs))
            col_idx_permutations = np.array(list(col_idxs) * len(row_idxs),
                                            dtype=int)

            start = i * num_partials
            stop = (i + 1) * num_partials
            jac_row_idxs[start:stop] = row_idx_permutations
            jac_col_idxs[start:stop] = col_idx_permutations

        return jac_row_idxs, jac_col_idxs

    def _gen_multi_arg_con_jac_func(self):
        """Instantiates a function that evaluates the Jacobian of the
        constraints.

        Instantiates
        ------------
        _multi_arg_con_jac_func : function
            A function which returns the numerical values of the constraints
            at time points 2,...,N.

        """
        xi_syms = self.current_discrete_state_symbols
        xp_syms = self.previous_discrete_state_symbols
        si_syms = self.current_discrete_specified_symbols
        h_sym = self.time_interval_symbol
        constant_syms = self.known_parameters + self.unknown_parameters

        # The free parameters are always the n * (N - 1) state values and
        # the user's specified unknown model constants, so the base Jacobian
        # needs to be taken with respect to the ith, and ith - 1 states, and
        # the free model constants.
        # TODO : This needs to eventually support unknown specified inputs
        # too.
        partials = xi_syms + xp_syms + self.unknown_parameters

        # The arguments to the Jacobian function include all of the free
        # Symbols/Functions in the matrix expression.
        args = xi_syms + xp_syms + si_syms + constant_syms + (h_sym,)

        symbolic_jacobian = self.discrete_eom.jacobian(partials)

        jac = ufuncify_matrix(args, symbolic_jacobian,
                              const=constant_syms + (h_sym,),
                              tmp_dir=self.tmp_dir)

        # jac is now a function that takes arguments that are made up of all
        # the variables in the discretized equations of motion. It will be
        # used to build the sparse constraint gradient matrix. This Jacobian
        # function returns the non-zero elements needed to build the sparse
        # constraint gradient.

        def constraints_jacobian(state_values, specified_values,
                                 constant_values, interval_value):
            """Returns a sparse matrix of constraint gradient given all of
            the unknowns in the equations of motion over the 2, ..., N time
            steps.

            Parameters
            ----------
            states : ndarray, shape(n, N)
                The array of n states through N time steps.
            specified_values : ndarray, shape(m, N) or shape(N,)
                The array of m specified inputs through N time steps.
            constant_values : ndarray, shape(p,)
                The array of p constants.
            interval_value : float
                The value of the dicretization time interval.

            Returns
            -------
            constraints_gradient : ndarray,
                The values of the non-zero entries to the constraints
                Jacobian.  These correspond to the triplet formatted indices
                returned from jacobian_indices.

            """
            if state_values.shape[0] < 2:
                raise ValueError('There should always be at least two states.')

            x_current = state_values[:, 1:]  # n x N - 1
            x_previous = state_values[:, :-1]  # n x N - 1

            num_time_steps = state_values.shape[1]  # N
            num_constraint_nodes = num_time_steps - 1  # N - 1

            # 2n x N - 1
            args = [x for x in x_current] + [x for x in x_previous]

            # 2n + m x N - 1
            if len(specified_values.shape) == 2:
                args += [s for s in specified_values[:, 1:]]
            else:
                args += [specified_values[1:]]

            args += [c for c in constant_values]
            args += [interval_value]

            result = np.empty((num_constraint_nodes,
                               symbolic_jacobian.shape[0] *
                               symbolic_jacobian.shape[1]))

            # shape(N - 1, n, 2*n+p) where p is len(free_constants)
            non_zero_derivatives = jac(result, *args)

            # Now loop through the N - 1 constraint nodes to compute the
            # non-zero entries to the gradient matrix (the partials for n
            # states will be computed at each iteration).
            num_partials = (non_zero_derivatives.shape[1] *
                            non_zero_derivatives.shape[2])

            # TODO : Move this to a class attribute so it is only created
            # once and reused.
            jac_vals = np.empty(num_partials * num_constraint_nodes,
                                dtype=float)

            # TODO : The ordered Jacobian values may be able to be gotten by
            # simply flattening non_zero_derivatives. And if not, then maybe
            # this loop needs to be in Cython.

            for i in range(num_constraint_nodes):
                start = i * num_partials
                stop = (i + 1) * num_partials
                # TODO : This flatten() call is currently the most time
                # consuming thing in this function at this point.
                jac_vals[start:stop] = non_zero_derivatives[i].flatten()

            return jac_vals

        self._multi_arg_con_jac_func = constraints_jacobian

    def _wrap_constraint_funcs(self, func):
        """Returns a function that evaluates all of the constraints or
        Jacobian of the constraints given the free optimization variables.

        Parameters
        ----------
        func : function
            A function that takes the full parameter set and evaluates the
            constraint functions or the Jacobian of the contraint functions.
            i.e. the output of _gen_multi_arg_con_func or
            _gen_multi_arg_con_jac_func.

        Returns
        -------
        func : function
            A function which returns constraint values given the system's
            free optimization variables.

        """

        def constraints(free):

            free_states, free_specified, free_constants = \
                parse_free(free, self.num_states,
                           self.num_unknown_input_trajectories,
                           self.num_collocation_nodes)

            all_specified = self._merge_fixed_free(self.input_trajectories,
                                                   self.known_trajectory_map,
                                                   free_specified)

            all_constants = self._merge_fixed_free(self.parameters,
                                                   self.known_parameter_map,
                                                   free_constants)

            return func(free_states, all_specified, all_constants,
                        self.node_time_interval)

        intro, second = func.__doc__.split('Parameters')
        params, returns = second.split('Returns')
        new_doc = '{}Parameters\n----------\nfree : ndarray, shape()\n\nReturns\n{}'
        constraints.__doc__ = new_doc.format(intro, returns)

        return constraints

    def generate_constraint_function(self):
        """Returns a function which evaluates the constraints given the
        array of free optimization variables."""
        self._gen_multi_arg_con_func()
        return self._wrap_constraint_funcs(self._multi_arg_con_func)

    def generate_jacobian_function(self):
        """Returns a function which evaluates the Jacobian of the
        constraints given the array of free optimization variables."""
        self._gen_multi_arg_con_jac_func()
        return self._wrap_constraint_funcs(self._multi_arg_con_jac_func)

    @staticmethod
    def _merge_fixed_free(syms, fixed, free):
        """Returns an array with the fixed and free values combined. This
        just takes the known and unknown values and combines them for the
        function evaluation.

        This assumes that you have the free constants in the correct order.

        """

        merged = []
        n = 0
        for i, s in enumerate(syms):
            if s in fixed.keys():
                merged.append(fixed[s])
            else:
                merged.append(free[n])
                n += 1
        return np.array(merged)
