#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
import sympy as sym
from sympy.physics import mechanics as me
import ipopt

from simulate import output_equations
from utils import ufuncify_matrix, parse_free


class Problem(ipopt.problem):

    def __init__(self, obj, obj_grad, *args, **kwargs):
        """

        Parameters
        ----------
        obj : function
            Returns the value of the objective function.
        obj_grad : function
            Returns the gradient of the objective function.

        """

        self.state_bounds = kwargs.pop('state_bounds')
        self.unknown_trajectory_bounds = kwargs.pop('unknown_trajectory_bounds')
        self.unknown_parameter_bounds = kwargs.pop('unknown_parameter_bounds')

        self.collocator = ConstraintCollocator(*args, **kwargs)

        self.obj = obj
        self.obj_grad = obj_grad
        self.con = self.collocator.generate_constraint_function()
        self.con_jac = self.collocator.generate_jacobian_function()

        self.con_jac_rows, self.con_jac_cols = \
            self.collocator.jacobian_indices()

        self.num_free = self.collocator.num_free
        self.num_constraints = self.collocator.num_constraints

        self._generate_bound_arrays()

        # All constraints are expected to be equalt to zero.
        con_bounds = np.zeros(self.num_constraints)

        super(Problem, self).__init__(n=self.num_free,
                                      m=self.num_constraints,
                                      lb=self.lower_bound,
                                      ub=self.upper_bound,
                                      cl=con_bounds,
                                      cu=con_bounds)

        self.output_filename = 'ipopt_output.txt'
        #self.addOption('derivative_test', 'first-order')
        self.addOption('output_file', self.output_filename)
        self.addOption('print_timing_statistics', 'yes')
        self.addOption('linear_solver', 'ma57')

        self.obj_value = []

    def _generate_bound_arrays(self):
        INF = 10e19
        lb = -INF * np.ones(self.num_free)
        ub = INF * np.ones(self.num_free)

        N = self.collocator.num_collocation_nodes

        if self.state_bounds is not None:
            state_syms = self.collocator.state_symbols

            for state, bounds in self.state_bounds.items():
                i = state_syms.index(state)
                start = i * N
                stop = start + N
                lb[start:stop] = bounds[0] * np.ones(N)
                ub[start:stop] = bounds[1] * np.ones(N)

        if self.unknown_trajectory_bounds is not None:
            unk_traj = self.collocator.unknown_input_trajectories
            num_state_nodes = N * self.collocator.num_states
            for traj, bounds in self.unknown_trajectory_bounds.items():
                i = unk_traj.index(traj)
                start = num_state_nodes + i * N
                stop = start + N
                lb[start:stop] = bounds[0] * np.ones(N)
                ub[start:stop] = bounds[1] * np.ones(N)

        if self.unknown_parameter_bounds is not None:
            unk_par = self.collocator.unknown_parameters
            num_non_par_nodes = N * (self.collocator.num_states +
                                     self.collocator.num_unknown_input_trajectories)
            for par, bounds in self.unknown_parameter_bounds.items():
                i = unk_par.index(par)
                idx = num_non_par_nodes + i
                lb[idx] = bounds[0]
                ub[idx] = bounds[1]

        self.lower_bound = lb
        self.upper_bound = ub

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
                 instance_constraints=None, time_symbol='t', tmp_dir=None):
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
        instance_constraints : iterable of SymPy expressions
            These expressions are for constraints on the states at specific
            time points. They can be expressions with any state instance and
            any of the known parameters found in the equations of motion.
            All states should be evaluated at a specific instant of time.
            For example, the constraint x(0) = 5.0 would be specified as
            x(0) - 5.0 and the constraint x(0) = x(5.0) would be specified
            as  x(0) - x(5.0). Unknown parameters and time varying
            parameters other than the states are currently not supported.
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

        self.instance_constraints = instance_constraints

        self.num_constraints = self.num_states * (num_collocation_nodes - 1)

        self.tmp_dir = tmp_dir

        self._sort_parameters()
        self._check_known_trajectories()
        self._sort_trajectories()
        self.num_free = ((self.num_states +
                          self.num_unknown_input_trajectories) *
                         self.num_collocation_nodes +
                         self.num_unknown_parameters)
        self._discrete_symbols()
        self._discretize_eom()

        if instance_constraints is not None:
            self.num_instance_constraints = len(instance_constraints)
            self.num_constraints += self.num_instance_constraints
            self._identify_functions_in_instance_constraints()
            self._find_closest_free_index()
            self.eval_instance_constraints = \
                self._instance_constraints_func()
            self.eval_instance_constraints_jacobian_values = \
                self._instance_constraints_jacobian_values_func()

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

        xi = [sym.Symbol(f.__class__.__name__ + 'i')
              for f in self.state_symbols]
        xp = [sym.Symbol(f.__class__.__name__ + 'p')
              for f in self.state_symbols]
        ki = [sym.Symbol(f.__class__.__name__ + 'i') for f in
              self.known_input_trajectories]
        ui = [sym.Symbol(f.__class__.__name__ + 'i') for f in
              self.unknown_input_trajectories]

        self.current_discrete_state_symbols = tuple(xi)
        self.previous_discrete_state_symbols = tuple(xp)
        self.current_known_discrete_specified_symbols = tuple(ki)
        self.current_unknown_discrete_specified_symbols = tuple(ui)
        self.current_discrete_specified_symbols = tuple(ki) + tuple(ui)

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

    def _identify_functions_in_instance_constraints(self):
        """Instantiates a set containing all of the instance functions, i.e.
        x(1.0) in the instance constraints."""

        all_funcs = set()

        for con in self.instance_constraints:
            all_funcs = all_funcs.union(con.atoms(sym.Function))

        self.instance_constraint_function_atoms = all_funcs

    def _find_closest_free_index(self):
        """Instantiates a dictionary mapping the instance functions to the
        nearest index in the free variables vector."""

        def determine_free_index(time_index, state):
            state_index = self.state_symbols.index(state)
            return time_index + state_index * self.num_collocation_nodes

        N = self.num_collocation_nodes
        h = self.node_time_interval
        duration = h * (N - 1)

        time_vector = np.linspace(0.0, duration, num=N)

        node_map = {}
        for func in self.instance_constraint_function_atoms:
            time_value = func.args[0]
            time_index = np.argmin(np.abs(time_vector - time_value))
            free_index = determine_free_index(time_index,
                                              func.__class__(self.time_symbol))
            node_map[func] = free_index

        self.instance_constraints_free_index_map = node_map

    def _instance_constraints_func(self):
        """Returns a function that evaluates the instance constraints given
        the free optimization variables."""
        free = sym.DeferredVector('FREE')
        def_map = {k: free[v] for k, v in
                   self.instance_constraints_free_index_map.items()}
        subbed_constraints = [con.subs(def_map) for con in
                              self.instance_constraints]
        f = sym.lambdify(([free] + self.known_parameter_map.keys()),
                         subbed_constraints, default_array=True)

        def wrapped(free):
            return f(free, *self.known_parameter_map.values())

        return wrapped

    def _instance_constraints_jacobian_indices(self):
        """Returns the row and column indices of the non-zero values in the
        Jacobian of the constraints."""
        idx_map = self.instance_constraints_free_index_map

        num_eom_constraints = 2 * (self.num_collocation_nodes - 1)

        rows = []
        cols = []

        for i, con in enumerate(self.instance_constraints):
            funcs = con.atoms(sym.Function)
            indices = [idx_map[f] for f in funcs]
            row_idxs = num_eom_constraints + i * np.ones(len(indices),
                                                         dtype=int)
            rows += list(row_idxs)
            cols += indices

        return np.array(rows), np.array(cols)

    def _instance_constraints_jacobian_values_func(self):
        """Retruns the non-zero values of the constraint Jacobian associated
        with the instance constraints."""
        free = sym.DeferredVector('FREE')

        def_map = {k: free[v] for k, v in
                   self.instance_constraints_free_index_map.items()}

        funcs = []
        num_vals_per_func = []
        for con in self.instance_constraints:
            partials = list(con.atoms(sym.Function))
            num_vals_per_func.append(len(partials))
            jac = sym.Matrix([con]).jacobian(partials)
            jac = jac.subs(def_map)
            funcs.append(sym.lambdify(([free] +
                                       self.known_parameter_map.keys()),
                                      jac, default_array=True))
        l = np.sum(num_vals_per_func)

        def wrapped(free):
            arr = np.zeros(l)
            j = 0
            for i, (f, num) in enumerate(zip(funcs, num_vals_per_func)):
                arr[j:j + num] = f(free, *self.known_parameter_map.values())
                j += num
            return arr

        return wrapped

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
            elif len(specified_values.shape) == 1 and specified_values.size != 0:
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

        N = self.num_collocation_nodes
        n = self.num_states

        num_constraint_nodes = N - 1
        num_partials = n * (2 * n + self.num_unknown_input_trajectories +
                            self.num_unknown_parameters)
        num_non_zero_values = num_partials * num_constraint_nodes

        if self.instance_constraints is not None:
            ins_row_idxs, ins_col_idxs = \
                self._instance_constraints_jacobian_indices()
            num_non_zero_values += len(ins_row_idxs)

        jac_row_idxs = np.empty(num_non_zero_values, dtype=int)
        jac_col_idxs = np.empty(num_non_zero_values, dtype=int)

        for i in range(num_constraint_nodes):
            # n: num_states
            # m: num_specified
            # p: num_free_constants

            # the states repeat every N - 1 constraints
            # row_idxs = [0 * (N - 1), 1 * (N - 1),  2 * (N - 1),  n * (N - 1)]

            row_idxs = [j * (num_constraint_nodes) + i
                        for j in range(n)]

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

            col_idxs = [j * N + i + 1 for j in range(n)]
            col_idxs += [j * N + i for j in range(n)]
            col_idxs += [j * N + i + n * N + 1 for j in
                         range(self.num_unknown_input_trajectories)]
            col_idxs += [(n + self.num_unknown_input_trajectories) * N + j
                         for j in range(self.num_unknown_parameters)]

            row_idx_permutations = np.repeat(row_idxs, len(col_idxs))
            col_idx_permutations = np.array(list(col_idxs) * len(row_idxs),
                                            dtype=int)

            start = i * num_partials
            stop = (i + 1) * num_partials
            jac_row_idxs[start:stop] = row_idx_permutations
            jac_col_idxs[start:stop] = col_idx_permutations

        if self.instance_constraints is not None:
            jac_row_idxs[-len(ins_row_idxs):] = ins_row_idxs
            jac_col_idxs[-len(ins_col_idxs):] = ins_col_idxs

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
        ki_syms = self.current_known_discrete_specified_symbols
        ui_syms = self.current_unknown_discrete_specified_symbols
        h_sym = self.time_interval_symbol
        constant_syms = self.known_parameters + self.unknown_parameters

        # The free parameters are always the n * (N - 1) state values and
        # the user's specified unknown model constants, so the base Jacobian
        # needs to be taken with respect to the ith, and ith - 1 states, and
        # the free model constants.
        partials = (xi_syms + xp_syms + ui_syms + self.unknown_parameters)

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
            elif len(specified_values.shape) == 1 and specified_values.size != 0:
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

    @staticmethod
    def _merge_fixed_free(syms, fixed, free, typ):
        """Returns an array with the fixed and free values combined. This
        just takes the known and unknown values and combines them for the
        function evaluation.

        This assumes that you have the free constants in the correct order.

        Parameters
        ----------
        syms : iterable of SymPy Symbols or Functions
        fixed : dictionary
            A mapping from Symbols to floats or Functions to 1d ndarrays.
        free : ndarray, (N,) or shape(n,N)
            An array
        typ : string
            traj or par


        """

        merged = []
        n = 0
        # syms is order as known (fixed) then unknown (free)
        for i, s in enumerate(syms):
            if s in fixed.keys():
                merged.append(fixed[s])
            else:
                if typ == 'traj' and len(free.shape) == 1:
                    merged.append(free)
                else:
                    merged.append(free[n])
                    n += 1
        return np.array(merged)

    def _wrap_constraint_funcs(self, func, typ):
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
                                                   free_specified, 'traj')

            all_constants = self._merge_fixed_free(self.parameters,
                                                   self.known_parameter_map,
                                                   free_constants, 'par')

            eom_con_vals = func(free_states, all_specified, all_constants,
                                self.node_time_interval)

            if self.instance_constraints is not None:
                if typ == 'con':
                    ins_con_vals = self.eval_instance_constraints(free)
                elif typ == 'jac':
                    ins_con_vals = self.eval_instance_constraints_jacobian_values(free)
                return np.hstack((eom_con_vals, ins_con_vals))
            else:
                return eom_con_vals

        intro, second = func.__doc__.split('Parameters')
        params, returns = second.split('Returns')
        new_doc = '{}Parameters\n----------\nfree : ndarray, shape()\n\nReturns\n{}'
        constraints.__doc__ = new_doc.format(intro, returns)

        return constraints

    def generate_constraint_function(self):
        """Returns a function which evaluates the constraints given the
        array of free optimization variables."""
        self._gen_multi_arg_con_func()
        return self._wrap_constraint_funcs(self._multi_arg_con_func, 'con')

    def generate_jacobian_function(self):
        """Returns a function which evaluates the Jacobian of the
        constraints given the array of free optimization variables."""
        self._gen_multi_arg_con_jac_func()
        return self._wrap_constraint_funcs(self._multi_arg_con_jac_func, 'jac')

