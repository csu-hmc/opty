#!/usr/bin/env python

import sys
from functools import wraps
import logging

import numpy as np
import sympy as sm
from sympy.physics import mechanics as me
import cyipopt
try:
    plt = sm.external.import_module('matplotlib.pyplot',
                                    __import__kwargs={'fromlist': ['']},
                                    catch=(RuntimeError,))
except TypeError:  # SymPy >=1.6
    plt = sm.external.import_module('matplotlib.pyplot',
                                    import_kwargs={'fromlist': ['']},
                                    catch=(RuntimeError,))

from .utils import (ufuncify_matrix, parse_free, _optional_plt_dep,
                    _forward_jacobian, sort_sympy)

__all__ = ['Problem', 'ConstraintCollocator']


class _DocInherit(object):
    """
    Docstring inheriting method descriptor

    The class itself is also used as a decorator

    Taken from https://stackoverflow.com/questions/2025562/inherit-docstrings-in-python-class-inheritance

    This is the rather complex solution to using the super classes method
    docstring and modifying it.
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):

        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError("Can't find '%s' in parents" % self.name)
        func.__doc__ = self._combine_docs(self.mthd.__doc__,
                                          ConstraintCollocator.__init__.__doc__)
        return func

    @staticmethod
    def _combine_docs(prob_doc, coll_doc):
        beg, end = prob_doc.split('bounds')
        if sys.version_info[1] >= 13:
            sep = 'Parameters\n==========\n'
        else:
            sep = 'Parameters\n        ==========\n        '
        _, middle = coll_doc.split(sep)
        return beg + middle[:-9] + '        bounds' + end

_doc_inherit = _DocInherit


class Problem(cyipopt.Problem):
    """This class allows the user to instantiate a problem object with the
    essential data required to solve a direct collocation optimal control or
    parameter identification problem.

    This is a subclass of `cyipopt's Problem class
    <https://cyipopt.readthedocs.io/en/stable/reference.html#cyipopt.Problem>`_.

    Notes
    =====

    - N : number of collocation nodes
    - n : number of states
    - m : number of input trajectories
    - q : number of unknown input trajectories
    - r : number of unknown parameters
    - s : number of unknown time intervals (0 or 1 if fixed duration or
      variable duration)
    - o : number of instance constraints
    - nN + qN + r + s : number of free variables
    - n(N - 1) + o : number of constraints

    If ``x`` are the state variables, ``u`` are the unknown input trajectories,
    and ``p`` are the unknown parameters, and ``h`` is the unknown time
    interval then the free optimization variables are in this order::

       free = [x11, ... x1N,
               xn1, ... xnN,
               u11, ... u1N,
               uq1, ... xqN,
               p1, ... pr,
               h]

    If the equations of motion are equations ``eom1`` to ``eomn`` and  instance
    constraints are ``c``,  the constraint array is ordered as::

       constraints = [eom12, ... eom1N,
                      eomn2, ... eomnN,
                      c1, ..., co]

    The attributes may be accessed as follows: ``Problem_instance.collocator.name_of_attribute``

    """

    INF = 10e19

    @_doc_inherit
    def __init__(self, obj, obj_grad, equations_of_motion, state_symbols,
                 num_collocation_nodes, node_time_interval,
                 known_parameter_map={}, known_trajectory_map={},
                 instance_constraints=None, time_symbol=None, tmp_dir=None,
                 integration_method='backward euler', parallel=False,
                 bounds=None, show_compile_output=False):
        """

        Parameters
        ==========
        obj : function
            Returns the value of the objective function given the free vector.
        obj_grad : function
            Returns the gradient of the objective function given the free
            vector.
        bounds : dictionary, optional
            This dictionary should contain a mapping from any of the symbolic
            states, unknown trajectories, unknown parameters, or unknown time
            interval to a 2-tuple of floats, the first being the lower bound
            and the second the upper bound for that free variable, e.g.
            ``{x(t): (-1.0, 5.0)}``.

        """

        if equations_of_motion.has(sm.Derivative) == False:
            raise ValueError('No time derivatives are present.' +
                ' The equations of motion must be ordinary ' +
                'differential equations (ODEs) or ' +
                'differential algebraic equations (DAEs).')

        self.collocator = ConstraintCollocator(
            equations_of_motion, state_symbols, num_collocation_nodes,
            node_time_interval, known_parameter_map, known_trajectory_map,
            instance_constraints, time_symbol, tmp_dir, integration_method,
            parallel, show_compile_output=show_compile_output)

        self.bounds = bounds
        self.obj = obj
        self.obj_grad = obj_grad
        self.con = self.collocator.generate_constraint_function()
        logging.info('Constraint function generated.')
        self.con_jac = self.collocator.generate_jacobian_function()
        logging.info('Jacobian function generated.')

        self.con_jac_rows, self.con_jac_cols = \
            self.collocator.jacobian_indices()

        self.num_free = self.collocator.num_free
        self.num_constraints = self.collocator.num_constraints

        self._generate_bound_arrays()

        # All constraints are expected to be equal to zero.
        con_bounds = np.zeros(self.num_constraints)

        super(Problem, self).__init__(n=self.num_free,
                                      m=self.num_constraints,
                                      lb=self.lower_bound,
                                      ub=self.upper_bound,
                                      cl=con_bounds,
                                      cu=con_bounds)

        self.obj_value = []

    def _generate_bound_arrays(self):
        lb = -self.INF * np.ones(self.num_free)
        ub = self.INF * np.ones(self.num_free)

        N = self.collocator.num_collocation_nodes
        num_state_nodes = N*self.collocator.num_states
        num_non_par_nodes = N*(self.collocator.num_states +
                               self.collocator.num_unknown_input_trajectories)
        state_syms = self.collocator.state_symbols
        unk_traj = self.collocator.unknown_input_trajectories
        unk_par = self.collocator.unknown_parameters

        if self.bounds is not None:
            for var, bounds in self.bounds.items():
                if var in state_syms:
                    i = state_syms.index(var)
                    start = i * N
                    stop = start + N
                    lb[start:stop] = bounds[0] * np.ones(N)
                    ub[start:stop] = bounds[1] * np.ones(N)
                elif var in unk_traj:
                    i = unk_traj.index(var)
                    start = num_state_nodes + i * N
                    stop = start + N
                    lb[start:stop] = bounds[0] * np.ones(N)
                    ub[start:stop] = bounds[1] * np.ones(N)
                elif var in unk_par:
                    i = unk_par.index(var)
                    idx = num_non_par_nodes + i
                    lb[idx] = bounds[0]
                    ub[idx] = bounds[1]
                elif (self.collocator._variable_duration and
                      var == self.collocator.time_interval_symbol):
                    lb[-1] = bounds[0]
                    ub[-1] = bounds[1]
                else:
                    msg = 'Bound variable {} not present in free variables.'
                    raise ValueError(msg.format(var))

        self.lower_bound = lb
        self.upper_bound = ub

    def objective(self, free):
        """Returns the value of the objective function given a solution to the
        problem.

        Parameters
        ==========
        free : ndarray, shape(n*N + q*N + r + s, )
            A solution to the optimization problem in the canonical form.

        Returns
        =======
        obj_val : float
            The value of the objective function.

        Notes
        =====

        - N : number of collocation nodes
        - n : number of unknown state trajectories
        - q : number of unknown input trajectories
        - r : number of unknown parameters
        - s : number of unknown time intervals

        """
        return self.obj(free)

    def gradient(self, free):
        """Returns the value of the gradient of the objective function given a
        solution to the problem.

        Parameters
        ==========
        free : ndarray, (n*N + q*N + r + s, )
            A solution to the optimization problem in the canonical form.

        Returns
        =======
        gradient_val : ndarray, shape(n*N + q*N + r + s, 1)
            The value of the gradient of the objective function.

        Notes
        =====

        - N : number of collocation nodes
        - n : number of unknown state trajectories
        - q : number of unknown input trajectories
        - r : number of unknown parameters
        - s : number of unknown time intervals

        """
        # This should return a column vector.
        return self.obj_grad(free)

    def constraints(self, free):
        """Returns the value of the constraint functions given a solution to
        the problem.

        Parameters
        ==========
        free : ndarray, (n*N + q*N + r + s, )
            A solution to the optimization problem in the canonical form.

        Returns
        =======
        constraints_val : ndarray, shape(n*(N - 1) + o, )
            The value of the constraint function.

        Notes
        =====

        - N : number of collocation nodes
        - n : number of unknown state trajectories
        - q : number of unknown input trajectories
        - r : number of unknown parameters
        - s : number of unknown time intervals
        - o : number of instance constraints

        """
        # This should return a column vector.
        return self.con(free)

    def jacobianstructure(self):
        """Returns the sparsity structure of the Jacobian of the constraint
        function.

        Returns
        =======
        jac_row_idxs : ndarray, shape(2*n + q + r + s, )
            The row indices for the non-zero values in the Jacobian.
        jac_col_idxs : ndarray, shape(n*(N - 1) + o, )
            The column indices for the non-zero values in the Jacobian.

        """
        return (self.con_jac_rows, self.con_jac_cols)

    def jacobian(self, free):
        """Returns the non-zero values of the Jacobian of the constraint
        function.

        Returns
        =======
        jac_vals : ndarray, shape((2*n + q + r + s)*(n*(N - 1)) + o, )
            Non-zero Jacobian values in triplet format.

        """
        return self.con_jac(free)

    def intermediate(self, *args):
        """This method is called at every optimization iteration. Not for pubic
        use."""
        self.obj_value.append(args[2])

    @_optional_plt_dep
    def plot_trajectories(self, vector, axes=None):
        """Returns the axes for two plots. The first plot displays the state
        trajectories versus time and the second plot displays the input
        trajectories versus time.

        Parameters
        ==========
        vector : ndarray, (n*N + q*N + r + s, )
            The initial guess, solution, or any other vector that is in the
            canonical form.
        axes : ndarray of AxesSubplot, shape(n + m, )
            An array of matplotlib axes to plot to.

        Returns
        =======
        axes : ndarray of AxesSubplot
            A matplotlib axes with the state and input trajectories plotted.

        Notes
        =====

        - N : number of collocation nodes
        - n : number of unknown state trajectories
        - m : number of input trajectories
        - q : number of unknown input trajectories
        - r : number of unknown parameters
        - s : number of unknown time intervals

        """

        if self.collocator._variable_duration:
            state_traj, input_traj, constants, node_time_interval = parse_free(
                vector, self.collocator.num_states,
                self.collocator.num_unknown_input_trajectories,
                self.collocator.num_collocation_nodes,
                variable_duration=self.collocator._variable_duration)
        else:
            state_traj, input_traj, constants = parse_free(
                vector, self.collocator.num_states,
                self.collocator.num_unknown_input_trajectories,
                self.collocator.num_collocation_nodes,
                variable_duration=self.collocator._variable_duration)
            node_time_interval = self.collocator.node_time_interval

        time = np.linspace(0,
                           (self.collocator.num_collocation_nodes-1) *
                           node_time_interval,
                           num=self.collocator.num_collocation_nodes)

        num_axes = (self.collocator.num_states +
                    self.collocator.num_input_trajectories)
        traj_syms = (self.collocator.state_symbols +
                     self.collocator.known_input_trajectories +
                     self.collocator.unknown_input_trajectories)

        trajectories = state_traj

        if self.collocator.num_known_input_trajectories > 0:
            for knw_sym in self.collocator.known_input_trajectories:
                trajectories = np.vstack(
                    (trajectories,
                     self.collocator.known_trajectory_map[knw_sym]))

        if self.collocator.num_unknown_input_trajectories > 0:
            # NOTE : input_traj should be in the same order as
            # self.unknown_input_trajectories.
            trajectories = np.vstack((trajectories, input_traj))

        if axes is None:
            fig, axes = plt.subplots(num_axes, 1, sharex=True,
                                     layout='compressed',
                                     figsize=(6.4, 0.8*num_axes))

        for ax, traj, symbol in zip(axes, trajectories, traj_syms):
            ax.plot(time, traj)
            ax.set_ylabel(sm.latex(symbol, mode='inline'))
        ax.set_xlabel('Time')
        axes[0].set_title('State Trajectories')
        if self.collocator.num_unknown_input_trajectories > 0:
            axes[self.collocator.num_states].set_title('Input Trajectories')

        return axes

    @_optional_plt_dep
    def plot_constraint_violations(self, vector, axes=None):
        """Returns an axis with the state constraint violations plotted versus
        node number and the instance constraints as a bar graph.

        Parameters
        ==========
        vector : ndarray, (n*N + q*N + r + s, )
            The initial guess, solution, or any other vector that is in the
            canonical form.

        Returns
        =======
        axes : ndarray of AxesSubplot
            A matplotlib axes with the constraint violations plotted. If the
            uses gives at least two axis, the method will tell the user how
            many are needed, unless the correct amount is given.

        Notes
        =====

        - N : number of collocation nodes
        - n : number of unknown state trajectories
        - q : number of unknown input trajectories
        - r : number of unknown parameters
        - s : number of unknown time intervals

        """

        bars_per_plot = None
        rotation = -45

        # find the number of bars per plot, so the bars per plot are
        # aproximately the same on each plot
        hilfs = []
        len_constr = self.collocator.num_instance_constraints
        for i in range(6, 11):
            hilfs.append((i, i - len_constr % i))
            if len_constr % i == 0:
                bars_per_plot = i
                if len_constr == bars_per_plot:
                    num_plots = 1
                else:
                    num_plots = len_constr // bars_per_plot

        if bars_per_plot is None:
            maximal = 100
            for i in range(len(hilfs)):
                if hilfs[i][1] < maximal:
                    maximal = hilfs[i][1]
                    bars_per_plot = hilfs[i][0]
            if len_constr <= bars_per_plot:
                num_plots = 1
            else:
                num_plots = len_constr // bars_per_plot + 1

        # ensure that len(axes) is correct, raise ValuError otherwise
        if axes is not None:
            len_axes = len(axes.ravel())
            len_constr = self.collocator.num_instance_constraints
            if (len_constr <= bars_per_plot) and (len_axes < 2):
                raise ValueError('len(axes) must be equal to 2')

            elif ((len_constr % bars_per_plot == 0) and
                  (len_axes < len_constr // bars_per_plot + 1)):
                msg = (f'len(axes) must be equal to '
                       f'{len_constr//bars_per_plot+1}')
                raise ValueError(msg)

            elif ((len_constr % bars_per_plot != 0) and
                  (len_axes < len_constr // bars_per_plot + 2)):
                msg = (f'len(axes) must be equal to '
                       f'{len_constr//bars_per_plot+2}')
                raise ValueError(msg)

            else:
                pass

        N = self.collocator.num_collocation_nodes
        con_violations = self.con(vector)
        state_violations = con_violations[
            :(N - 1) * self.collocator.num_states]
        instance_violations = con_violations[len(state_violations):]
        state_violations = state_violations.reshape(
            (self.collocator.num_states, N - 1))
        con_nodes = range(1, self.collocator.num_collocation_nodes)

        if axes is None:
            fig, axes = plt.subplots(1 + num_plots, 1,
                                     figsize=(6.4, 1.50*(1 + num_plots)),
                                     layout='compressed')

        axes = np.asarray(axes).ravel()

        axes[0].plot(con_nodes, state_violations.T)
        axes[0].set_title('Constraint violations')
        axes[0].set_xlabel('Node Number')
        axes[0].set_ylabel('EoM violation')

        if self.collocator.instance_constraints is not None:
            # reduce the instance constrtaints to 2 digits after the decimal
            # point.  give the time in tha variables with 2 digits after the
            # decimal point.  if variable h is used, use the result for h in
            # the time.
            num_inst_viols = self.collocator.num_instance_constraints
            instance_constr_plot = []
            a_before = ''
            a_before_before = ''
            for exp1 in self.collocator.instance_constraints:
                for a in sm.preorder_traversal(exp1):
                    if ((isinstance(a_before, sm.Integer) or
                            isinstance(a_before, sm.Float)) and
                            (a == self.collocator.node_time_interval)):
                        a_before = float(a_before)
                        hilfs = a_before * vector[-1]
                        exp1 = exp1.subs(a_before_before,
                                         sm.Float(round(hilfs, 2)))
                    elif ((isinstance(a_before, sm.Float) and
                          (a != self.collocator.node_time_interval))):
                        exp1 = exp1.subs(a_before, round(a_before, 2))
                    a_before_before = a_before
                    a_before = a
                instance_constr_plot.append(exp1)

            for i in range(num_plots):
                num_ticks = bars_per_plot
                if i == num_plots - 1:
                    beginn = i * bars_per_plot
                    endd = num_inst_viols
                    num_ticks = num_inst_viols % bars_per_plot
                    if (num_inst_viols % bars_per_plot == 0):
                        num_ticks = bars_per_plot
                else:
                    endd = (i + 1) * bars_per_plot
                    beginn = i * bars_per_plot

                inst_viol = instance_violations[beginn: endd]
                inst_constr = instance_constr_plot[beginn: endd]

                width = [0.06*num_ticks for _ in range(num_ticks)]
                axes[i+1].bar(range(num_ticks), inst_viol,
                              tick_label=[sm.latex(s, mode='inline') for s in
                                          inst_constr], width=width)
                axes[i+1].set_ylabel('Instance')
                axes[i+1].set_xticklabels(axes[i+1].get_xticklabels(),
                                          rotation=rotation)

        return axes

    @_optional_plt_dep
    def plot_objective_value(self):
        """Returns an axis with the objective value plotted versus the
        optimization iteration. solve() must be run first."""

        fig, ax = plt.subplots(1, layout='compressed')
        ax.set_title('Objective Value')
        ax.plot(self.obj_value)
        ax.set_ylabel('Objective Value')
        ax.set_xlabel('Iteration Number')

        return ax

    def parse_free(self, free):
        """Parses the free parameters vector and returns it's components.

        Parameters
        ==========
        free : ndarray, shape(n*N + q*N + r + s)
            The free parameters of the system.

        Returns
        =======
        states : ndarray, shape(n, N)
            The array of n states through N time steps.
        specified_values : ndarray, shape(q, N) or shape(N,), or None
            The array of q specified inputs through N time steps.
        constant_values : ndarray, shape(r,)
            The array of r constants.
        time_interval : float
            The time between collocation nodes. Only returned if
            ``variable_duration`` is ``True``.

        Notes
        =====

        - N : number of collocation nodes
        - n : number of unknown state trajectories
        - q : number of unknown input trajectories
        - r : number of unknown parameters
        - s : number of unknown time intervals (s=1 if ``variable duration`` is
          ``True`` else s=0)

        """

        n = self.collocator.num_states
        N = self.collocator.num_collocation_nodes
        q = self.collocator.num_unknown_input_trajectories
        variable_duration = self.collocator._variable_duration

        return parse_free(free, n, q, N, variable_duration)

class ConstraintCollocator(object):
    """This class is responsible for generating the constraint function and the
    sparse Jacobian of the constraint function using direct collocation methods
    for a non-linear programming problem where the essential constraints are
    defined from the equations of motion of the system.

    Attributes
    ==========
    current_discrete_state_symbols : n-tuple
        The symbols for the current discrete states.
    current_discrete_specified_symbols : q-tuple
        The symbols for the current discrete specified inputs.
    discrete_eom : sympy.Matrix, shape(n, 1)
        Discretized equations of motion. Depending on the integration method
        used.
    eom: sympy.Matrix, shape(n, 1)
        The equations of motion used.
    input_trajectories : tuple
        known_input_trajectories + unknown_input_trajectories.
    instance_constraints : o-tuple
        The instance constraints used in the optimization.
    integration_method : str
        The integration method used.
    known_parameters : tuple
        The symbols of the known parameters in the problem.
    known_parameter_map : dict
        A mapping of known parameters to their values.
    known_trajectory_map : dict
        A mapping of known trajectories to their values.
    known_trajectory_symbols : (m-q)-tuple
        The known trajectory symbols.
    next_discrete_specified_symbols : q-tuple
        The symbols for the next discrete specified inputs.
    next_discrete_state_symbols : n-tuple
        The symbols for the next discrete states.
    node_time_interval : float or sympy.Symbol
        The time interval between the collocation nodes. float if the interval
        is fixed, ``sympy.Symbol`` if the interval is variable.
    num_collocation_nodes : int
        Number of times spaced evenly between the initial and final time of
        the optimization = N.
    num_constraints : int
        The number of constraints = (num_collection_nodes-1)*num_states +
        len(instance_constraints).
    num_free : int
        Number of variables to be optimized = n*N + q*N + r + s.
    num_input_trajectories : int
        The number of input trajectories = len(input_trajectories).
    num_instance_constraints : int
        The number of instance constraints = len(instance_constraints).
    num_known_trajectories : int
        The number of known trajectories = len(known_trajectory_symbols).
    num_parameters : int
        The number of parameters = len(parameters).
    num_states : int
        The number of states = len(state_symbols) = n.
    num_unknown_input_trajectories : int
        The number of unknown input trajectories =
        len(unknown_input_trajectories).
    num_unknown_parameters : int
        The number of unknown parameters = r.
    parameters : tuple
        known_parameters + unknown_parameters.
    parallel : bool
        Whether to use parallel processing or not.
    previous_discrete_state_symbols : n-tuple
        The symbols for the previous discrete states.
    show_compile_output : bool
        Whether to show the compile output or not.
    state_derivative_symbols : n-tuple
        symbols for the time derivatives of the states.
    time_symbol : sympy.Symbol
        The symbol used to represent time, usually `t`.
    tmp_dir
        The temporary directory used to store files generated.
    unknown_input_trajectories : q-tuple
        The unknown input trajectories symbols.
    unknown_parameters : r-tuple
        The unknown parameters in the problem, in the sequence in which they
        appear in the solution of the optimization.

    Notes
    =====

    - N : number of collocation nodes
    - n : number of states
    - m : number of input trajectories
    - q : number of unknown input trajectories
    - r : number of unknown parameters
    - s : number of unknown time intervals (0 or 1 if fixed duration or
      variable duration)
    - o : number of instance constraints
    - nN + qN + r + s : number of free variables
    - n(N - 1) + o : number of constraints

    Some of the attributes are explained in more detail under Parameters below.

    It is best to treat ``ConstraintCollocator`` as immutable, changing
    attributes after initialization will inevitably fail.

    """
    def __init__(self, equations_of_motion, state_symbols,
                 num_collocation_nodes, node_time_interval,
                 known_parameter_map={}, known_trajectory_map={},
                 instance_constraints=None, time_symbol=None, tmp_dir=None,
                 integration_method='backward euler', parallel=False,
                 show_compile_output=False):
        """Instantiates a ConstraintCollocator object.

        Parameters
        ==========
        equations_of_motion : sympy.Matrix, shape(n, 1)
            A column matrix of SymPy expressions defining the right hand side
            of the equations of motion when the left hand side is zero, e.g.
            ``0 = x'(t) - f(x(t), u(t), p)`` or ``0 = f(x'(t), x(t), u(t),
            p)``. These should be in first order form but not necessairly
            explicit. They can be ordinary differential equations or
            differential algebraic equations.
        state_symbols : iterable
            An iterable containing all ``n`` of the SymPy functions of time
            which represent the states in the equations of motion.
        num_collocation_nodes : integer
            The number of collocation nodes, ``N``. All known trajectory arrays
            should be of this length.
        node_time_interval : float or Symbol
            The time interval between collocation nodes. If a SymPy symbol is
            provided, the time interval will be treated as a free variable
            resulting in a variable duration solution.
        known_parameter_map : dictionary, optional
            A dictionary that maps the SymPy symbols representing the known
            constant parameters to floats. Any parameters in the equations of
            motion not provided in this dictionary will become free
            optimization variables.
        known_trajectory_map : dictionary, optional
            A dictionary that maps the non-state SymPy functions of time to
            ndarrays of floats of ``shape(N,)``. Any time varying parameters in
            the equations of motion not provided in this dictionary will become
            free trajectories optimization variables. If solving a variable
            duration problem, note that the values here are fixed at each node
            and will not scale with a varying time interval.
        instance_constraints : iterable of SymPy expressions, optional
            These expressions are for constraints on the states at specific
            times. They can be expressions with any state instance and any of
            the known parameters found in the equations of motion. All states
            should be evaluated at a specific instant of time. For example, the
            constraint ``x(0) = 5.0`` would be specified as ``x(0) - 5.0``. For
            variable duration problems you must specify time as an integer
            multiple of the node time interval symbol, for example ``x(0*h) -
            5.0``. The integer must be a value from 0 to
            ``num_collocation_nodes - 1``. Unknown parameters and time varying
            parameters other than the states are currently not supported.
        time_symbol : SymPy Symbol, optional
            The symbol representating time in the equations of motion. If not
            given, it is assumed to be the default stored in
            ``sympy.physics.vector.dynamicsymbols._t``.
        tmp_dir : string, optional
            If you want to see the generated Cython and C code for the
            constraint and constraint Jacobian evaluations, pass in a path to a
            directory here.
        integration_method : string, optional
            The integration method to use, either ``backward euler`` or
            ``midpoint``.
        parallel : boolean, optional
            If true and openmp is installed, constraints and the Jacobian of
            the constraints will be executed across multiple threads. This is
            only useful for performance when the equations of motion have an
            extremely large number of operations.
        show_compile_output : boolean, optional
            If True, STDOUT and STDERR of the Cython compilation call will be
            shown.

        """
        self.eom = equations_of_motion

        if time_symbol is not None:
            self.time_symbol = time_symbol
            me.dynamicsymbols._t = time_symbol
        else:
            self.time_symbol = me.dynamicsymbols._t

        self.state_symbols = tuple(state_symbols)
        self.state_derivative_symbols = tuple([s.diff(self.time_symbol) for
                                               s in state_symbols])
        self.num_states = len(self.state_symbols)

        self.num_collocation_nodes = num_collocation_nodes

        if isinstance(node_time_interval, sm.Symbol):
            self.time_interval_symbol = node_time_interval
            self._variable_duration = True
        else:
            self.time_interval_symbol = sm.Symbol('h_opty', real=True)
            self._variable_duration = False
        self.node_time_interval = node_time_interval

        self.known_parameter_map = known_parameter_map
        self.known_trajectory_map = known_trajectory_map

        self.instance_constraints = instance_constraints

        self.num_constraints = self.num_states * (num_collocation_nodes - 1)

        self.tmp_dir = tmp_dir
        self.parallel = parallel
        self.show_compile_output = show_compile_output

        self._sort_parameters()
        self._check_known_trajectories()
        self._sort_trajectories()
        self.num_free = ((self.num_states +
                          self.num_unknown_input_trajectories) *
                         self.num_collocation_nodes +
                         self.num_unknown_parameters +
                         int(self._variable_duration))

        self.integration_method = integration_method

        if instance_constraints is not None:
            self.num_instance_constraints = len(instance_constraints)
            self.num_constraints += self.num_instance_constraints
            self._identify_functions_in_instance_constraints()
            self._find_closest_free_index()
            self.eval_instance_constraints = self._instance_constraints_func()
            self.eval_instance_constraints_jacobian_values = \
                self._instance_constraints_jacobian_values_func()
        else:
            self.num_instance_constraints = 0

    @property
    def integration_method(self):
        return self._integration_method

    @integration_method.setter
    def integration_method(self, method):
        """The method can be ``'backward euler'`` or ``'midpoint'``."""
        if method not in ['backward euler', 'midpoint']:
            msg = ("{} is not a valid integration method.")
            raise ValueError(msg.format(method))
        else:
            self._integration_method = method
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

        # TODO : Should the full parameter list be sorted here for consistency?
        parameters = self.eom.free_symbols.copy()
        if self.time_symbol in parameters:
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
        """Raises and error if the known trajectories are not the correct
        length."""

        N = self.num_collocation_nodes

        for k, v in self.known_trajectory_map.items():
            if len(v) != N:
                msg = 'The known parameter {} is not length {}.'
                raise ValueError(msg.format(k, N))

    def _sort_trajectories(self):
        """Finds and counts all of the non-state, time varying parameters in
        the equations of motion and categorizes them based on which parameters
        the user supplies. The unknown parameters are sorted by name."""

        states = set(self.state_symbols)
        states_derivatives = set(self.state_derivative_symbols)

        # TODO : Add tests for time symbols that are not `t`.
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
        """Instantiates discrete symbols for each time varying variable in the
        equations of motion.

        Instantiates
        ------------
        previous_discrete_state_symbols : tuple of sympy.Symbols
            The n symbols representing the system's (ith - 1) states.
        current_discrete_state_symbols : tuple of sympy.Symbols
            The n symbols representing the system's ith states.
        next_discrete_state_symbols : tuple of sympy.Symbols
            The n symbols representing the system's (ith + 1) states.
        current_known_discrete_specified_symbols : tuple of sympy.Symbols
            The symbols representing the system's ith known input
            trajectories.
        next_known_discrete_specified_symbols : tuple of sympy.Symbols
            The symbols representing the system's (ith + 1) known input
            trajectories.
        current_unknown_discrete_specified_symbols : tuple of sympy.Symbols
            The symbols representing the system's ith unknown input
            trajectories.
        next_unknown_discrete_specified_symbols : tuple of sympy.Symbols
            The symbols representing the system's (ith + 1) unknown input
            trajectories.
        current_discrete_specified_symbols : tuple of sympy.Symbols
            The m symbols representing the system's ith specified inputs.
        next_discrete_specified_symbols : tuple of sympy.Symbols
            The m symbols representing the system's (ith + 1) specified
            inputs.

        """

        # The previus, current, and next states.
        self.previous_discrete_state_symbols = \
            tuple([sm.Symbol(f.__class__.__name__ + 'p', real=True)
                   for f in self.state_symbols])
        self.current_discrete_state_symbols = \
            tuple([sm.Symbol(f.__class__.__name__ + 'i', real=True)
                   for f in self.state_symbols])
        self.next_discrete_state_symbols = \
            tuple([sm.Symbol(f.__class__.__name__ + 'n', real=True)
                   for f in self.state_symbols])

        # The current and next known input trajectories.
        self.current_known_discrete_specified_symbols = \
            tuple([sm.Symbol(f.__class__.__name__ + 'i', real=True)
                   for f in self.known_input_trajectories])
        self.next_known_discrete_specified_symbols = \
            tuple([sm.Symbol(f.__class__.__name__ + 'n', real=True)
                   for f in self.known_input_trajectories])

        # The current and next unknown input trajectories.
        self.current_unknown_discrete_specified_symbols = \
            tuple([sm.Symbol(f.__class__.__name__ + 'i', real=True)
                   for f in self.unknown_input_trajectories])
        self.next_unknown_discrete_specified_symbols = \
            tuple([sm.Symbol(f.__class__.__name__ + 'n', real=True)
                   for f in self.unknown_input_trajectories])

        self.current_discrete_specified_symbols = (
            self.current_known_discrete_specified_symbols +
            self.current_unknown_discrete_specified_symbols)
        self.next_discrete_specified_symbols = (
            self.next_known_discrete_specified_symbols +
            self.next_unknown_discrete_specified_symbols)

    def _discretize_eom(self):
        """Instantiates the constraint equations in a discretized form using
        backward Euler or midpoint discretization.

        Instantiates
        ------------
        discrete_eoms : sympy.Matrix, shape(n, 1)
            The column vector of the discretized equations of motion.

        """
        logging.info('Discretizing the equations of motion.')
        x = self.state_symbols
        xd = self.state_derivative_symbols
        u = self.input_trajectories

        xp = self.previous_discrete_state_symbols
        xi = self.current_discrete_state_symbols
        xn = self.next_discrete_state_symbols
        ui = self.current_discrete_specified_symbols
        un = self.next_discrete_specified_symbols

        h = self.time_interval_symbol

        if self.integration_method == 'backward euler':

            deriv_sub = {d: (i - p) / h for d, i, p in zip(xd, xi, xp)}

            func_sub = dict(zip(x + u, xi + ui))

            self.discrete_eom = me.msubs(self.eom, deriv_sub, func_sub)

        elif self.integration_method == 'midpoint':

            xdot_sub = {d: (n - i) / h for d, i, n in zip(xd, xi, xn)}
            x_sub = {d: (i + n) / 2 for d, i, n in zip(x, xi, xn)}
            u_sub = {d: (i + n) / 2 for d, i, n in zip(u, ui, un)}
            self.discrete_eom = me.msubs(self.eom, xdot_sub, x_sub, u_sub)

    def _identify_functions_in_instance_constraints(self):
        """Instantiates a set containing all of the instance functions, i.e.
        x(1.0) in the instance constraints."""

        all_funcs = set()

        for con in self.instance_constraints:
            all_funcs = all_funcs.union(con.atoms(sm.Function))

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

        node_map = {}
        for func in self.instance_constraint_function_atoms:
            if self._variable_duration:
                if func.args[0] == 0:
                    time_idx = 0
                else:
                    try:
                        time_idx = int(func.args[0]/self.time_interval_symbol)
                    except TypeError as err:  # can't convert to integer
                        msg = ('Instance constraint {} is not a correct '
                               'integer multiple of the time interval.')
                        raise TypeError(msg.format(func)) from err
                if time_idx not in range(self.num_collocation_nodes):
                    msg = ('Instance constraint {} gives an index of {} which '
                           'is not between 0 and {}.')
                    raise ValueError(msg.format(
                        func, time_idx, self.num_collocation_nodes - 1))
            else:
                time_value = func.args[0]
                time_vector = np.linspace(0.0, duration, num=N)
                time_idx = np.argmin(np.abs(time_vector - time_value))
            free_index = determine_free_index(time_idx,
                                              func.__class__(self.time_symbol))
            node_map[func] = free_index

        self.instance_constraints_free_index_map = node_map

    def _instance_constraints_func(self):
        """Returns a function that evaluates the instance constraints given
        the free optimization variables."""
        free = sm.DeferredVector('FREE')
        def_map = {k: free[v] for k, v in
                   self.instance_constraints_free_index_map.items()}
        subbed_constraints = [con.subs(def_map) for con in
                              self.instance_constraints]
        f = sm.lambdify(([free] + list(self.known_parameter_map.keys())),
                        subbed_constraints, modules=[{'ImmutableMatrix':
                                                      np.array}, "numpy"])

        return lambda free: f(free, *self.known_parameter_map.values())

    def _instance_constraints_jacobian_indices(self):
        """Returns the row and column indices of the non-zero values in the
        Jacobian of the constraints."""
        idx_map = self.instance_constraints_free_index_map

        num_eom_constraints = self.num_states*(self.num_collocation_nodes - 1)

        rows = []
        cols = []

        for i, con in enumerate(self.instance_constraints):
            funcs = con.atoms(sm.Function)
            indices = [idx_map[f] for f in funcs]
            row_idxs = num_eom_constraints + i * np.ones(len(indices),
                                                         dtype=int)
            rows += list(row_idxs)
            cols += indices

        return np.array(rows, dtype=int), np.array(cols, dtype=int)

    def _instance_constraints_jacobian_values_func(self):
        """Returns the non-zero values of the constraint Jacobian associated
        with the instance constraints."""
        free = sm.DeferredVector('FREE')

        def_map = {k: free[v] for k, v in
                   self.instance_constraints_free_index_map.items()}

        funcs = []
        num_vals_per_func = []
        for con in self.instance_constraints:
            partials = list(con.atoms(sm.Function))
            num_vals_per_func.append(len(partials))
            jac = sm.Matrix([con]).jacobian(partials)
            jac = jac.subs(def_map)
            funcs.append(sm.lambdify(([free] +
                                      list(self.known_parameter_map.keys())),
                                     jac, modules=[{'ImmutableMatrix':
                                                    np.array}, "numpy"]))
        length = np.sum(num_vals_per_func)

        def wrapped(free):
            arr = np.zeros(length)
            j = 0
            for i, (f, num) in enumerate(zip(funcs, num_vals_per_func)):
                arr[j:j + num] = f(free, *self.known_parameter_map.values())
                j += num
            return arr

        return wrapped

    def _gen_multi_arg_con_func(self):
        """Instantiates a function that evaluates the constraints given all of
        the arguments of the functions, i.e. not just the free optimization
        variables.

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
        xn_syms = self.next_discrete_state_symbols
        si_syms = self.current_discrete_specified_symbols
        sn_syms = self.next_discrete_specified_symbols
        h_sym = self.time_interval_symbol
        constant_syms = self.known_parameters + self.unknown_parameters

        if self.integration_method == 'backward euler':

            args = [x for x in xi_syms] + [x for x in xp_syms]
            args += [s for s in si_syms] + list(constant_syms) + [h_sym]

            current_start = 1
            current_stop = None
            adjacent_start = None
            adjacent_stop = -1

        elif self.integration_method == 'midpoint':

            args = [x for x in xi_syms] + [x for x in xn_syms]
            args += [s for s in si_syms] + [s for s in sn_syms]
            args += list(constant_syms) + [h_sym]

            current_start = None
            current_stop = -1
            adjacent_start = 1
            adjacent_stop = None

        logging.info('Compiling the constraint function.')
        f = ufuncify_matrix(args, self.discrete_eom,
                            const=constant_syms + (h_sym,),
                            tmp_dir=self.tmp_dir, parallel=self.parallel,
                            show_compile_output=self.show_compile_output)

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
            # n x N - 1
            x_current = state_values[:, current_start:current_stop]
            # n x N - 1
            x_adjacent = state_values[:, adjacent_start:adjacent_stop]

            # 2n x N - 1
            args = [x for x in x_current] + [x for x in x_adjacent]

            # 2n + m x N - 1
            if len(specified_values.shape) == 2:
                assert specified_values.shape == (self.num_input_trajectories,
                                                  self.num_collocation_nodes)
                si = specified_values[:, current_start:current_stop]
                args += [s for s in si]
                if self.integration_method == 'midpoint':
                    sn = specified_values[:, adjacent_start:adjacent_stop]
                    args += [s for s in sn]
            elif (len(specified_values.shape) == 1 and
                  specified_values.size != 0):
                assert specified_values.shape == (self.num_collocation_nodes,)
                si = specified_values[current_start:current_stop]
                args += [si]
                if self.integration_method == 'midpoint':
                    sn = specified_values[adjacent_start:adjacent_stop]
                    args += [sn]

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
        jac_row_idxs : ndarray, shape(2*n + q + r + s,)
            The row indices for the non-zero values in the Jacobian.
        jac_col_idxs : ndarray, shape(n + o,)
            The column indices for the non-zero values in the Jacobian.

        """

        N = self.num_collocation_nodes
        n = self.num_states

        num_constraint_nodes = N - 1

        if self.integration_method == 'backward euler':

            num_partials = n * (2 * n + self.num_unknown_input_trajectories +
                                self.num_unknown_parameters +
                                int(self._variable_duration))

        elif self.integration_method == 'midpoint':

            num_partials = n * (2 * n + 2 *
                                self.num_unknown_input_trajectories +
                                self.num_unknown_parameters +
                                int(self._variable_duration))

        num_non_zero_values = num_constraint_nodes * num_partials

        if self.instance_constraints is not None:
            ins_row_idxs, ins_col_idxs = \
                self._instance_constraints_jacobian_indices()
            num_non_zero_values += len(ins_row_idxs)

        jac_row_idxs = np.empty(num_non_zero_values, dtype=int)
        jac_col_idxs = np.empty(num_non_zero_values, dtype=int)

        """
        The symbolic derivative matrix for a single constraint node follows
        these patterns:

        Backward Euler
        --------------
        i: ith, p: ith-1

        For example:
        x1i = the first state at the ith constraint node
        uqi = the qth input at the ith constraint node
        uqn = the qth input at the ith+1 constraint node

        [x1] [x1i, ..., xni, x1p, ..., xnp, u1i, .., uqi, p1, ..., pr, h]
        [. ]
        [. ]
        [. ]
        [xn]

        Midpoint
        --------
        i: ith, n: ith+1

        [x1] [x1i, ..., xni, x1n, ..., xnn, u1i, .., uqi, u1n, ..., uqn, p1, ..., pp, h]
        [. ]
        [. ]
        [. ]
        [xn]

        Each of these matrices are evaulated at N-1 constraint nodes and
        then the 3D matrix is flattened into a 1d array. The backward euler
        uses nodes 1 <= i <= N-1 and the midpoint uses 0 <= i <= N - 2. So
        the flattened arrays looks like:

        M = N-1
        P = N-2

        Backward Euler
        --------------

        i=1  x1  | [x11, ..., xn1, x10, ..., xn0, u11, .., uq1, p1, ..., pr, h,
             x2  |  x11, ..., xn1, x10, ..., xn0, u11, .., uq1, p1, ..., pr, h,
             ... |  ...,
             xn  |  x11, ..., xn1, x10, ..., xn0, u11, .., uq1, p1, ..., pr, h,
        i=2  x1  |  x12, ..., xn2, x11, ..., xn1, u12, .., uq2, p1, ..., pr, h,
             x2  |  x12, ..., xn2, x11, ..., xn1, u12, .., uq2, p1, ..., pr, h,
             ... |  ...,
             xn  |  x12, ..., xn2, x11, ..., xn1, u12, .., uq2, p1, ..., pr, h,
                 |  ...,
        i=M  x1  |  x1M, ..., xnM, x1P, ..., xnP, u1M, .., uqM, p1, ..., pr, h,
             x2  |  x1M, ..., xnM, x1P, ..., xnP, u1M, .., uqM, p1, ..., pr, h,
             ... |  ...,
             xn  |  x1M, ..., xnM, x1P, ..., xnP, u1M, .., uqM, p1, ..., pr, h]

        Midpoint
        --------

        i=0   x1  | [x10, ..., xn0, x11, ..., xn1, u10, .., uq0, u11, .., uq1, p1, ..., pr, h,
                x2  |  x10, ..., xn0, x11, ..., xn1, u10, .., uq0, u11, .., uq1, p1, ..., pr, h,
                ... |  ...,
                xn  |  x10, ..., xn0, x11, ..., xn1, u10, .., uq0, u11, .., uq1, p1, ..., pr, h,
        i=1   x1  |  x11, ..., xn1, x12, ..., xn2, u11, .., uq1, u12, .., uq2, p1, ..., pr, h,
                x2  |  x11, ..., xn1, x12, ..., xn2, u11, .., uq1, u12, .., uq2, p1, ..., pr, h,
                ... |  ...,
                xn  |  x11, ..., xn1, x12, ..., xn2, u11, .., uq1, u12, .., uq2, p1, ..., pr, h,
                ... |  ...,
        i=P   x1  |  x1P, ..., xnP, x1M, ..., xnM, u1P, .., uqP, u1M, .., uqM, p1, ..., pr, h,
                x2  |  x1P, ..., xnP, x1M, ..., xnM, u1P, .., uqP, u1M, .., uqM, p1, ..., pr, h,
                ... |  ...,
                xn  |  x1P, ..., xnP, x1M, ..., xnM, u1P, .., uqP, u1M, .., uqM, p1, ..., pr, h]

        These two arrays contain of the non-zero values of the sparse
        Jacobian[#]_.

        .. [#] Some of the partials can be equal to zero and could be
            excluded from the array. These could be a significant number.

        Now we need to generate the triplet format indices of the full
        sparse Jacobian for each one of the entries in these arrays. The
        format of the Jacobian matrix is:

        Backward Euler
        --------------

                [x10, ..., x1N-1, ..., xn0, ..., xnN-1, u10, ..., u1N-1, ..., uq0, ..., uqN-1, p1, ..., pr, h]
        [x11]
        [x12]
        [...]
        [x1M]
        [...]
        [xn1]
        [xn2]
        [...]
        [xnM]

        Midpoint
        --------

                [x10, ..., x1N-1, ..., xn0, ..., xnN-1, u10, ..., u1N-1, ..., uq0, ..., uqN-1, p1, ..., pr, h]
        [x10]
        [x11]
        [...]
        [x1P]
        [...]
        [xn0]
        [xn1]
        [...]
        [xnP]


        """
        for i in range(num_constraint_nodes):

            # n : number of states
            # m : number of input trajectories
            # p : number of parameters
            # q : number of unknown input trajectories
            # r : number of unknown parameters
            # s : number of unknown time intervals

            # the states repeat every N - 1 constraints
            # row_idxs = [0 * (N - 1), 1 * (N - 1),  2 * (N - 1), ..., n * (N - 1)]

            # This gives the Jacobian row indices matching the ith
            # constraint node for each state. ith corresponds to the loop
            # indice.
            row_idxs = [j * (num_constraint_nodes) + i for j in range(n)]

            # first row, the columns indices mapping is:
            # [1, N + 1, ..., N - 1] : [x1p, x1i, 0, ..., 0]
            # [0, N, ..., 2 * (N - 1)] : [x2p, x2i, 0, ..., 0]
            # [-p:] : p1,..., pp  the free constants

            # i=0: [1, ..., n * N + 1, 0, ..., n * N + 0, n * N:n * N + p]
            # i=1: [2, ..., n * N + 2, 1, ..., n * N + 1, n * N:n * N + p]
            # i=2: [3, ..., n * N + 3, 2, ..., n * N + 2, n * N:n * N + p]

            if self.integration_method == 'backward euler':

                col_idxs = [j * N + i + 1 for j in range(n)]
                col_idxs += [j * N + i for j in range(n)]
                col_idxs += [n * N + j * N + i + 1 for j in
                             range(self.num_unknown_input_trajectories)]
                col_idxs += [(n + self.num_unknown_input_trajectories) * N + j
                             for j in range(self.num_unknown_parameters +
                                            int(self._variable_duration))]

            elif self.integration_method == 'midpoint':

                col_idxs = [j * N + i for j in range(n)]
                col_idxs += [j * N + i + 1 for j in range(n)]
                col_idxs += [n * N + j * N + i for j in
                             range(self.num_unknown_input_trajectories)]
                col_idxs += [n * N + j * N + i + 1 for j in
                             range(self.num_unknown_input_trajectories)]
                col_idxs += [(n + self.num_unknown_input_trajectories) * N + j
                             for j in range(self.num_unknown_parameters +
                                            int(self._variable_duration))]

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
        xn_syms = self.next_discrete_state_symbols
        si_syms = self.current_discrete_specified_symbols
        sn_syms = self.next_discrete_specified_symbols
        ui_syms = self.current_unknown_discrete_specified_symbols
        un_syms = self.next_unknown_discrete_specified_symbols
        h_sym = self.time_interval_symbol
        constant_syms = self.known_parameters + self.unknown_parameters

        if self.integration_method == 'backward euler':

            # The free parameters are always the n * (N - 1) state values,
            # the unknown input trajectories, and the unknown model
            # constants, so the base Jacobian needs to be taken with respect
            # to the ith, and ith - 1 states, and the free model constants.
            wrt = (xi_syms + xp_syms + ui_syms + self.unknown_parameters)
            if self._variable_duration:
                wrt += (h_sym,)

            # The arguments to the Jacobian function include all of the free
            # Symbols/Functions in the matrix expression.
            args = xi_syms + xp_syms + si_syms + constant_syms + (h_sym,)

            current_start = 1
            current_stop = None
            adjacent_start = None
            adjacent_stop = -1

        elif self.integration_method == 'midpoint':

            wrt = (xi_syms + xn_syms + ui_syms + un_syms +
                   self.unknown_parameters)
            if self._variable_duration:
                wrt += (h_sym,)

            # The arguments to the Jacobian function include all of the free
            # Symbols/Functions in the matrix expression.
            args = (xi_syms + xn_syms + si_syms + sn_syms + constant_syms +
                    (h_sym,))

            current_start = None
            current_stop = -1
            adjacent_start = 1
            adjacent_stop = None

        # This creates a matrix with all of the symbolic partial derivatives
        # necessary to compute the full Jacobian.
        logging.info('Differentiating the constraint function.')
        discrete_eom_matrix = sm.ImmutableDenseMatrix(self.discrete_eom)
        wrt_matrix = sm.ImmutableDenseMatrix([list(wrt)])
        symbolic_partials = _forward_jacobian(discrete_eom_matrix,
                                              wrt_matrix.T)

        # This generates a numerical function that evaluates the matrix of
        # partial derivatives. This function returns the non-zero elements
        # needed to build the sparse constraint Jacobian.
        logging.info('Compiling the Jacobian function.')
        eval_partials = ufuncify_matrix(args, symbolic_partials,
                                        const=constant_syms + (h_sym,),
                                        tmp_dir=self.tmp_dir,
                                        parallel=self.parallel)

        if isinstance(symbolic_partials, tuple) and len(symbolic_partials) == 2:
            num_rows = symbolic_partials[1][0].shape[0]
            num_cols = symbolic_partials[1][0].shape[1]
        else:
            num_rows = symbolic_partials.shape[0]
            num_cols = symbolic_partials.shape[1]
        result = np.empty((self.num_collocation_nodes - 1, num_rows*num_cols))

        def constraints_jacobian(state_values, specified_values,
                                 parameter_values, interval_value):
            """Returns the values of the sparse constraing Jacobian matrix
            given all of the values for each variable in the equations of
            motion over the N - 1 nodes.

            Parameters
            ----------
            states : ndarray, shape(n, N)
                The array of n states through N time steps. There are always
                at least two states.
            specified_values : ndarray, shape(m, N) or shape(N,)
                The array of m specified inputs through N time steps.
            parameter_values : ndarray, shape(p,)
                The array of p parameter.
            interval_value : float
                The value of the discretization time interval.

            Returns
            -------
            constraint_jacobian_values : ndarray, shape(see below,)
                backward euler: shape((N - 1) * n * (2*n + q + r + s),)
                midpoint: shape((N - 1) * n * (2*n + 2*q + r + s),)
                The values of the non-zero entries of the constraints
                Jacobian. These correspond to the triplet formatted indices
                returned from jacobian_indices.

            Notes
            -----
            - N : number of collocation nodes
            - n : number of states
            - m : number of input trajectories
            - p : number of parameters
            - q : number of unknown input trajectories
            - r : number of unknown parameters
            - s : number of unknown time intervals
            - n*(N - 1) : number of constraints

            """
            if state_values.shape[0] < 2:
                raise ValueError('There should always be at least two states.')

            # Each of these arrays are shape(n, N - 1). The x_adjacent is
            # either the previous value of the state or the next value of
            # the state, depending on the integration method.
            x_current = state_values[:, current_start:current_stop]
            x_adjacent = state_values[:, adjacent_start:adjacent_stop]

            # 2n x N - 1
            args = [x for x in x_current] + [x for x in x_adjacent]

            # 2n + m x N - 1
            if len(specified_values.shape) == 2:
                si = specified_values[:, current_start:current_stop]
                args += [s for s in si]
                if self.integration_method == 'midpoint':
                    sn = specified_values[:, adjacent_start:adjacent_stop]
                    args += [s for s in sn]
            elif len(specified_values.shape) == 1 and specified_values.size != 0:
                si = specified_values[current_start:current_stop]
                args += [si]
                if self.integration_method == 'midpoint':
                    sn = specified_values[adjacent_start:adjacent_stop]
                    args += [sn]

            args += [c for c in parameter_values]
            args += [interval_value]

            # backward euler: shape(N - 1, n, 2*n + q + r)
            # midpoint: shape(N - 1, n, 2*n + 2*q + r)
            non_zero_derivatives = eval_partials(result, *args)

            return non_zero_derivatives.ravel()

        self._multi_arg_con_jac_func = constraints_jacobian

    @staticmethod
    def _merge_fixed_free(syms, fixed, free, typ):
        """Returns an array with the fixed and free values combined. This just
        takes the known and unknown values and combines them for the function
        evaluation.

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

            if self._variable_duration:
                (free_states, free_specified, free_constants,
                 time_interval) = parse_free(
                     free, self.num_states,
                     self.num_unknown_input_trajectories,
                     self.num_collocation_nodes,
                     variable_duration=self._variable_duration)
            else:
                free_states, free_specified, free_constants = parse_free(
                    free, self.num_states, self.num_unknown_input_trajectories,
                    self.num_collocation_nodes,
                    variable_duration=self._variable_duration)
                time_interval = self.node_time_interval

            all_specified = self._merge_fixed_free(self.input_trajectories,
                                                   self.known_trajectory_map,
                                                   free_specified, 'traj')

            all_constants = self._merge_fixed_free(self.parameters,
                                                   self.known_parameter_map,
                                                   free_constants, 'par')

            eom_con_vals = func(free_states, all_specified, all_constants,
                                time_interval)

            if self.instance_constraints is not None:
                if typ == 'con':
                    ins_con_vals = self.eval_instance_constraints(free)
                elif typ == 'jac':
                    ins_con_vals = \
                        self.eval_instance_constraints_jacobian_values(free)
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
        logging.info('Generating constraint function.')
        self._gen_multi_arg_con_func()
        return self._wrap_constraint_funcs(self._multi_arg_con_func, 'con')

    def generate_jacobian_function(self):
        """Returns a function which evaluates the Jacobian of the
        constraints given the array of free optimization variables."""
        logging.info('Generating jacobian function.')
        self._gen_multi_arg_con_jac_func()
        return self._wrap_constraint_funcs(self._multi_arg_con_jac_func, 'jac')
