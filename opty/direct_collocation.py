#!/usr/bin/env python

import sys
from functools import wraps
import logging

import numpy as np
import sympy as sm
from sympy.physics import mechanics as me
import cyipopt
plt = sm.external.import_module('matplotlib.pyplot',
                                import_kwargs={'fromlist': ['']},
                                catch=(RuntimeError,))

from .utils import (ufuncify_matrix, lambdify_matrix, parse_free,
                    _optional_plt_dep, _forward_jacobian, sort_sympy)

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
        func.__doc__ = self._combine_docs(
            self.mthd.__doc__, ConstraintCollocator.__init__.__doc__)
        return func

    @staticmethod
    def _combine_docs(prob_doc, coll_doc):
        beg, end = prob_doc.split('SPLIT')
        if sys.version_info[1] >= 13:
            sep = 'Parameters\n==========\n'
            _, middle = coll_doc.split(sep)
            mid = middle[:-1]
        else:
            sep = 'Parameters\n        ==========\n        '
            _, middle = coll_doc.split(sep)
            mid = middle[:-9]
        return beg + mid + end


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
    - M : number of equations of motion
    - n : number of states
    - m : number of input trajectories
    - q : number of unknown input trajectories
    - r : number of unknown parameters
    - s : number of unknown time intervals (0 or 1 if fixed duration or
      variable duration)
    - o : number of instance constraints
    - nN + qN + r + s : number of free variables
    - M(N - 1) + o : number of constraints

    If ``x`` are the state variables, ``u`` are the unknown input trajectories,
    and ``p`` are the unknown parameters, and ``h`` is the unknown time
    interval then the free optimization variables are in this order::

       free = [x11, ... x1N,
               xn1, ... xnN,
               u11, ... u1N,
               uq1, ... xqN,
               p1, ... pr,
               h]

    If the equations of motion are equations ``eom1`` to ``eomM`` and  instance
    constraints are ``c``,  the constraint array is ordered as::

       constraints = [eom12, ... eom1N,
                      eomM2, ... eomMN,
                      c1, ..., co]

    """

    INF = 10e19

    @_doc_inherit
    def __init__(self, obj, obj_grad, equations_of_motion, state_symbols,
                 num_collocation_nodes, node_time_interval,
                 known_parameter_map={}, known_trajectory_map={},
                 instance_constraints=None, time_symbol=None, tmp_dir=None,
                 integration_method='backward euler', parallel=False,
                 bounds=None, show_compile_output=False, backend='cython',
                 eom_bounds=None):
        """

        Parameters
        ==========
        obj : function
            Returns the value of the objective function given the free vector.
        obj_grad : function
            Returns the gradient of the objective function given the free
            vector.
        SPLIT
        bounds : dictionary, optional
            This dictionary should contain a mapping from any of the symbolic
            states, unknown trajectories, unknown parameters, or unknown time
            interval to a 2-tuple of floats, the first being the lower bound
            and the second the upper bound for that free variable, e.g.
            ``{x(t): (-1.0, 5.0)}``.
        eom_bounds : dictionary, optional
            Optional lower and upper bounds for the equations of motion,
            default is ``(0.0, 0.0)`` for each equation making them equality
            constraints. Dictionary is a mapping of equation of motion integer
            indices to a tuple of a lower and upper bounds given as floats.
            The index integer corresponds to the order of
            ``equations_of_motion``.  Example: ``{3: (0.0, np.inf)}`` would
            make the 4th equation of motion an inequality constraint that
            cannot be below zero. Beware of transforming essential differential
            equations into inequality constraints, as that is likely not
            desired. These are typically used only for additional path
            constraints.

        """

        # TODO : This check belongs in the ConstraintCollocator, not here.
        if not equations_of_motion.has(sm.Derivative):
            raise ValueError('No time derivatives are present.'
                             ' The equations of motion must be ordinary '
                             'differential equations (ODEs) or '
                             'differential algebraic equations (DAEs).')

        self.collocator = ConstraintCollocator(
            equations_of_motion, state_symbols, num_collocation_nodes,
            node_time_interval, known_parameter_map, known_trajectory_map,
            instance_constraints, time_symbol, tmp_dir, integration_method,
            parallel, show_compile_output=show_compile_output, backend=backend)

        self.bounds = bounds
        self.eom_bounds = eom_bounds
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
        self._generate_constraint_bound_arrays()

        super(Problem, self).__init__(n=self.num_free,
                                      m=self.num_constraints,
                                      lb=self.lower_bound,
                                      ub=self.upper_bound,
                                      cl=self._low_con_bounds,
                                      cu=self._upp_con_bounds)

        self.obj_value = []

    def solve(self, free, lagrange=[], zl=[], zu=[], respect_bounds=False):
        """Returns the optimal solution and an info dictionary.

        Solves the posed optimization problem starting at point x.

        Parameters
        ----------
        x : array-like, shape(n*N + q*N + r + s, )
            Initial guess.

        lagrange : array-like, shape(n*(N-1) + o, ), optional (default=[])
            Initial values for the constraint multipliers (only if warm start
            option is chosen).

        zl : array-like, shape(n*N + q*N + r + s, ), optional (default=[])
            Initial values for the multipliers for lower variable bounds (only
            if warm start option is chosen).

        zu : array-like, shape(n*N + q*N + r + s, ), optional (default=[])
            Initial values for the multipliers for upper variable bounds (only
            if warm start option is chosen).

        respect_bounds : bool, optional (default=False)
            If True, the initial guess is checked to ensure that it is within
            the bounds, and a ValueError is raised if it is not. If False, the
            initial guess is not checked.

        Returns
        -------
        x : :py:class:`numpy.ndarray`, shape`(n*N + q*N + r + s, )`
            Optimal solution.
        info: :py:class:`dict` with the following entries
            ``x``: :py:class:`numpy.ndarray`, shape`(n*N + q*N + r + s, )`
                optimal solution
            ``g``: :py:class:`numpy.ndarray`, shape`(M*(N-1) + o, )`
                constraints at the optimal solution
            ``obj_val``: :py:class:`float`
                objective value at optimal solution
            ``mult_g``: :py:class:`numpy.ndarray`, shape`(M*(N-1) + o, )`
                final values of the constraint multipliers
            ``mult_x_L``: :py:class:`numpy.ndarray`, shape`(M*N + q*N + r + s, )`
                bound multipliers at the solution
            ``mult_x_U``: :py:class:`numpy.ndarray`, shape`(M*N + q*N + r + s, )`
                bound multipliers at the solution
            ``status``: :py:class:`int`
                gives the status of the algorithm
            ``status_msg``: :py:class:`str`
                gives the status of the algorithm as a message

        """
        if respect_bounds:
            self.check_bounds_conflict(free)
        return super().solve(free, lagrange=lagrange, zl=zl, zu=zu)

    def check_bounds_conflict(self, free):
        """
        Ascertains that the initial guesses for all variables are within the
        limits prescribed by their respective bounds. Raises a ValueError if
        for any variable the initial guess is outside its bounds, or if the
        lower bound is greater than the upper bound.

        Parameters
        ----------
        free : array_like, shape(n*N + q*N + r + s, )
            Initial guess given to solve.

        Raises
        ------
        ValueError
            If the lower bound for variable is greater than its upper bound,
            ``opty`` may not break, but the solution will likely not be
            correct. Hence a ValueError is raised in such as case.

            If the initial guess for any variable is outside its bounds,
            a ValueError is raised.

        """
        if self.bounds is not None:
            errors = []
            # check for reversed bounds
            for key in self.bounds.keys():
                if self.bounds[key][0] > self.bounds[key][1]:
                    errors.append(key)
            if len(errors) > 0:
                msg = (f'The lower bound(s) for {errors} is (are) greater than'
                       f' the upper bound(s).')
                raise ValueError(msg)

            violating_variables = []

            if self.collocator._variable_duration:
                local_ts = self.collocator.time_interval_symbol
                if local_ts in self.bounds.keys():
                    if (free[-1] < self.bounds[local_ts][0]
                        or free[-1] > self.bounds[local_ts][1]):
                        violating_variables.append(local_ts)

            symbole = (self.collocator.state_symbols +
                       self.collocator.unknown_input_trajectories)
            for symb in symbole:
                if symb in self.bounds.keys():
                    idx = symbole.index(symb)
                    feld = free[idx*self.collocator.num_collocation_nodes:
                                (idx+1)*self.collocator.num_collocation_nodes]
                    if (np.any(feld < self.bounds[symb][0])
                        or np.any(feld > self.bounds[symb][1])):
                        violating_variables.append(symb)

            # check that initial guesses for unknown parameters are within
            startidx = len(symbole) * self.collocator.num_collocation_nodes
            for symb in self.collocator.unknown_parameters:
                if symb in self.bounds.keys():
                    idx = self.collocator.unknown_parameters.index(symb)
                    if (free[startidx+idx] < self.bounds[symb][0]
                        or free[startidx+idx] > self.bounds[symb][1]):
                        violating_variables.append(symb)

            if len(violating_variables) > 0:
                msg = (f'The initial guesses for {violating_variables} are in '
                       f'conflict with their bounds.')
                raise ValueError(msg)

        else:
            pass

    def _generate_constraint_bound_arrays(self):

        # The default is that all constraints associated with the provided
        # equations of motion are equality constraints.
        low_con_bounds = np.zeros(self.num_constraints)
        upp_con_bounds = np.zeros(self.num_constraints)

        # If the user provides bounds for the equations of motion, process
        # them.
        if self.eom_bounds is not None:
            N = self.collocator.num_collocation_nodes
            for eom_idx, bnds in self.eom_bounds.items():
                low_con_bounds[eom_idx*(N - 1):(eom_idx + 1)*(N - 1)] = bnds[0]
                upp_con_bounds[eom_idx*(N - 1):(eom_idx + 1)*(N - 1)] = bnds[1]

        self._low_con_bounds = low_con_bounds
        self._upp_con_bounds = upp_con_bounds

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
        constraints_val : ndarray, shape(M*(N - 1) + o, )
            The value of the constraint function.

        Notes
        =====

        - N : number of collocation nodes
        - M : number of equations of motion
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
        jac_col_idxs : ndarray, shape(M*(N - 1) + o, )
            The column indices for the non-zero values in the Jacobian.

        Notes
        =====

        - N : number of collocation nodes
        - M : number of equations of motion
        - n : number of unknown state trajectories
        - q : number of unknown input trajectories
        - r : number of unknown parameters
        - s : number of unknown time intervals
        - o : number of instance constraints

        """
        return (self.con_jac_rows, self.con_jac_cols)

    def jacobian(self, free):
        """Returns the non-zero values of the Jacobian of the constraint
        function.

        Returns
        =======
        jac_vals : ndarray, shape((2*n + q + r + s)*(M*(N - 1)) + o, )
            Non-zero Jacobian values in triplet format.

        """
        return self.con_jac(free)

    def intermediate(self, *args):
        """This method is called at every optimization iteration. Not for pubic
        use."""
        self.obj_value.append(args[2])

    @_optional_plt_dep
    def plot_trajectories(self, vector, axes=None, show_bounds=False):
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
        show_bounds : bool, optional
            If True, the bounds will be plotted in the plot of the respective
            trajectory.

        Returns
        =======
        axes : ndarray of AxesSubplot
            A matplotlib axes with the state and input trajectories plotted.

        Notes
        =====

        - N : number of collocation nodes
        - M : number of equations of motion
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
                try:
                    trajectories = np.vstack(
                        (trajectories,
                        self.collocator.known_trajectory_map[knw_sym]))
                except ValueError:
                    trajectories = np.vstack(
                        (trajectories,
                        self.collocator.known_trajectory_map[knw_sym](vector)))

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

            if self.bounds is not None and show_bounds:
                if symbol in self.bounds.keys():
                    ax.axhline(self.bounds[symbol][0], color='C1', lw=1.0,
                               linestyle='--')
                    ax.axhline(self.bounds[symbol][1], color='C1', lw=1.0,
                               linestyle='--')
        ax.set_xlabel('Time')
        axes[0].set_title('State Trajectories')
        if (self.collocator.num_unknown_input_trajectories +
            self.collocator.num_known_input_trajectories) > 0:
            axes[self.collocator.num_states].set_title('Input Trajectories')
        return axes

    @_optional_plt_dep
    def plot_constraint_violations(self, vector, axes=None, subplots=False):
        """Returns an axis with the state constraint violations plotted versus
        node number and the instance constraints as a bar graph.

        Parameters
        ==========
        vector : ndarray, (n*N + q*N + r + s, )
            The initial guess, solution, or any other vector that is in the
            canonical form.
        axes : ndarray of AxesSubplot, optional.
            If given, it is the user's responsibility to provide the correct
            number of axes.
        subplots : boolean, optional.
            If True, the equations of motion will be plotted in a separate plot
            for each equation of motion. The default is False. If a user wants
            to provide the axes, it is recommended to run once without
            providing axes, to see how many are needed.

        Returns
        =======
        axes : ndarray of AxesSubplot
            A matplotlib axes with the constraint violations plotted. If the
            uses gives at least two axis, the method will tell the user how
            many are needed, unless the correct amount is given.

        Notes
        =====

        - N : number of collocation nodes
        - M : number of equations of motion
        - n : number of unknown state trajectories
        - q : number of unknown input trajectories
        - r : number of unknown parameters
        - s : number of unknown time intervals

        """

        bars_per_plot = None
        rotation = -45

        if subplots:
            figsize = 1.25
        else:
            figsize = 1.75

        if not isinstance(figsize, float):
            raise ValueError('figsize given must be a float.')

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
        eom_violations = con_violations[:self.collocator.num_eom*(N - 1)]
        instance_violations = con_violations[len(eom_violations):]
        eom_violations = eom_violations.reshape((self.collocator.num_eom,
                                                 N - 1))
        # TODO : figure out a way to plot the inequality constraint violations
        # don't plot inequality
        if self.eom_bounds is not None:
            for k, v in self.eom_bounds.items():
                eom_violations[k] = np.nan

        con_nodes = range(1, self.collocator.num_collocation_nodes)

        if axes is None:
            if subplots is False or self.collocator.num_eom == 1:
                num_eom_plots = 1
            else:
                num_eom_plots = self.collocator.num_eom

            fig, axes = plt.subplots(num_eom_plots + num_plots, 1,
                                     figsize=(6.4, figsize*(num_eom_plots +
                                                            num_plots)),
                                     layout='constrained')

        else:
            num_eom_plots = len(axes) - num_plots

        axes = np.asarray(axes).ravel()

        if subplots is False or self.collocator.num_eom == 1:
            axes[0].plot(con_nodes, eom_violations.T)
            axes[0].set_title('Constraint violations')
            axes[0].set_xlabel('Node Number')
            axes[0].set_ylabel('EoM violation')

        else:
            for i in range(self.collocator.num_eom):
                if ((self.eom_bounds is not None) and
                    (i in self.eom_bounds.keys())):  # don't plot if inequality
                    axes[i].plot(con_nodes, np.nan*np.ones_like(con_nodes))
                    axes[i].set_ylabel(f'Eq. {str(i+1)} \n not shown',
                                       fontsize=9)
                else:
                    axes[i].plot(con_nodes, eom_violations[i])
                    axes[i].set_ylabel(f'Eq. {str(i+1)} \n violation',
                                       fontsize=9)
                if i < self.collocator.num_eom - 1:
                    axes[i].set_xticklabels([])
            axes[num_eom_plots-1].set_xlabel('Node Number')
            axes[0].set_title('Constraint violations')

        if self.collocator.instance_constraints is not None:
            # reduce the instance constraints to 2 digits after the decimal
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

                    elif isinstance(a, sm.Float):
                        exp1 = exp1.subs(a, round(a, 2))
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
                axes[i+num_eom_plots].bar(
                    range(num_ticks), inst_viol,
                    tick_label=[sm.latex(s, mode='inline') for s in
                                inst_constr], width=width)
                axes[i+num_eom_plots].set_ylabel('Instance')
                axes[i+num_eom_plots].set_xticklabels(
                    axes[i+num_eom_plots].get_xticklabels(), rotation=rotation)

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

    @_optional_plt_dep
    def _plot_jacobian_sparsity(self, ax=None):
        # TODO : Make this a public method.
        # %%
        # Visualize the sparseness of the Jacobian for the non-linear programming
        # problem.

        # TODO : Figure out how to not depend on scipy if possible.
        from scipy.sparse import coo_matrix
        jac_vals = self.jacobian(np.ones(self.num_free))
        row_idxs, col_idxs = self.jacobianstructure()
        jacobian_matrix = coo_matrix((jac_vals, (row_idxs, col_idxs)))
        if ax is None:
            fig, ax = plt.subplots()
            ax.spy(jacobian_matrix)
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
        - M : number of equations of motion
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

    def time_vector(self, solution=None, start_time=0.0):
        """Returns the time array.

        Parameters
        ==========
        solution : ndarray, shape(n*N + q*N + r + s,), optional
            Solution to to problem; required if the time interval is variable.
        start_time : float, optional
            Initial time; default is ``0.0``.

        Returns
        =======
        time_vector : ndarray, shape(num_collocation_nodes,)
            The array of time instances.

        """
        t0 = start_time
        N = self.collocator.num_collocation_nodes

        if self.collocator._variable_duration:
            if solution is None:
                msg = 'Solution vector must be provided for variable duration.'
                raise ValueError(msg)
            else:
                h = solution[-1]

            if h <= 0.0:
                msg = 'Time interval must be strictly greater than zero.'
                raise ValueError(msg)
            elif t0 >= h*(N - 1):
                msg = 'Start time must be less than the final time.'
                raise ValueError(msg)
        else:
            h = self.collocator.node_time_interval

        return np.linspace(t0, t0 + h*(N - 1), num=N)


class ConstraintCollocator(object):
    """This class is responsible for generating the constraint function and the
    sparse Jacobian of the constraint function using direct collocation methods
    for a non-linear programming problem where the essential constraints are
    defined from the equations of motion of the system.

    Notes
    =====

    - N : number of collocation nodes
    - M : number of equations of motion
    - n : number of states
    - m : number of input trajectories
    - q : number of unknown input trajectories
    - r : number of unknown parameters
    - s : number of unknown time intervals (0 or 1 if fixed duration or
      variable duration)
    - o : number of instance constraints
    - nN + qN + r + s : number of free variables
    - M(N - 1) + o : number of constraints

    Some of the attributes are explained in more detail under Parameters below.

    It is best to treat ``ConstraintCollocator`` as immutable, changing
    attributes after initialization will inevitably fail.

    """
    def __init__(self, equations_of_motion, state_symbols,
                 num_collocation_nodes, node_time_interval,
                 known_parameter_map={}, known_trajectory_map={},
                 instance_constraints=None, time_symbol=None, tmp_dir=None,
                 integration_method='backward euler', parallel=False,
                 show_compile_output=False, backend='cython'):
        """Instantiates a ConstraintCollocator object.

        Parameters
        ==========
        equations_of_motion : sympy.Matrix, shape(M, 1)
            A column matrix of SymPy expressions defining the right hand side
            of the equations of motion when the left hand side is zero, i.e.
            ``0 = f(x'(t), x(t), u(t), p)``. These should be in first order
            form but not necessairly explicit. They can be ordinary
            differential equations or differential algebraic equations.
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
            ndarrays of floats of ``shape(N,)`` or functions that generate
            ndarrays of floats given the free optimization vector as an input.
            Any time varying parameters in the equations of motion not provided
            in this dictionary will become free trajectories optimization
            variables. If solving a variable duration problem, note that the
            values here are fixed at each node and will not scale with a
            varying time interval.
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
            extremely large number of operations. Only available with the
            ``'cython'`` backend.
        show_compile_output : boolean, optional
            If True, STDOUT and STDERR of the Cython compilation call will be
            shown. Only available with the ``'cython'`` backend.
        backend : string, optional
            Backend used to generate the numerical functions, either
            ``'cython'`` (default) or ``'numpy'``.

        """
        self._eom = equations_of_motion

        if time_symbol is not None:
            self._time_symbol = time_symbol
            me.dynamicsymbols._t = time_symbol
        else:
            self._time_symbol = me.dynamicsymbols._t

        self._state_symbols = tuple(state_symbols)
        if len(self.state_symbols) != len(set(self.state_symbols)):
            raise ValueError('State symbols must be unique.')

        # TODO : Check that for every derivative of time in eom, there is a
        # state variable in state_symbols.

        if backend not in ['cython', 'numpy']:
            raise ValueError('backend must be either "cython" or "numpy".')

        self._state_derivative_symbols = tuple([s.diff(self.time_symbol) for
                                               s in state_symbols])

        self._num_collocation_nodes = num_collocation_nodes

        if isinstance(node_time_interval, sm.Symbol):
            self._time_interval_symbol = node_time_interval
            self._variable_duration = True
        else:
            self._time_interval_symbol = sm.Symbol('h_opty', real=True)
            self._variable_duration = False
        self._node_time_interval = node_time_interval

        self._known_parameter_map = known_parameter_map
        self._known_trajectory_map = known_trajectory_map

        self._instance_constraints = instance_constraints

        self._num_constraints = self.num_eom*(num_collocation_nodes - 1)

        self._tmp_dir = tmp_dir
        self._parallel = parallel
        self._show_compile_output = show_compile_output
        self._backend = backend

        self._sort_parameters()
        self._sort_trajectories()
        self._num_free = ((self.num_states +
                           self.num_unknown_input_trajectories) *
                          self.num_collocation_nodes +
                          self.num_unknown_parameters +
                          int(self._variable_duration))
        self._check_known_trajectories()

        self.integration_method = integration_method

        self._discrete_symbols()
        self._discretize_eom()

        if instance_constraints is not None:
            self._num_instance_constraints = len(instance_constraints)
            self._num_constraints += self.num_instance_constraints
            self._identify_functions_in_instance_constraints()
            self._find_closest_free_index()
            self.eval_instance_constraints = self._instance_constraints_func()
            self.eval_instance_constraints_jacobian_values = \
                self._instance_constraints_jacobian_values_func()
        else:
            self._num_instance_constraints = 0

    @property
    def current_discrete_specified_symbols(self):
        """
        The symbols for the current discrete specified inputs.
        Type: tuple

        """
        return self._current_discrete_specified_symbols

    @property
    def current_discrete_state_symbols(self):
        """
        The symbols for the current discrete states.
        Type: n-tuple
        """
        return self._current_discrete_state_symbols

    @property
    def current_known_discrete_specified_symbols(self):
        """
        The symbols for the current discrete specified inputs.
        Type: tuple
        """
        return self._current_known_discrete_specified_symbols

    @property
    def current_unknown_discrete_specified_symbols(self):
        """
        The symbols for the current unknown discrete specified inputs.
        Type: tuple
        """
        return self._current_unknown_discrete_specified_symbols

    @property
    def discrete_eom(self):
        """
        Discretized equations of motion. Depending on the integration method
        used.
        Type: sympy.Matrix, shape(M, 1)
        """
        return self._discrete_eom

    @property
    def eom(self):
        """
        The equations of motion used.
        Type: sympy.Matrix, shape(M, 1)
        """
        return self._eom

    @property
    def input_trajectories(self):
        """
        known_input_trajectories + unknown_input_trajectories.
        Type: tuple
        """
        return self._input_trajectories

    @property
    def instance_constraints(self):
        """
        The instance constraints used in the optimization.
        Type: o-tuple
        """
        return self._instance_constraints

    @property
    def integration_method(self):
        """
        The integration method used. Presently, ``backward euler`` and
        ``midpoint`` are supported.
        Type: str
        """
        return self._integration_method

    @property
    def known_input_trajectories(self):
        """
        The known input trajectories symbols.
        Type: tuple
        """
        return self._known_input_trajectories

    @property
    def known_parameters(self):
        """
        The symbols of the known parameters in the problem.
        Type: tuple
        """
        return self._known_parameters

    @property
    def known_parameter_map(self):
        """
        A mapping of known parameters to their values.
        Type: dict
        """
        return self._known_parameter_map

    @property
    def known_trajectory_map(self):
        """
        A mapping of known trajectories to their values.
        Type: dict
        """
        return self._known_trajectory_map

    @property
    def known_trajectory_symbols(self):
        """
        The known trajectory symbols.
        Type: (m-q)-tuple
        """
        return self._known_trajectory_symbols

    @property
    def next_known_discrete_specified_symbols(self):
        """
        The symbols for the next discrete specified inputs.
        Type: tuple
        """
        return self._next_known_discrete_specified_symbols

    @property
    def next_discrete_specified_symbols(self):
        """
        The symbols for the next discrete specified inputs.
        Type: tuple
        """
        return self._next_discrete_specified_symbols

    @property
    def next_discrete_state_symbols(self):
        """
        The symbols for the next discrete states.
        Type: n-tuple
        """
        return self._next_discrete_state_symbols

    @property
    def next_unknown_discrete_specified_symbols(self):
        """
        The symbols for the next unknown discrete specified inputs.
        Type: tuple
        """
        return self._next_unknown_discrete_specified_symbols

    @property
    def node_time_interval(self):
        """
        The time interval between collocation nodes. float if the interval is
        fixed, ``sympy.Symbol`` if the interval is variable.
        Type: float or sympy.Symbol
        """
        return self._node_time_interval

    @property
    def num_collocation_nodes(self):
        """
        Number of times spaced evenly between the initial and final time of
        the optimization
        Type: int
        """
        return self._num_collocation_nodes

    @property
    def num_constraints(self):
        """
        The number of constraints = (num_collection_nodes-1)*num_states +
        len(instance_constraints).
        Type: int
        """
        return self._num_constraints

    @property
    def num_eom(self):
        """
        Number of equations in the equations of motion.
        Type: int
        """
        return self.eom.shape[0]

    @property
    def num_free(self):
        """
        Number of variables to be optimized = n*N + q*N + r + s.
        Type: int
        """
        return self._num_free

    @property
    def num_input_trajectories(self):
        """
        The number of input trajectories = len(input_trajectories).
        Type: int
        """
        return self._num_input_trajectories

    @property
    def num_instance_constraints(self):
        """
        The number of instance constraints = len(instance_constraints).
        Type: int
        """
        return self._num_instance_constraints

    @property
    def num_known_input_trajectories(self):
        """
        The number of known trajectories = len(known_trajectory_symbols).
        Type: int
        """
        return self._num_known_input_trajectories

    @property
    def num_parameters(self):
        """
        The number of parameters = len(parameters).
        Type: int
        """
        return self._num_parameters

    @property
    def num_known_parameters(self):
        """
        The number of known parameters = len(known_parameters).
        Type: int
        """
        return self._num_known_parameters

    @property
    def num_states(self):
        """
        The number of states = len(state_symbols) = n.
        Type: int
        """
        return len(self.state_symbols)

    @property
    def num_unknown_input_trajectories(self):
        """
        The number of unknown input trajectories =
        len(unknown_input_trajectories).
        Type: int
        """
        return self._num_unknown_input_trajectories

    @property
    def num_unknown_parameters(self):
        """
        The number of unknown parameters = r.
        Type: int
        """
        return self._num_unknown_parameters

    @property
    def parameters(self):
        """
        known_parameters + unknown_parameters.
        Type: tuple
        """
        return self._parameters

    @property
    def parallel(self):
        """
        Whether to use parallel processing or not.
        Type: bool
        """
        return self._parallel

    @property
    def previous_discrete_state_symbols(self):
        """
        The symbols for the previous discrete states.
        Type: n-tuple
        """
        return self._previous_discrete_state_symbols

    @property
    def show_compile_output(self):
        """
        Whether to show the compile output or not.
        Type: bool
        """
        return self._show_compile_output

    @property
    def state_derivative_symbols(self):
        """
        symbols for the time derivatives of the states.
        Type: n-tuple
        """
        return self._state_derivative_symbols

    @property
    def state_symbols(self):
        """
        The symbols for the states.
        Type: n-tuple
        """
        return self._state_symbols

    @property
    def time_interval_symbol(self):
        """
        sympy.Symbol if the time interval is variable, float if the time
        interval is fixed.
        Type: sympy.Symbol or float
        """
        return self._time_interval_symbol

    @property
    def time_symbol(self):
        """
        The symbol used to represent time, usually `t`.
        Type: sympy.Symbol
        """
        return self._time_symbol

    @property
    def tmp_dir(self):
        """
        The temporary directory used to store files generated.
        Type: str
        """
        return self._tmp_dir

    @property
    def unknown_input_trajectories(self):
        """
        The unknown input trajectories symbols.
        Type: q-tuple
        """
        return self._unknown_input_trajectories

    @property
    def unknown_parameters(self):
        """
        The unknown parameters in the problem, in the sequence in which they
        appear in the solution of the optimization.
        Type: r-tuple
        """
        return self._unknown_parameters

    @integration_method.setter
    def integration_method(self, method):
        """The method can be ``'backward euler'`` or ``'midpoint'``."""
        if method not in ['backward euler', 'midpoint']:
            msg = ("{} is not a valid integration method.")
            raise ValueError(msg.format(method))
        else:
            self._integration_method = method

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

        self._known_parameters = res[0]
        self._num_known_parameters = res[1]
        self._unknown_parameters = res[2]
        self._num_unknown_parameters = res[3]

        self._parameters = res[0] + res[2]
        self._num_parameters = len(self.parameters)

    def _check_known_trajectories(self):
        """Raises and error if the known trajectories are not the correct
        length."""

        N = self.num_collocation_nodes

        for k, v in self.known_trajectory_map.items():
            if isinstance(v, type(lambda x: x)):
                v = v(np.ones(self.num_free))
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
        # non_states may contain the a Function('?')(state)
        implicit_funcs_of_time = []
        # NOTE : Do we support implicit functions of more than one state
        # variable?
        self.implicit_derivative_repl = {}
        for thing in non_states.copy():
            if thing.args == (self.time_symbol,):  # explicit functions of time
                pass
            else:  # is implicit function of time
                implicit_funcs_of_time.append(thing)
                # TODO : Pass on assumptions from thing?
                deriv_var = sm.Function('d' + thing.name + '_d' +
                                        thing.args[0].name)(self.time_symbol)
                #non_states.add(deriv_var)
                self.implicit_derivative_repl[thing.diff(thing.args[0])] = deriv_var
        if sm.Matrix(list(non_states)).has(sm.Derivative):
            msg = ('Too few state variables provided for state time '
                   'derivatives found in equations of motion.')
            raise ValueError(msg)

        res = self._parse_inputs(non_states,
                                 self.known_trajectory_map.keys())

        self._known_input_trajectories = res[0]
        self._num_known_input_trajectories = res[1]
        self._unknown_input_trajectories = res[2]
        self._num_unknown_input_trajectories = res[3]

        self._input_trajectories = res[0] + res[2]
        self._num_input_trajectories = len(self.input_trajectories)

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
        self._previous_discrete_state_symbols = \
            tuple([sm.Symbol(f.__class__.__name__ + 'p', real=True)
                   for f in self.state_symbols])
        self._current_discrete_state_symbols = \
            tuple([sm.Symbol(f.__class__.__name__ + 'i', real=True)
                   for f in self.state_symbols])
        self._next_discrete_state_symbols = \
            tuple([sm.Symbol(f.__class__.__name__ + 'n', real=True)
                   for f in self.state_symbols])

        # The current and next known input trajectories.
        self._current_known_discrete_specified_symbols = \
            tuple([sm.Symbol(f.__class__.__name__ + 'i', real=True)
                   for f in self.known_input_trajectories])
        self._next_known_discrete_specified_symbols = \
            tuple([sm.Symbol(f.__class__.__name__ + 'n', real=True)
                   for f in self.known_input_trajectories])

        # The current and next unknown input trajectories.
        self._current_unknown_discrete_specified_symbols = \
            tuple([sm.Symbol(f.__class__.__name__ + 'i', real=True)
                   for f in self.unknown_input_trajectories])
        self._next_unknown_discrete_specified_symbols = \
            tuple([sm.Symbol(f.__class__.__name__ + 'n', real=True)
                   for f in self.unknown_input_trajectories])

        self._current_discrete_specified_symbols = (
            self.current_known_discrete_specified_symbols +
            self.current_unknown_discrete_specified_symbols)
        self._next_discrete_specified_symbols = (
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

            self._discrete_eom = me.msubs(self.eom, deriv_sub, func_sub)

        elif self.integration_method == 'midpoint':

            xdot_sub = {d: (n - i) / h for d, i, n in zip(xd, xi, xn)}
            x_sub = {d: (i + n) / 2 for d, i, n in zip(x, xi, xn)}
            u_sub = {d: (i + n) / 2 for d, i, n in zip(u, ui, un)}
            self._discrete_eom = me.msubs(self.eom, xdot_sub, x_sub, u_sub)

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

        N = self.num_collocation_nodes
        n = self.num_states

        def determine_free_index(time_index, state):
            if state in self.state_symbols:
                state_index = self.state_symbols.index(state)
                return time_index + state_index*N
            elif state in self.unknown_input_trajectories:
                state_index = self.unknown_input_trajectories.index(state)
                return time_index + n*N + state_index*N

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

        num_eom_constraints = self.num_eom*(self.num_collocation_nodes - 1)

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
             con_2_N, ..., con_M_2, ..., con_M_N]

        for M equatiosn of motion and N-1 constraints at the time points.

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

        if self._backend == 'cython':
            logging.info('Compiling the constraint function.')
            f = ufuncify_matrix(args, self.discrete_eom,
                                const=constant_syms + (h_sym,),
                                tmp_dir=self.tmp_dir, parallel=self.parallel,
                                show_compile_output=self.show_compile_output)
        elif self._backend == 'numpy':
            f = lambdify_matrix(args, self.discrete_eom)

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
            constraints : ndarray, shape(M*(N-1),)
                The array of constraints from t = 2, ..., N.
                [con_1_2, ..., con_1_N, con_2_2, ...,
                 con_2_N, ..., con_M_2, ..., con_M_N]

            """

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
            result = np.empty((num_constraints, self.num_eom))

            return f(result, *args).T.flatten()

        self._multi_arg_con_func = constraints

    def jacobian_indices(self):
        """Returns the row and column indices for the non-zero values in the
        constraint Jacobian.

        Returns
        -------
        jac_row_idxs : ndarray, shape(2*n + q + r + s,)
            The row indices for the non-zero values in the Jacobian.
        jac_col_idxs : ndarray, shape(M + o,)
            The column indices for the non-zero values in the Jacobian.

        """

        N = self.num_collocation_nodes
        M = self.num_eom
        n = self.num_states

        num_constraint_nodes = N - 1

        if self.integration_method == 'backward euler':

            num_partials = M*(2*n + self.num_unknown_input_trajectories +
                              self.num_unknown_parameters +
                              int(self._variable_duration))

        elif self.integration_method == 'midpoint':

            num_partials = M*(2*n + 2*self.num_unknown_input_trajectories +
                              self.num_unknown_parameters +
                              int(self._variable_duration))

        num_non_zero_values = num_constraint_nodes * num_partials

        if self.instance_constraints is not None:
            ins_row_idxs, ins_col_idxs = \
                self._instance_constraints_jacobian_indices()
            num_non_zero_values += len(ins_row_idxs)

        jac_row_idxs = np.empty(num_non_zero_values, dtype=int)
        jac_col_idxs = np.empty(num_non_zero_values, dtype=int)

        # TODO : Go over the remainder of this function and comments to make
        # sure it is correct for the change to allow M equations of motion != n
        # states.

        """
        M : number of equations of motion
        N : number of collocation nodes
        Q = N-1
        P = N-2

        The symbolic derivative matrix for a single constraint node follows
        these patterns:

        Backward Euler
        --------------
        i: ith, b: ith-1 (b = before)

        This Jacobian calculates the partials at the ith node::

                 d eom(xi, xb, ui, p, h)  in R^M
            Ji = -----------------------
                 d [xi, xb, ui, p, h]     in R^(2*n + q + r + 1)

        For example:

        x1i = the first state at the ith constraint node
        uqi = the qth input at the ith constraint node

        Walk through i = 1 to N and calculate Ji with the correct input values
        that follow this pattern:

        [x1] [x11, ..., xn1, x10, ..., xn0, u11, .., uq1, p1, ..., pr, h]
        [. ]
        [xi] [x1i, ..., xni, x1b, ..., xnb, u1i, .., uqi, p1, ..., pr, h]
        [. ]
        [xQ] [x1Q, ..., xnQ, x1P, ..., xnP, u1Q, .., uqQ, p1, ..., pr, h]

        Midpoint
        --------
        i: ith, f: ith+1 (f = following)

        uqn = the q input at the ith+1 constraint node
        n: also number of states (confusing)

        This Jacobian calculates the partials at the ith node::

                 d eom(xi, xf, ui, uf, p, h)  in R^M
            Ji = ---------------------------
                 d [xi, xf, ui, uf, p, h]     in R^(2*n + 2*q + r + 1)

        Walk through i = 0 to Q and calculate Ji with the correct input values
        that follow this pattern:

        [x0] [x10, ..., xn0, x1f, ..., xnf, u10, .., uq0, u1f, ..., uqf, p1, ..., pp, h]
        [. ]
        [xi] [x1i, ..., xni, x1f, ..., xnf, u1i, .., uqi, u1f, ..., uqf, p1, ..., pp, h]
        [. ]
        [xP] [x1P, ..., xnP, x1Q, ..., xnQ, u1P, .., uqP, u1Q, ..., uqQ, p1, ..., pp, h]

        Each of these Jacobian matrices are evaulated at N-1 constraint nodes
        and then the 3D matrix is flattened into a 1D array. The backward euler
        uses nodes 1 <= i <= N-1 and the midpoint uses 0 <= i <= N - 2 for any
        given Jacobian evaluation. So the flattened arrays looks like:

        Backward Euler
        --------------

        i=1  eom1  | [x11, ..., xn1, x10, ..., xn0, u11, .., uq1, p1, ..., pr, h,
             eom2  |  x11, ..., xn1, x10, ..., xn0, u11, .., uq1, p1, ..., pr, h,
             ...   |  ...,
             eomM  |  x11, ..., xn1, x10, ..., xn0, u11, .., uq1, p1, ..., pr, h,
        i=2  eom1  |  x12, ..., xn2, x11, ..., xn1, u12, .., uq2, p1, ..., pr, h,
             eom2  |  x12, ..., xn2, x11, ..., xn1, u12, .., uq2, p1, ..., pr, h,
             ...   |  ...,
             eomM  |  x12, ..., xn2, x11, ..., xn1, u12, .., uq2, p1, ..., pr, h,
                   |  ...,
        i=Q  eom1  |  x1Q, ..., xnQ, x1P, ..., xnP, u1Q, .., uqQ, p1, ..., pr, h,
             eom2  |  x1Q, ..., xnQ, x1P, ..., xnP, u1Q, .., uqQ, p1, ..., pr, h,
             ...   |  ...,
             eomM  |  x1Q, ..., xnQ, x1P, ..., xnP, u1Q, .., uqQ, p1, ..., pr, h]

        Midpoint
        --------

        i=0   eom1  | [x10, ..., xn0, x11, ..., xn1, u10, .., uq0, u11, .., uq1, p1, ..., pr, h,
              eom2  |  x10, ..., xn0, x11, ..., xn1, u10, .., uq0, u11, .., uq1, p1, ..., pr, h,
              ...   |  ...,
              eomM  |  x10, ..., xn0, x11, ..., xn1, u10, .., uq0, u11, .., uq1, p1, ..., pr, h,
        i=1   eom1  |  x11, ..., xn1, x12, ..., xn2, u11, .., uq1, u12, .., uq2, p1, ..., pr, h,
              eom2  |  x11, ..., xn1, x12, ..., xn2, u11, .., uq1, u12, .., uq2, p1, ..., pr, h,
              ...   |  ...,
              eomM  |  x11, ..., xn1, x12, ..., xn2, u11, .., uq1, u12, .., uq2, p1, ..., pr, h,
              ...   |  ...,
        i=P   eom1  |  x1P, ..., xnP, x1Q, ..., xnQ, u1P, .., uqP, u1Q, .., uqQ, p1, ..., pr, h,
              eom2  |  x1P, ..., xnP, x1Q, ..., xnQ, u1P, .., uqP, u1Q, .., uqQ, p1, ..., pr, h,
              ...   |  ...,
              eomM  |  x1P, ..., xnP, x1Q, ..., xnQ, u1P, .., uqP, u1Q, .., uqQ, p1, ..., pr, h]

        These two arrays contain of the non-zero values of the sparse
        Jacobian[#]_.

        .. [#] Some of the partials can be equal to zero and could be
            excluded from the array. These could be a significant number.

        Now we need to generate the triplet format indices of the full sparse
        Jacobian for each one of the entries in these arrays. The format of the
        Jacobian matrix is:

        Backward Euler
        --------------

                [x10, ..., x1Q, ..., xn0, ..., xnQ, u10, ..., u1Q, ..., uq0, ..., uqQ, p1, ..., pr, h]
        [eom10]
        [eom11]
        [...]
        [eom1Q]
        [...]
        [eomM0]
        [eomM1]
        [...]
        [eomMQ]

        Midpoint
        --------

               [x10, ..., x1N-1, ..., xn0, ..., xnN-1, u10, ..., u1N-1, ..., uq0, ..., uqN-1, p1, ..., pr, h]
        [eom10]
        [eom11]
        [...]
        [eom1P]
        [...]
        [eomM0]
        [eomM1]
        [...]
        [eomMP]

        """
        for i in range(num_constraint_nodes):

            # N : number of collocation nodes
            # M : number of equations of motion
            # n : number of states
            # m : number of input trajectories
            # p : number of parameters
            # q : number of unknown input trajectories
            # r : number of unknown parameters
            # s : number of unknown time intervals

            # the eoms repeat every N - 1 constraints
            # row_idxs = [0*(N - 1), 1*(N - 1),  2*(N - 1), ..., M*(N - 1)]

            # This gives the Jacobian row indices matching the ith constraint
            # node for each state. ith corresponds to the loop indice.
            row_idxs = [j*(num_constraint_nodes) + i for j in range(M)]

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
        if self._backend == 'cython':
            symbolic_partials = _forward_jacobian(discrete_eom_matrix,
                                                  wrt_matrix.T)
        elif self._backend == 'numpy':
            symbolic_partials = discrete_eom_matrix.jacobian(wrt_matrix.T)

        # This generates a numerical function that evaluates the matrix of
        # partial derivatives. This function returns the non-zero elements
        # needed to build the sparse constraint Jacobian.
        if self._backend == 'cython':
            logging.info('Compiling the Jacobian function.')
            eval_partials = ufuncify_matrix(args, symbolic_partials,
                                            const=constant_syms + (h_sym,),
                                            tmp_dir=self.tmp_dir,
                                            parallel=self.parallel)
        elif self._backend == 'numpy':
            eval_partials = lambdify_matrix(args, symbolic_partials)

        if (isinstance(symbolic_partials, tuple) and
            len(symbolic_partials) == 2):
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
                backward euler: shape((N - 1) * M * (2*n + q + r + s),)
                midpoint: shape((N - 1) * M * (2*n + 2*q + r + s),)
                The values of the non-zero entries of the constraints
                Jacobian. These correspond to the triplet formatted indices
                returned from jacobian_indices.

            Notes
            -----
            - N : number of collocation nodes
            - M : number of equations of motion
            - n : number of states
            - m : number of input trajectories
            - p : number of parameters
            - q : number of unknown input trajectories
            - r : number of unknown parameters
            - s : number of unknown time intervals
            - n*(N - 1) : number of constraints

            """
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
            elif (len(specified_values.shape) == 1 and
                  specified_values.size != 0):
                si = specified_values[current_start:current_stop]
                args += [si]
                if self.integration_method == 'midpoint':
                    sn = specified_values[adjacent_start:adjacent_stop]
                    args += [sn]

            args += [c for c in parameter_values]
            args += [interval_value]

            # backward euler: shape(N - 1, M, 2*n + q + r)
            # midpoint: shape(N - 1, M, 2*n + 2*q + r)
            non_zero_derivatives = eval_partials(result, *args)

            return non_zero_derivatives.ravel()

        self._multi_arg_con_jac_func = constraints_jacobian

    @staticmethod
    def _merge_fixed_free(syms, fixed, free, typ, free_op_vals):
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
                if isinstance(fixed[s], type(lambda x: x)):
                    merged.append(fixed[s](free_op_vals))
                else:
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
        typ : string
            ``'con'`` or ``'jac'`` for constraints or Jacobian of the
            constraints, respectively.

        Returns
        -------
        func : function
            A function which returns constraint values given the system's free
            optimization variables, has signature f(free), where free is
            ndarray, shape(nN + qN + r + s, ).

        """

        def constraints(free):
            """
            Parameters
            ==========
            free : ndarray, shape(nN + qN + r + s, )

            """
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
                                                   free_specified, 'traj',
                                                   free)

            all_constants = self._merge_fixed_free(self.parameters,
                                                   self.known_parameter_map,
                                                   free_constants, 'par', free)

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
        new_doc = ('{}Parameters\n----------\n'
                   'free : ndarray, shape()\n\nReturns\n{}')
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
