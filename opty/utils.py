#!/usr/bin/env python

import os
import sys
import shutil
import tempfile
import subprocess
import importlib
from functools import wraps
import warnings
import logging
from distutils.ccompiler import new_compiler
from distutils.errors import CompileError
from distutils.sysconfig import customize_compiler
from collections import Counter
from timeit import default_timer as timer
import logging
import locale

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from sympy.utilities.iterables import numbered_symbols
from sympy.printing.c import C99CodePrinter
plt = sm.external.import_module('matplotlib.pyplot',
                                import_kwargs={'fromlist': ['']},
                                catch=(RuntimeError,))

__all__ = [
    'parse_free',
    'create_objective_function',
]

logger = logging.getLogger(__name__)


def _coo_matrix(jac_vals, row_idxs, col_idxs):
    """Quick and dirty replacement for scipy.sparse.coo_matrix."""
    arr = np.zeros((np.max(row_idxs + 1), np.max(col_idxs + 1)),
                   dtype=jac_vals.dtype)
    for v, r, c in zip(jac_vals, row_idxs, col_idxs):
        arr[r, c] = v
    return arr


class MathJaxRepr():
    """Returns an object wrapping a SymPy expression that will render output in
    MathJax LaTeX syntax compatible with rendering in HTML in Sphinx
    Gallery."""
    def __init__(self, expr):
        self.expr = expr

    def _repr_html_(self):
        r"""
        Creates ``$$\begin{equation}...\end{equation}$$``
        """
        return me.vlatex(self.expr, mode='equation', itex=True)


class OptyC99CodePrinter(C99CodePrinter):
    """Printer that appends an underscore to all C variable names, to minimize
    clashes with variables declared in headers we link against."""

    # TODO : Add sanitizer to remove invalid characters in symbol names, for
    # example deal with latex symbol names e.g. I_{zz}.

    def _print_Symbol(self, expr):
        name = super()._print_Symbol(expr)
        return name + '_'

    def _print_Function(self, expr):
        name = super()._print_Function(expr)
        return name + '_'


def ccode(expr, assign_to=None, **settings):
    """Mimics SymPy's ccode, but uses our printer."""
    return OptyC99CodePrinter(settings).doprint(expr, assign_to)


def _forward_jacobian(expr, wrt):

    def add_to_cache(node):
        if node in expr_to_replacement_cache:
            replacement_symbol = expr_to_replacement_cache[node]
            return (replacement_symbol,
                    replacement_to_reduced_expr_cache[replacement_symbol])
        elif node in replacement_to_reduced_expr_cache:
            return node, replacement_to_reduced_expr_cache[node]
        elif isinstance(node, sm.Tuple):
            return None, None
        elif not node.free_symbols:
            return node, node

        replacement_symbol = replacement_symbols.__next__()
        replaced_subexpr = node.xreplace(expr_to_replacement_cache)
        replacement_to_reduced_expr_cache[replacement_symbol] = replaced_subexpr
        expr_to_replacement_cache[node] = replacement_symbol
        return replacement_symbol, replaced_subexpr

    if not isinstance(expr, sm.ImmutableDenseMatrix):
        msg = (
            'The forward Jacobian differentiation algorithm can only be used '
            'to differentiate a single matrix expression at a time.'
        )
        raise NotImplementedError(msg)
    elif expr.shape[1] != 1:
        msg = 'Can only compute the Jacobian for column matrices.'
        raise NotImplementedError(msg)
    elif not isinstance(wrt, sm.ImmutableDenseMatrix) or wrt.shape[1] != 1:
        msg = (
            'The forward Jacobian differentiation algorithm can compute '
            'Jacobians with respect to column matrices.'
        )
        raise NotImplementedError

    # TODO : Ideally a replacement symbol would somehow carry the appropriate
    # assumptions that derive from the subexpression it replaces. I add
    # real=True here, as we assume all symbols and functions of time are real
    # in opty.
    replacement_symbols = numbered_symbols(
        prefix='z',
        cls=sm.Symbol,
        exclude=expr.free_symbols,
        real=True,
    )

    expr_to_replacement_cache = {}
    replacement_to_reduced_expr_cache = {}

    logger.info('Adding expression nodes to cache...')
    start = timer()
    replacements, reduced_exprs = sm.cse(expr.args[2], replacement_symbols)
    for replacement_symbol, reduced_subexpr in replacements:
        replaced_subexpr = reduced_subexpr.xreplace(expr_to_replacement_cache)
        replacement_to_reduced_expr_cache[replacement_symbol] = replaced_subexpr
        expr_to_replacement_cache[reduced_subexpr] = replacement_symbol
        for node in sm.postorder_traversal(reduced_subexpr):
            _ = add_to_cache(node)
    for reduced_expr in reduced_exprs:
        for node in reduced_expr:
            _ = add_to_cache(node)
    finish = timer()
    logger.info(f'Completed in {finish - start:.2f}s')

    reduced_matrix = sm.ImmutableDenseMatrix(reduced_exprs).xreplace(
        expr_to_replacement_cache)
    replacements = list(replacement_to_reduced_expr_cache.items())

    partial_derivative_mapping = {}
    absolute_derivative_mapping = {}
    for i, wrt_symbol in enumerate(wrt.args[2]):
        absolute_derivative = [sm.S.Zero] * len(wrt)
        absolute_derivative[i] = sm.S.One
        absolute_derivative_mapping[wrt_symbol] = sm.ImmutableDenseMatrix(
            [absolute_derivative])

    logger.info('Differentiating expression nodes...')
    start = timer()
    zeros = sm.ImmutableDenseMatrix.zeros(1, len(wrt))
    for symbol, subexpr in replacements:
        free_symbols = subexpr.free_symbols
        absolute_derivative = zeros
        for free_symbol in free_symbols:
            replacement_symbol, partial_derivative = add_to_cache(
                subexpr.diff(free_symbol))
            absolute_derivative += (
                partial_derivative *
                absolute_derivative_mapping.get(free_symbol, zeros))
        absolute_derivative_mapping[symbol] = sm.ImmutableDenseMatrix(
            [[add_to_cache(a)[0] for a in absolute_derivative]])

    replaced_jacobian = sm.ImmutableDenseMatrix.vstack(*[
        absolute_derivative_mapping[e] for e in reduced_matrix])
    finish = timer()
    logger.info(f'Completed in {finish - start:.2f}s')

    logger.info('Determining required replacements...')
    start = timer()
    required_replacement_symbols = set()
    stack = [entry for entry in replaced_jacobian if entry.free_symbols]
    while stack:
        entry = stack.pop()
        if entry in required_replacement_symbols or entry in wrt:
            continue
        children = list(
            replacement_to_reduced_expr_cache.get(entry, entry).free_symbols)
        for child in children:
            if child not in required_replacement_symbols and child not in wrt:
                stack.append(child)
        required_replacement_symbols.add(entry)
    finish = timer()
    logger.info(f'Completed in {finish - start:.2f}s')

    required_replacements_dense = {
        replacement_symbol: replaced_subexpr
        for replacement_symbol, replaced_subexpr in
        replacement_to_reduced_expr_cache.items()
        if replacement_symbol in required_replacement_symbols
    }

    counter = Counter(replaced_jacobian.free_symbols)
    for replaced_subexpr in required_replacements_dense.values():
        counter.update(replaced_subexpr.free_symbols)

    logger.info('Substituting required replacements...')
    required_replacements = {}
    unrequired_replacements = {}
    for replacement_symbol, replaced_subexpr in required_replacements_dense.items():
        if (isinstance(replaced_subexpr, sm.Symbol) or
            counter[replacement_symbol] == 1):
            unrequired_replacements[replacement_symbol] = replaced_subexpr.xreplace(unrequired_replacements)
        else:
            required_replacements[replacement_symbol] = replaced_subexpr.xreplace(unrequired_replacements)
    finish = timer()
    logger.info(f'Completed in {finish - start:.2f}s')

    return (list(required_replacements.items()),
            [replaced_jacobian.xreplace(unrequired_replacements)])

from sympy.utilities._compilation import compilation as pycompilation


def building_docs():
    if 'READTHEDOCS' in os.environ:
        return True
    elif 'SPHINX' in os.environ:
        return True
    else:
        return False


def _optional_plt_dep(func):
    """Decorator that aborts function/method call if matplotlib is not
    installed."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if plt is None:
            raise ImportError('Install matplotlib for plotting features.')
        else:
            return func(*args, **kwargs)
    return wrapper


def state_derivatives(states):
    """Returns functions of time which represent the time derivatives of the
    states."""
    return [state.diff() for state in states]


def f_minus_ma(mass_matrix, forcing_vector, states):
    """Returns Fr + Fr* from the mass_matrix and forcing vector."""

    xdot = sm.Matrix(state_derivatives(states))

    return mass_matrix * xdot - forcing_vector


def parse_free(free, n, q, N, variable_duration=False):
    """Parses the free parameters vector and returns it's components.

    Parameters
    ----------
    free : ndarray, shape(n*N + q*N + r + s)
        The free parameters of the system.
    n : integer
        The number of states.
    q : integer
        The number of free specified inputs.
    N : integer
        The number of time steps.
    variable_duration : boolean, optional
        If True, the last value in ``free`` is the node time interval and it
        will be returned.

    Returns
    -------
    states : ndarray, shape(n, N)
        The array of n states through N time steps.
    specified_values : ndarray, shape(q, N) or shape(N,), or None
        The array of q specified inputs through N time steps.
    constant_values : ndarray, shape(r,)
        The array of r constants.
    time_interval : float
        The time between collocation nodes. Only returned if
        ``variable_duration`` is ``True``.

    """

    len_states = n * N
    len_specified = q * N

    free_states = free[:len_states].reshape((n, N))

    if q == 0:
        free_specified = None
    else:
        free_specified = free[len_states:len_states + len_specified]
        if q > 1:
            free_specified = free_specified.reshape((q, N))

    if variable_duration:
        free_time_interval = free[-1]
        free_constants = free[len_states + len_specified:-1]
        return free_states, free_specified, free_constants, free_time_interval
    else:
        free_constants = free[len_states + len_specified:]
        return free_states, free_specified, free_constants


def create_objective_function(objective, state_symbols,
                              unknown_input_trajectories, unknown_parameters,
                              num_collocation_nodes, node_time_interval,
                              integration_method="backward euler",
                              time_symbol=me.dynamicsymbols._t):
    """Creates function to evaluate the objective and objective gradient.

    Parameters
    ----------
    objective : Expr
        Symbolic objective function to be minimized. It should solely depend on
        the states, unknown inputs, and unknown parameters. Any known inputs or
        parameters should be substituted beforehand. Additionally, the
        objective function can contain non-nested indefinite integrals of time,
        e.g. ``Integral(f(t)**2, t)``.
    state_symbols : iterable of Function()(t)
        An iterable containing all ``n`` of the SymPy functions of time which
        represent the states in the equations of motion.
    unknown_input_trajectories : iterable of Function()(t)
        An iterable containing all ``q`` of the SymPy functions of time which
        represent the unknown input trajectories in the equations of motion.
    unknown_parameters : iterable of Symbol
        An iterable containing all ``r`` of the SymPy symbols which represent
        the unknown parameters in the equations of motion.
    num_collocation_nodes : int
        Number of collocation nodes, i.e. the number of time steps.
    node_time_interval : float
        The value of the time interval. The default is 1.0, as this term only
        appears in the objective function as a scaling factor.
    integration_method : str, optional
        The method used to integrate the system. The default is ``"backward
        euler"``.
    time_symbol : Symbol, optional
        If not supplied, ``sympy.physics.mechanics.dynamicsymbols._t`` is used.

    """
    def lambdify_function(expr, multiplication_array, take_sum):
        if take_sum:
            def integration_function(x):
                return node_time_interval * np.sum(x * multiplication_array)
        else:
            def integration_function(x):
                return node_time_interval * x * multiplication_array
        return sm.lambdify(
            (states, inputs, params), expr,
            modules=[{int_placeholder.name: integration_function}, "numpy"],
            cse=True)

    def parse_expr(expr, in_integral=False):
        if not expr.args:
            return expr
        if isinstance(expr, sm.Integral):
            if in_integral:
                msg = "Nested integrals are not supported."
                raise NotImplementedError(msg)
            if expr.limits != ((time_symbol,),):
                raise NotImplementedError(
                    "Only indefinite integrals of time are supported.")
            return int_placeholder(parse_expr(expr.function, True))
        return expr.func(*(parse_expr(arg) for arg in expr.args))

    # Parse function arguments
    states = sm.ImmutableMatrix(state_symbols)
    inputs = sm.ImmutableMatrix(sort_sympy(unknown_input_trajectories))
    params = sm.ImmutableMatrix(sort_sympy(unknown_parameters))
    if states.shape[1] > 1 or inputs.shape[1] > 1 or params.shape[1] > 1:
        raise ValueError(
            'The state, input, and unknown symbols must be column matrices.')
    n, q = states.shape[0], inputs.shape[0]
    N = num_collocation_nodes
    i_idx, r_idx = n * N, (n + q) * N

    # Compute analytical gradient of the objective function
    objective_grad = sm.ImmutableMatrix([objective]).jacobian(
        states[:] + inputs[:] + params[:])

    # Replace the integral with a custom function
    int_placeholder = sm.Function("_IntegralFunction")
    objective = parse_expr(objective)
    objective_grad = tuple(parse_expr(objective_grad))

    # Replace zeros with an array of zeros, otherwise lambdify will return a
    # scalar zero instead of an array of zeros.
    objective_grad = (tuple(np.zeros(N)
                            if grad == 0 else grad
                            for grad in objective_grad[:n + q]) +
                      tuple(objective_grad[n + q:]))

    # Define evaluation functions based on the integration method
    if integration_method == "backward euler":
        obj_expr_eval = lambdify_function(
            objective, np.hstack((0, np.ones(N - 1))), True)
        obj_grad_time_expr_eval = lambdify_function(
            objective_grad[:n + q], np.hstack((0, np.ones(N - 1))), False)
        obj_grad_param_expr_eval = lambdify_function(
            objective_grad[n + q:], np.hstack((0, np.ones(N - 1))), True)

        def obj(free):
            states = free[:i_idx].reshape((n, N))
            inputs = free[i_idx:r_idx].reshape((q, N))
            return obj_expr_eval(states, inputs, free[r_idx:])

        def obj_grad(free):
            states = free[:i_idx].reshape((n, N))
            inputs = free[i_idx:r_idx].reshape((q, N))
            return np.hstack((
                *obj_grad_time_expr_eval(states, inputs, free[r_idx:]),
                obj_grad_param_expr_eval(states, inputs, free[r_idx:])
            ))

    elif integration_method == "midpoint":
        obj_expr_eval = lambdify_function(
            objective, np.ones(N - 1), True)
        obj_grad_time_expr_eval = lambdify_function(
            objective_grad[:n + q], np.hstack((0.5, np.ones(N - 2), 0.5)),
            False)
        obj_grad_param_expr_eval = lambdify_function(
            objective_grad[n + q:], np.ones(N - 1), True)

        def obj(free):
            states = free[:i_idx].reshape((n, N))
            states_mid = 0.5 * (states[:, :-1] + states[:, 1:])
            inputs = free[i_idx:r_idx].reshape((q, N))
            inputs_mid = 0.5 * (inputs[:, :-1] + inputs[:, 1:])
            return obj_expr_eval(states_mid, inputs_mid, free[r_idx:])

        def obj_grad(free):
            states = free[:i_idx].reshape((n, N))
            states_mid = 0.5 * (states[:, :-1] + states[:, 1:])
            inputs = free[i_idx:r_idx].reshape((q, N))
            inputs_mid = 0.5 * (inputs[:, :-1] + inputs[:, 1:])
            return np.hstack((
                *obj_grad_time_expr_eval(states, inputs, free[r_idx:]),
                obj_grad_param_expr_eval(states_mid, inputs_mid, free[r_idx:])
            ))

    else:
        raise NotImplementedError(
            f"Integration method '{integration_method}' is not implemented.")

    return obj, obj_grad


def sort_sympy(seq):
    """Returns a sorted list of the symbols."""
    seq = list(seq)
    try:  # symbols
        seq.sort(key=lambda x: x.name)
    except AttributeError:  # functions
        seq.sort(key=lambda x: x.__class__.__name__)
    return seq


_c_template = """\
{win_math_def}
#include <math.h>

void {routine_name}(double matrix[{matrix_output_size}],
{input_args})
{{
{eval_code}
}}
"""

_cython_template = """\
# cython: language_level=3

import numpy as np
from cython.parallel import prange
cimport numpy as np
cimport cython

cdef extern void c_{routine_name} "{routine_name}" (double matrix[{matrix_output_size}], {input_args}){head_gil}

@cython.boundscheck(False)
@cython.wraparound(False)
def {routine_name}_loop(matrix,
{numpy_typed_input_args}):

    cdef double[:, ::1] matrix_memview = matrix
    {memory_views}

    cdef int n = matrix.shape[0]

    cdef int i

    for i in {loop_sig}:
        c_{routine_name}(&matrix_memview[i, 0],
{indexed_input_args})

    return matrix.reshape(n, {num_rows}, {num_cols})
"""

_setup_template = """\
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extension = Extension(name="{file_prefix}",
                      sources=["{file_prefix}.pyx",
                               "{file_prefix}_c.c"],
                      extra_compile_args=[{compile_args}],
                      extra_link_args=[{link_args}],
                      include_dirs=[numpy.get_include()])

setup(name="{routine_name}",
      ext_modules=cythonize([extension]))
"""

module_counter = 0


def openmp_installed():
    """Returns true if openmp is installed, false if not.

    Modified from:
    https://stackoverflow.com/questions/16549893/programatically-testing-for-openmp-support-from-a-python-setup-script

    """
    tmpdir = tempfile.mkdtemp(".opty_openmp_check")
    curdir = os.getcwd()
    os.chdir(tmpdir)

    filename = r'test.c'
    contents = r"""\
#include <omp.h>
#include <stdio.h>
int main(void) {
    #pragma omp parallel
    printf("Hello from thread %d, nthreads %d\n",
           omp_get_thread_num(), omp_get_num_threads());
}"""

    with open(filename, 'w') as f:
        f.write(contents)

    ccompiler = new_compiler()
    customize_compiler(ccompiler)
    try:
        # .compile() should return ['test.o'] on linux
        if sys.platform == "win32":
            ccompiler.compile([filename], extra_postargs=['/openmp'])
        elif sys.platform == "darwin":
            ccompiler.compile([filename], extra_postargs=['-Xclang', '-fopenmp'])
        else:
            ccompiler.compile([filename], extra_postargs=['-fopenmp'])
        exit = True
    except CompileError:
        exit = False
    finally:
        os.chdir(curdir)
        # NOTE : I can't figure out how to get rmtree to work on Windows, so I
        # don't delete the directory on Windows.
        if sys.platform != "win32":
            shutil.rmtree(tmpdir)

    return exit


def lambdify_matrix(args, expr):
    """Returns a function that evaluates a matrix of expressions using NumPy
    and broadcasts over an 3D array.

    Parameters
    ----------
    args : iterable of Symbol or Function(Symbol)
        A list of all symbols in ``expr`` in the desired order for the output
        function.
    expr : Matrix, shape(m, p)
        A matrix of expressions.

    Returns
    -------
    function
        Function takes the form ``result = f(store, arg1, arg2, ...)`` where:

            - ``store`` : ndarray, shape(n, m*p)
            - ``result`` : ndarray, shape(n, m, p), reshaped ``store``
            - ``argn`` : ndarray shape(n,) or float

    """
    eval_single_mat = sm.lambdify(args, expr, modules='numpy', cse=True)

    # TODO : this is terribly computationally costly but there is not a clean
    # way to do this using appropriate broadcasting via lambdify. See:
    # https://github.com/sympy/sympy/issues/27632
    def loop_function(result, *num_args):
        n = result.shape[0]
        for i in range(n):
            array_num_args = [a if isinstance(a, float) else a[i]
                              for a in num_args]
            result[i] = eval_single_mat(*array_num_args).flatten().squeeze()
        return result.reshape(n, expr.shape[0], expr.shape[1])

    return loop_function


def ufuncify_matrix(args, expr, const=None, tmp_dir=None, parallel=False,
                    show_compile_output=False):
    """Returns a function that evaluates a matrix of expressions in a tight
    loop.

    Parameters
    ----------
    args : iterable of sympy.Symbol
        A list of all symbols in expr in the desired order for the output
        function.
    expr : sympy.Matrix
        A matrix of expressions.
    const : tuple, optional
        This should include any of the symbols in args that should be
        constant with respect to the loop.
    tmp_dir : string, optional
        The path to a directory in which to store the generated files. If
        None then the files will be not be retained after the function is
        compiled.
    parallel : boolean, optional
        If True and openmp is installed, the generated code will be
        parallelized across threads. This is only useful when expr are
        extremely large.
    show_compile_output : boolean, optional
        If True, STDOUT and STDERR of the Cython compilation call will be
        shown.

    """

    # TODO : This is my first ever global variable in Python. It'd probably
    # be better if this was a class attribute of a Ufuncifier class. And I'm
    # not sure if this current version counts sequentially.
    global module_counter

    if hasattr(expr, 'shape'):
        num_rows = expr.shape[0]
        num_cols = expr.shape[1]
    else:
        num_rows = expr[1][0].shape[0]
        num_cols = expr[1][0].shape[1]

    matrix_size = num_rows * num_cols

    file_prefix_base = 'ufuncify_matrix'
    file_prefix = '{}_{}'.format(file_prefix_base, module_counter)

    if tmp_dir is None:
        codedir = tempfile.mkdtemp(".opty_ufuncify_compile")
    else:
        codedir = os.path.abspath(tmp_dir)

    if not os.path.exists(codedir):
        os.makedirs(codedir)

    taken = False

    while not taken:
        try:
            open(os.path.join(codedir, file_prefix + '.pyx'), 'r')
        except IOError:
            taken = True
        else:
            file_prefix = '{}_{}'.format(file_prefix_base, module_counter)
            module_counter += 1

    d = {'routine_name': 'eval_matrix',
         'file_prefix': file_prefix,
         'matrix_output_size': matrix_size,
         'num_rows': num_rows,
         'num_cols': num_cols}

    if parallel:
        if openmp_installed():
            openmp = True
        else:
            openmp = False
            msg = ('openmp is not installed or not working properly, request '
                   'for parallel execution ignored.')
            warnings.warn(msg)

    if parallel and openmp:
        d['loop_sig'] = "prange(n, nogil=True)"
        d['head_gil'] = " nogil"
        if sys.platform == "win32":
            d['compile_args'] = r"'\openmp'"
            d['link_args'] = ""
        elif sys.platform == "darwin":
            d['compile_args'] = "'-Xclang', '-fopenmp'"
            d['link_args'] = "'-Xclang', '-fopenmp'"
        else:
            d['compile_args'] = "'-fopenmp'"
            d['link_args'] = "'-fopenmp'"
    else:
        d['loop_sig'] = "range(n)"
        d['head_gil'] = ""
        d['compile_args'] = ""
        d['link_args'] = ""

    matrix_sym = sm.MatrixSymbol('matrix', num_rows, num_cols)

    if isinstance(expr, tuple) and len(expr) == 2:
        sub_exprs, simple_mat = expr
    else:
        sub_exprs, simple_mat = sm.cse(expr, sm.numbered_symbols('z_'))

    sub_expr_code = '\n'.join(['double ' + ccode(sub_expr[1], sub_expr[0])
                               for sub_expr in sub_exprs])

    matrix_code = ccode(simple_mat[0], matrix_sym)

    d['eval_code'] = '    ' + '\n    '.join((sub_expr_code + '\n' +
                                             matrix_code).split('\n'))

    c_indent = len('void {routine_name}('.format(**d))
    c_arg_spacer = ',\n' + ' ' * c_indent

    input_args = ['double {}'.format(ccode(a)) for a in args]
    d['input_args'] = ' '*c_indent + c_arg_spacer.join(input_args)

    cython_input_args = []
    indexed_input_args = []
    memory_views = []
    for a in args:
        if const is not None and a in const:
            typ = 'double'
            idexy = '{}'
            cython_input_args.append('{} {}'.format(typ, ccode(a)))
        else:
            idexy = '{}_memview[i]'
            memview = 'cdef double[::1] {}_memview = {}'
            memory_views.append(memview.format(ccode(a), ccode(a)))
            cython_input_args.append('{}'.format(ccode(a)))

        indexed_input_args.append(idexy.format(ccode(a)))

    cython_indent = len('def {routine_name}_loop('.format(**d))
    cython_arg_spacer = ',\n' + ' '*cython_indent

    loop_indent = len('        {routine_name}('.format(**d))
    loop_spacer = ',\n' + ' '*loop_indent

    d['numpy_typed_input_args'] = (' '*cython_indent +
                                   cython_arg_spacer.join(cython_input_args))

    d['indexed_input_args'] = (' '*loop_indent +
                               loop_spacer.join(indexed_input_args))

    d['memory_views'] = '\n    '.join(memory_views)

    if os.name == 'nt':
        d['win_math_def'] = '#define _USE_MATH_DEFINES'
    else:
        d['win_math_def'] = ''

    files = {}
    files[d['file_prefix'] + '_c.c'] = _c_template.format(**d)
    files[d['file_prefix'] + '.pyx'] = _cython_template.format(**d)
    files[d['file_prefix'] + '_setup.py'] = _setup_template.format(**d)

    try:
        logger.info('opty:Compiling with sympy.utilities._compilation module.')
        sources = [
            (d['file_prefix'] + '_c.c', files[d['file_prefix'] + '_c.c']),
            (d['file_prefix'] + '.pyx', files[d['file_prefix'] + '.pyx']),
        ]

        options = []
        if parallel:
            options += ['-fopenmp']

        cython_module, info = pycompilation.compile_link_import_strings(
            sources,
            compile_kwargs={
                "std": 'c99',
                "include_dirs": [np.get_include()],
                'flags': options},
            link_kwargs={'flags': options},
            build_dir=codedir)
    except:
        raise RuntimeError("SymPy's compilation failed.")
        logging.info("opty:Compiling with Opty's compilation functions.")
        workingdir = os.getcwd()
        os.chdir(codedir)

        try:
            sys.path.append(codedir)
            for filename, code in files.items():
                with open(filename, 'w') as f:
                    f.write(code)
            cmd = [sys.executable, d['file_prefix'] + '_setup.py', 'build_ext',
                   '--inplace']
            subprocess.call(cmd, stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE)
            cython_module = importlib.import_module(d['file_prefix'])
            # NOTE : This may not always work on Windows (seems to be dependent
            # on how Python is invoked). There is explanation in
            # https://github.com/python/cpython/issues/105312 but it is not
            # crystal clear what the solution is.
            # device_encoding() takes: 0: stdin, 1: stdout, 2: stderr
            # device_encoding() always returns UTF-8 on Unix but will return
            # different encodings on Windows and only if it is "attached to a
            # terminal".
            # locale.getencoding() tries to guess the encoding
            if sys.platform == 'win32':
                try:  # Python >=  3.11
                    encoding = locale.getencoding()
                except AttributeError:  # Python < 3.11
                    encoding = locale.getlocale()[1]
            else:
                encoding = None
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      encoding=encoding)
            # On Windows this can raise a UnicodeDecodeError, but only in the
            # subprocess.
            except UnicodeDecodeError:
                stdout = 'STDOUT not captured, decoding error.'
                stderr = 'STDERR not captured, decoding error.'
            else:
                stdout = proc.stdout
                stderr = proc.stderr

            if show_compile_output:
                print(stdout)
                print(stderr)
            try:
                cython_module = importlib.import_module(d['file_prefix'])
            except ImportError as error:
                msg = ('Unable to import the compiled Cython module {}, '
                       'compilation likely failed. STDERR output from '
                       'compilation:\n{}')
                raise ImportError(msg.format(d['file_prefix'], stderr)) from error
        finally:
            module_counter += 1
            sys.path.remove(codedir)
            os.chdir(workingdir)
            if tmp_dir is None:
                # NOTE : I can't figure out how to get rmtree to work on
                # Windows, so I don't delete the directory on Windows.
                if sys.platform != "win32":
                    shutil.rmtree(codedir)

    return getattr(cython_module, d['routine_name'] + '_loop')


def controllable(a, b):
    """Returns true if the system is controllable and false if not.

    Parameters
    ----------
    a : array_like, shape(n,n)
        The state matrix.
    b : array_like, shape(n,r)
        The input matrix.

    Returns
    -------
    controllable : boolean

    """
    a = np.asmatrix(a)
    b = np.asmatrix(b)
    n = a.shape[0]
    controllability_matrix = []
    for i in range(n):
        controllability_matrix.append(a ** i * b)
    controllability_matrix = np.hstack(controllability_matrix)

    return np.linalg.matrix_rank(controllability_matrix) == n


def substitute_matrix(matrix, row_idxs, col_idxs, sub_matrix):
    """Returns the matrix with the values given by the row and column
    indices with those in the sub-matrix.

    Parameters
    ----------
    matrix : ndarray, shape(n, m)
        A matrix (i.e. 2D array).
    row_idxs : array_like, shape(p<=n,)
        The row indices which designate which entries should be replaced by
        the sub matrix entries.
    col_idxs : array_like, shape(q<=m,)
        The column indices which designate which entries should be replaced
        by the sub matrix entries.
    sub_matrix : ndarray, shape(p, q)
        A matrix of values to substitute into the specified rows and
        columns.

    Notes
    -----
    This makes a copy of the sub_matrix, so if it is large it may be slower
    than a more optimal implementation.

    Examples
    --------

    >>> a = np.zeros((3, 4))
    >>> sub = np.arange(4).reshape((2, 2))
    >>> substitute_matrix(a, [1, 2], [0, 2], sub)
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 2.,  0.,  3.,  0.]])

    """

    assert sub_matrix.shape == (len(row_idxs), len(col_idxs))

    row_idx_permutations = np.repeat(row_idxs, len(col_idxs))
    col_idx_permutations = np.array(list(col_idxs) * len(row_idxs))

    matrix[row_idx_permutations, col_idx_permutations] = sub_matrix.flatten()

    return matrix


def sum_of_sines(sigma, frequencies, time):
    """Returns a sum of sines centered at zero along with its first and
    second derivatives.

    Parameters
    ==========
    sigma : float
        The desired standard deviation of the series.
    frequencies : iterable of floats
        The frequencies of the sin curves to be included in the sum.
    time : array_like, shape(n,)
        The montonically increasing time vector.

    Returns
    =======
    sines : ndarray, shape(n,)
        A sum of sines.
    sines_prime : ndarray, shape(n,)
        The first derivative of the sum of sines.
    sines_double_prime : ndarray, shape(n,)
        The second derivative of the sum of sines.

    """

    phases = 2.0 * np.pi * np.random.ranf(len(frequencies))

    sines = np.zeros_like(time)
    sines_prime = np.zeros_like(time)
    sines_double_prime = np.zeros_like(time)

    amplitude = sigma / 2.0

    for w, p in zip(frequencies, phases):
        sines += amplitude * np.sin(w * time + p)
        sines_prime += amplitude * w * np.cos(w * time + p)
        sines_double_prime -= amplitude * w**2 * np.sin(w * time + p)

    return sines, sines_prime, sines_double_prime
