#!/usr/bin/env python

import os
import sys
import shutil
import tempfile
import subprocess
import importlib
from functools import wraps
import warnings

import numpy as np
import sympy as sm
plt = sm.external.import_module('matplotlib.pyplot',
                                __import__kwargs={'fromlist': ['']},
                                catch=(RuntimeError,))


def building_docs():
    try:
        os.eviron['READTHEDOCS']
    except:
        try:
            os.environ['SPHINX']
        except:
            pass
        else:
            return True
    else:
        return True


def _optional_plt_dep(func):
    """Decorator that aborts function/method call if matplotlib is not
    installed."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if plt is None:
            raise ImportError('Install matplotlib for plotting features.')
        else:
            func(*args, **kwargs)
    return wrapper


def state_derivatives(states):
    """Returns functions of time which represent the time derivatives of the
    states."""
    return [state.diff() for state in states]


def f_minus_ma(mass_matrix, forcing_vector, states):
    """Returns Fr + Fr* from the mass_matrix and forcing vector."""

    xdot = sm.Matrix(state_derivatives(states))

    return mass_matrix * xdot - forcing_vector


def parse_free(free, n, r, N):
    """Parses the free parameters vector and returns it's components.

    free : ndarray, shape(n * N + m * M + q)
        The free parameters of the system.
    n : integer
        The number of states.
    r : integer
        The number of free specified inputs.
    N : integer
        The number of time steps.

    Returns
    -------
    states : ndarray, shape(n, N)
        The array of n states through N time steps.
    specified_values : ndarray, shape(r, N) or shape(N,), or None
        The array of r specified inputs through N time steps.
    constant_values : ndarray, shape(q,)
        The array of q constants.

    """

    len_states = n * N
    len_specified = r * N

    free_states = free[:len_states].reshape((n, N))

    if r == 0:
        free_specified = None
    else:
        free_specified = free[len_states:len_states + len_specified]
        if r > 1:
            free_specified = free_specified.reshape((r, N))

    free_constants = free[len_states + len_specified:]

    return free_states, free_specified, free_constants


_c_template = """\
#include <math.h>
#include "{file_prefix}_h.h"

void {routine_name}(double matrix[{matrix_output_size}], {input_args})
{{
{eval_code}
}}
"""

_h_template = """\
void {routine_name}(double matrix[{matrix_output_size}], {input_args});
"""

_cython_template = """\
import numpy as np
from cython.parallel import prange
cimport numpy as np
cimport cython

cdef extern from "{file_prefix}_h.h"{head_gil}:
    void {routine_name}(double matrix[{matrix_output_size}], {input_args})

@cython.boundscheck(False)
@cython.wraparound(False)
def {routine_name}_loop(np.ndarray[np.double_t, ndim=2] matrix, {numpy_typed_input_args}):

    cdef int n = matrix.shape[0]

    cdef int i

    for i in {loop_sig}:
        {routine_name}(&matrix[i, 0], {indexed_input_args})

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
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    filename = r'test.c'
    contents = r"""\
#include <omp.h>
#include <stdio.h>
int main() {
    #pragma omp parallel
    printf("Hello from thread %d, nthreads %d\n",
           omp_get_thread_num(), omp_get_num_threads());
}"""

    with open(filename, 'w') as f:
        f.write(contents)

    compiler = os.getenv('CC', 'cc')

    exit = 1
    try:
        with open(os.devnull, 'w') as fnull:
            exit = subprocess.call([compiler, '-fopenmp', filename],
                                   stdout=fnull, stderr=fnull)
    except:
        raise
    finally:  # cleanup even if compilation fails
        os.chdir(curdir)
        shutil.rmtree(tmpdir)

    return True if exit == 0 else False


def ufuncify_matrix(args, expr, const=None, tmp_dir=None, parallel=False):
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

    """

    # TODO : This is my first ever global variable in Python. It'd probably
    # be better if this was a class attribute of a Ufuncifier class. And I'm
    # not sure if this current version counts sequentially.
    global module_counter

    matrix_size = expr.shape[0] * expr.shape[1]

    file_prefix_base = 'ufuncify_matrix'
    file_prefix = '{}_{}'.format(file_prefix_base, module_counter)

    if tmp_dir is None:
        codedir = tempfile.mkdtemp(".ufuncify_compile")
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
         'num_rows': expr.shape[0],
         'num_cols': expr.shape[1]}

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
        d['compile_args'] = "'-fopenmp'"
        d['link_args'] = "'-fopenmp'"
    else:
        d['loop_sig'] = "range(n)"
        d['head_gil'] = ""
        d['compile_args'] = ""
        d['link_args'] = ""

    matrix_sym = sm.MatrixSymbol('matrix', expr.shape[0], expr.shape[1])

    sub_exprs, simple_mat = sm.cse(expr, sm.numbered_symbols('z_'))

    sub_expr_code = '\n'.join(['double ' + sm.ccode(sub_expr[1], sub_expr[0])
                               for sub_expr in sub_exprs])

    matrix_code = sm.ccode(simple_mat[0], matrix_sym)

    d['eval_code'] = '    ' + '\n    '.join((sub_expr_code + '\n' +
                                             matrix_code).split('\n'))

    c_indent = len('void {routine_name}('.format(**d))
    c_arg_spacer = ',\n' + ' ' * c_indent

    input_args = ['double {}'.format(sm.ccode(a)) for a in args]
    d['input_args'] = c_arg_spacer.join(input_args)

    cython_input_args = []
    indexed_input_args = []
    for a in args:
        if const is not None and a in const:
            typ = 'double'
            idexy = '{}'
        else:
            typ = 'np.ndarray[np.double_t, ndim=1]'
            idexy = '{}[i]'

        cython_input_args.append('{} {}'.format(typ, sm.ccode(a)))
        indexed_input_args.append(idexy.format(sm.ccode(a)))

    cython_indent = len('def {routine_name}_loop('.format(**d))
    cython_arg_spacer = ',\n' + ' ' * cython_indent

    d['numpy_typed_input_args'] = cython_arg_spacer.join(cython_input_args)

    d['indexed_input_args'] = ',\n'.join(indexed_input_args)

    files = {}
    files[d['file_prefix'] + '_c.c'] = _c_template.format(**d)
    files[d['file_prefix'] + '_h.h'] = _h_template.format(**d)
    files[d['file_prefix'] + '.pyx'] = _cython_template.format(**d)
    files[d['file_prefix'] + '_setup.py'] = _setup_template.format(**d)

    workingdir = os.getcwd()
    os.chdir(codedir)

    try:
        sys.path.append(codedir)
        for filename, code in files.items():
            with open(filename, 'w') as f:
                f.write(code)
        cmd = [sys.executable, d['file_prefix'] + '_setup.py', 'build_ext',
               '--inplace']
        subprocess.call(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        cython_module = importlib.import_module(d['file_prefix'])
    finally:
        module_counter += 1
        sys.path.remove(codedir)
        os.chdir(workingdir)
        if tmp_dir is None:
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
