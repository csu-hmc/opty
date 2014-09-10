#!/usr/bin/env python

import subprocess
import importlib

import numpy as np
import sympy as sy
from sympy.utilities.lambdify import implemented_function
from sympy.utilities.autowrap import autowrap

_c_template = """\
#include <math.h>
#include "{file_prefix}_h.h"

void {routine_name}({input_args}, double matrix[{matrix_output_size}])
{{
{eval_code}
}}
"""

_h_template = """\
void {routine_name}({input_args}, double matrix[{matrix_output_size}]);
"""

_cython_template = """\
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "{file_prefix}_h.h":
    void {routine_name}({input_args}, double matrix[{matrix_output_size}])

@cython.boundscheck(False)
@cython.wraparound(False)
def {routine_name}_loop({numpy_typed_input_args}, np.ndarray[np.double_t, ndim=2] matrix):

    cdef int n = matrix.shape[0]

    cdef int i

    for i in range(n):
        {routine_name}({indexed_input_args},
                       &matrix[i, 0])

    return matrix.reshape(n, {num_rows}, {num_cols})
"""

_setup_template = """\
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

extension = Extension(name="{file_prefix}",
                      sources=["{file_prefix}.pyx",
                               "{file_prefix}_c.c"],
                      include_dirs=[numpy.get_include()])

setup(name="{routine_name}",
      cmdclass={{'build_ext': build_ext}},
      ext_modules=[extension])
"""


def ufuncify_matrix(args, expr, cse=True):

    matrix_size = expr.shape[0] * expr.shape[1]

    d = {'routine_name': 'eval_matrix',
         'file_prefix': 'ufuncify_matrix',
         'matrix_output_size': matrix_size,
         'num_rows': expr.shape[0],
         'num_cols': expr.shape[1]}

    matrix_sym = sy.MatrixSymbol('matrix', expr.shape[0], expr.shape[1])

    sub_exprs, simple_mat = sy.cse(expr, sy.numbered_symbols('z_'))

    sub_expr_code = '\n'.join(['double ' + sy.ccode(sub_expr[1], sub_expr[0])
                               for sub_expr in sub_exprs])

    matrix_code = sy.ccode(simple_mat[0], matrix_sym)

    d['eval_code'] = '    ' + '\n    '.join((sub_expr_code + '\n' + matrix_code).split('\n'))

    c_indent = len('void {routine_name}('.format(**d))
    c_arg_spacer = ',\n' + ' ' * c_indent

    d['input_args'] = c_arg_spacer.join(['double {}'.format(a) for a in args])

    cython_indent = len('def {routine_name}_loop('.format(**d))
    cython_arg_spacer = ',\n' + ' ' * cython_indent

    d['numpy_typed_input_args'] = cython_arg_spacer.join(['np.ndarray[np.double_t, ndim=1] {}'.format(a) for a in args])

    d['indexed_input_args'] = ',\n'.join(['{}[i]'.format(a) for a in args])

    files = {}
    files[d['file_prefix'] + '_c.c'] = _c_template.format(**d)
    files[d['file_prefix'] + '_h.h'] = _h_template.format(**d)
    files[d['file_prefix'] + '.pyx'] = _cython_template.format(**d)
    files[d['file_prefix'] + '_setup.py'] = _setup_template.format(**d)

    for filename, code in files.items():
        with open(filename, 'w') as f:
            f.write(code)

    cmd = ['python', d['file_prefix'] + '_setup.py', 'build_ext', '--inplace']
    subprocess.call(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

    cython_module = importlib.import_module(d['file_prefix'])
    return getattr(cython_module, d['routine_name'] + '_loop')


"""This is a small rewrite of the ufuncify function in sympy so it supports
array arguments for all values."""


def ufuncify(args, expr, **kwargs):
    """
    Generates a binary ufunc-like lambda function for numpy arrays

    ``args``
        Either a Symbol or a tuple of symbols. Specifies the argument sequence
        for the ufunc-like function.

    ``expr``
        A SymPy expression that defines the element wise operation

    ``kwargs``
        Optional keyword arguments are forwarded to autowrap().

    The returned function can only act on one array at a time, as only the
    first argument accept arrays as input.

    .. Note:: a *proper* numpy ufunc is required to support broadcasting, type
       casting and more.  The function returned here, may not qualify for
       numpy's definition of a ufunc.  That why we use the term ufunc-like.

    References
    ==========
    [1] http://docs.scipy.org/doc/numpy/reference/ufuncs.html

    Examples
    ========

    >>> from sympy.utilities.autowrap import ufuncify
    >>> from sympy.abc import x, y
    >>> import numpy as np
    >>> f = ufuncify([x, y], y + x**2)
    >>> f([1, 2, 3], [2, 2, 2])
    [ 3.  6.  11.]
    >>> a = f(np.arange(5), 3 * np.ones(5))
    >>> isinstance(a, np.ndarray)
    True
    >>> print a
    [ 3. 4. 7. 12. 19.]

    """
    y = sy.IndexedBase(sy.Dummy('y')) # result array

    i = sy.Dummy('i', integer=True) # index of the array
    m = sy.Dummy('m', integer=True) # length of array (dimension)
    i = sy.Idx(i, m) # index of the array

    l = sy.Lambda(args, expr) # A lambda function that represents the inputs and outputs of the expression
    f = implemented_function('f', l) # maps an UndefinedFunction('f') to the lambda function

    if isinstance(args, sy.Symbol):
        args = [args]
    else:
        args = list(args)

    # For each of the args create an indexed version.
    indexed_args = [sy.IndexedBase(sy.Dummy(str(a))) for a in args]

    # ensure correct order of arguments
    kwargs['args'] = [y] + indexed_args + [m]
    args_with_indices = [a[i] for a in indexed_args]
    return autowrap(sy.Eq(y[i], f(*args_with_indices)), **kwargs)

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
