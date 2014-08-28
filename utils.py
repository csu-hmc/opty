#!/usr/bin/env python

import numpy as np
import sympy as sy
from sympy.utilities.lambdify import implemented_function
from sympy.utilities.autowrap import autowrap

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
