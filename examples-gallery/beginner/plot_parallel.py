"""
Parallel Evaluation of SymPy Matrices
=====================================

This script generates equations of motion that have a very large number of
operations. It then evaluates a parallelized and non-parallelized version and
compares the evaluation time. :py:class:`Problem` and
:py:class:`ConstraintCollocator` use :py:func:`ufuncify_matrix` in the
background to evaluate the NLP constraints and its Jacobian. For large
equations of motion a performance gain can be had if parallelized.

Objectives
----------

- Demonstrate the performance gain for parallel evaluation of SymPy expressions
  over a large set of inputs.

"""

import timeit

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from pydy.models import n_link_pendulum_on_cart
from opty.utils import ufuncify_matrix

# %%
# Generate equations of motion with a large number of operations.
sys = n_link_pendulum_on_cart(10, False, False)
udots = sm.Matrix(sys.speeds).diff()
expr = sys.eom_method.mass_matrix*udots - sys.eom_method.forcing
sm.count_ops(expr)

# %%
# Substitute symbols for all functions of time.
ud_sub = {u.diff(): sm.Symbol(u.__class__.__name__ + 'd') for u in sys.speeds}
q_sub = {q: sm.Symbol(q.__class__.__name__) for q in sys.coordinates}
u_sub = {u: sm.Symbol(u.__class__.__name__) for u in sys.speeds}
expr = me.msubs(expr, ud_sub, u_sub, q_sub)

# %%
# Create 4 million random values per unique symbol in the expression.
sym_args = (
    list(ud_sub.values()) +
    list(u_sub.values()) +
    list(q_sub.values()) +
    list(sys.constants_symbols)
)
num = int(4e6)
args = np.random.random((len(sym_args), num))
res = np.empty((num, len(u_sub)))
res.shape

# %%
# Evaluate the expressions for each of the 4 million inputs without
# parallelization.
f_nonpar = ufuncify_matrix(sym_args, expr, parallel=False)
start_time = timeit.default_timer()
f_nonpar(res, *args)
elapsed = timeit.default_timer() - start_time
print('Time for non-parallel: {}'.format(elapsed))

# %%
# Evaluate the expressions for each of the 4 million inputs with
# parallelization.
f_par = ufuncify_matrix(sym_args, expr, parallel=True)
start_time = timeit.default_timer()
f_par(res, *args)
elapsed = timeit.default_timer() - start_time
print('Time for parallelized: {}'.format(elapsed))
