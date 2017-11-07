"""This script generates equations of motion that have a very large number of
operations. It then creates a parallelized and non-parallelized version and
compares the evaluation time."""

import timeit

import numpy as np
import sympy as sm
from pydy.models import n_link_pendulum_on_cart
from opty.utils import ufuncify_matrix

sys = n_link_pendulum_on_cart(10, False, False)

c = np.random.random(int(1e7))

args = [c for i in range(len(list(sys.constants_symbols) +
                             list(sys.coordinates) + list(sys.speeds)))]

res = np.empty((int(1e7), 11))

udots = sm.Matrix([5.0 for u in sys.speeds])

expr = sys.eom_method.mass_matrix * udots - sys.eom_method.forcing

q_sub = {q: sm.Symbol(q.__class__.__name__) for q in sys.coordinates}

u_sub = {u: sm.Symbol(u.__class__.__name__) for u in sys.speeds}

expr = expr.subs(q_sub).subs(u_sub)

f_nonpar = ufuncify_matrix(list(sys.constants_symbols) + list(q_sub.values()) +
                           list(u_sub.values()), expr, parallel=False)

f_par = ufuncify_matrix(list(sys.constants_symbols) + list(q_sub.values()) +
                        list(u_sub.values()), expr, parallel=True)

start_time = timeit.default_timer()
f_nonpar(res, *args)
elapsed = timeit.default_timer() - start_time
print('Time for non-parallel: {}'.format(elapsed))

start_time = timeit.default_timer()
f_par(res, *args)
elapsed = timeit.default_timer() - start_time
print('Time for parallelized: {}'.format(elapsed))
