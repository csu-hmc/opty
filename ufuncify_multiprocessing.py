from multiprocessing import Pool

import sympy as sy
from sympy.utilities.autowrap import ufuncify


def my_func(tup):
    ufuncify(tup[0], tup[1])


if __name__ == '__main__':

    a_syms = sy.symbols('a:10')

    exprs = [a ** (n + 1) for n, a in enumerate(a_syms)]

    p = Pool(4)

    funcs = p.map(my_func, zip(a_syms, exprs))

    for f in funcs:
        print(f(5.0))
