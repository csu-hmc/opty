"""This solves the simple pendulum swing up problem presented here:

    http://hmc.csuohio.edu/resources/human-motion-seminar-jan-23-2014

A simple pendulum is controlled by a torque at its joint. The goal is to
swing the pendulum from its rest equilibrium to a target angle by minimizing
the energy used to do so.

"""

from collections import OrderedDict

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

from direct_collocation import ConstraintCollocator, Problem

duration = 3.0
num_nodes = 100
interval_value = duration / (num_nodes - 1)

I, m, g, d, t = sym.symbols('I, m, g, d, t')
theta, omega, T = [f(t) for f in sym.symbols('theta, omega, T',
                                             cls=sym.Function)]
state_symbols = (theta, omega)
constant_symbols = (I, m, g, d)
specified_symbols = (T,)

eom = sym.Matrix([theta.diff() - omega,
                  I * omega.diff() + m * g * d * sym.sin(theta) - T])

par_map = OrderedDict(zip(constant_symbols, (1.0, 1.0, 9.81, 1.0)))


def obj(free):
    T = free[2 * num_nodes:]
    return np.sum(T**2)


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[2 * num_nodes:] = 2.0 * interval_value * free[2 * num_nodes:]
    return grad

theta, omega = sym.symbols('theta, omega', cls=sym.Function)
instance_constraints = (theta(0.0),
                        theta(duration) - sym.pi,
                        omega(0.0),
                        omega(duration))

con_col = ConstraintCollocator(eom, state_symbols, num_nodes,
                               interval_value, known_parameter_map=par_map,
                               instance_constraints=instance_constraints)

prob = Problem(con_col.num_free,
               con_col.num_constraints,
               obj,
               obj_grad,
               con_col.generate_constraint_function(),
               con_col.generate_jacobian_function(),
               con_col.jacobian_indices)

# Known solution as initial guess.
initial_guess = np.random.randn(con_col.num_free)

# Find the optimal solution.
solution, info = prob.solve(initial_guess)

time = np.linspace(0.0, duration, num=num_nodes)

angle = solution[:num_nodes]
rate = solution[num_nodes:2 * num_nodes]
torque = solution[2 * num_nodes:]
con_violations = prob.con(solution)

fig, axes = plt.subplots(5)

axes[0].plot(time, angle)
axes[1].plot(time, rate)
axes[2].plot(time, torque)
axes[3].plot(con_violations)
axes[4].plot(prob.obj_value)

plt.show()
