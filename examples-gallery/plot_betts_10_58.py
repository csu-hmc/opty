"""
Heat Equation
=============

This is example 10.58 from Betts' book "Practical Methods for Optimal Control
Using Nonlinear Programming", 3rd edition, chapter 10: Test Problems.
It deals with the 'discretization' of a PDE.

**States**

- :math:`y_0, .....y_{10}, w` : state variables

**Specifieds**

- :math:`v, q_{00}, q_{11}` : control variables

"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem

# %%
# Equations of Motion.
#---------------------
t = me.dynamicsymbols._t
T = sm.symbols('T', cls=sm.Function)

q = list(me.dynamicsymbols(f'q:{10}'))
w = me.dynamicsymbols('w')
v, q00, q11 = me.dynamicsymbols('v q00 q11')

# %%
# parameters fom the example
qa = 0.2
gamma = 0.04
h = 10.0
delta = 1/9

# %%
# In addition to the 11 DEs for the 11 state variables, the book gives two
# algebraic equations. As opty needs exactly one equation per state variable,
# I solve the AEs for a state variable and insert into the DEs.

q01 = sm.solve(sm.Matrix([h*(q[0] - w) - 1/(2*delta)*(q[1] - q00)]), q[0])[q[0]]
q81 = sm.solve(sm.Matrix([1/(2*delta)*(q11 - q[-1])]), q[-1])[q[-1]]

# %%
uq = [q[i].diff(t) for i in range(10)]

eom = sm.Matrix([
    -uq[0] + 1/delta**2 * (q[1] - 2*q01 + q00),
    -uq[1] + 1/delta**2 * (q[2] - 2*q[1] + q01),
    *[-uq[i] - 1/delta**2 * (q[i+1] - 2*q[i] + q[i-1]) for i in range(2, 9)],
    -uq[-1] + 1/delta**2 * (q11 - 2*q[-1] + q81),
    -w.diff(t) + 1/gamma*(v - w),
])

# %%
# Optimization
# ------------
# I put the optimization into a function,because this makes it a bit easier
# to change parameters.

def solve_optimization(nodes, tf, iterations=1):
    t0, tf = 0.0, tf
    num_nodes = nodes
    interval_value = (tf - t0)/(num_nodes - 1)

    state_symbols = q + [w]
    specified_symbols = (v, q00, q11)

    # Specify the objective function and form the gradient.

    #obj_func = 1/(2*delta) * (qa - q[0](tf))**2
    #   + 1/(2*delta) * (q[9](tf) - qa)**2
    #   + sum([1/(2*delta) * (q[i](tf) - qa)**2 for i in range(1, 9)])
    # this is the objective function to be minimized.

    def obj(free):
        value1 = 1/(2*delta) * (qa - free[num_nodes-1])**2
        value2 = 1/(2*delta) * (qa - free[10*num_nodes-1])**2
        value3 = 1/delta * np.sum([(qa - free[(i+1)*num_nodes-1])**2 for i
                in range(1, 9)])
        return value1 + value2 + value3

    def obj_grad(free):
        grad = np.zeros_like(free)
        grad[num_nodes-1] = 2*1/(2*delta) * (qa - free[num_nodes-1])
        grad[10*num_nodes-1] = 2*1/(2*delta) * (qa - free[10*num_nodes-1])
        for i in range(1, 9):
            grad[(i+1)*num_nodes-1] = 2*1/delta * (qa - free[(i+1)*num_nodes-1])
        return -grad

    # Specify the symbolic instance constraints, as per the example
    instance_constraints = (
        *[q[i].func(t0) - 0 for i in range(10)],
        w.func(t0) - 0,
    )

    bounds = {v: (0, 1)}
    # Create the optimization problem and set any options.
    prob = Problem(obj,
                obj_grad,
                eom,
                state_symbols,
                num_nodes,
                interval_value,
                instance_constraints=instance_constraints,
                bounds=bounds,
    )

    prob.add_option('max_iter', 3000 )

    # Give some rough estimates for the trajectories.
    initial_guess = np.zeros(prob.num_free)

    # Find the optimal solution.
    for _ in range(iterations):
        solution, info = prob.solve(initial_guess)
        initial_guess = solution
        print(info['status_msg'])
        print(f'Objective value achieved: {info['obj_val']:.4e}, ' +
            f'as per the book it is {3.879*1.e-5} \n')

    return prob, solution, info

# %%
# As per the example final time tf = 0.2
tf = 0.2
num_nodes = 101
prob, solution, info = solve_optimization(num_nodes, tf, iterations=3)

# %%
# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# %%
# Plot the objective function as a function of optimizer iteration.
prob.plot_objective_value()

# %%
# sphinx_gallery_thumbnail_number = 2

