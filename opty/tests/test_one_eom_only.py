 # %%
"""
Hypersensitive Control
======================

This is example 10.7 from Betts' book Practical Methods for Optimal Control
Using Nonlinear Programming, 3rd edition, chapter 10, Test Problems.
It has only one equation of motion.

**States**

- y : state variable

**Specifieds**

- u : control variable

"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
from opty.utils import create_objective_function
import time

# %%
# Equations of motion.
t = me.dynamicsymbols._t
y, u = me.dynamicsymbols('y u')

eom = sm.Matrix([-y.diff(t) - y**3 + u])
sm.pprint(eom)

# %%

t0, tf = 0.0, 10.0
num_nodes = 1000
interval_value = (tf - t0)/(num_nodes - 1)

state_symbols = (y, )
specified_symbols = (u,)

# Specify the objective function and form the gradient.
start = time.time()
obj_func = sm.Integral(y**2 + u**2, t)
sm.pprint(obj_func)
obj, obj_grad = create_objective_function(
        obj_func,
        state_symbols,
        specified_symbols,
        tuple(),
        num_nodes,
        node_time_interval=interval_value
)

# Specify the symbolic instance constraints.
instance_constraints = (
        y.func(t0) - 1,
        y.func(tf) - 1.5,
)

# Create the optimization problem and set any options.
prob = Problem(
        obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        instance_constraints=instance_constraints,
)

# Give some rough estimates for the x and y trajectories.
initial_guess = np.zeros(prob.num_free)

# Find the optimal solution.
solution, info = prob.solve(initial_guess)
print(info['status_msg'])
print(f'Objective value achieved: {info['obj_val']:.4f}, as per the book ' +
        f'it is {6.7241}, so the error is: '
        f'{(info['obj_val'] - 6.7241)/6.7241*100:.3f} % ')
print(f'Time taken for the simulation: {time.time() - start:.2f} s')

# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# Plot the objective function as a function of optimizer iteration.
prob.plot_objective_value()