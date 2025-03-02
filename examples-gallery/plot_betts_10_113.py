"""
Mixed State-Control Constraints
===============================

This is example 10.113 from Betts' book "Practical Methods for Optimal Control
Using NonlinearProgramming", 3rd edition, Chapter 10: Test Problems.

This simulation shows how to handle inequality constraints, here of the form:
:math:`0 \\geq u + \\dfrac{y_1}{6}`.

I  set:
:math:`J = u + \\dfrac{y_1}{6}`, and bound :math:`J \\leq 0`.


**States**

- :math:`y_1, y_2` : state variables
- :math:`J` : additional state variable to handle the constraint.

**Controls**

- :math:`u` : control variable

Note: I simply copied the equations of motion, the bounds and the constants
from the book. I do not know their meaning.

"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
from opty.utils import create_objective_function

# %%
# Equations of Motion
# -------------------
t = me.dynamicsymbols._t

# Parameter
p = 0.14

y1, y2, J = me.dynamicsymbols(f'y1, y2, J')
u = me.dynamicsymbols('u')

eom = sm.Matrix([
    -y1.diff(t) + y2,
    -y2.diff(t) -y1  + y2*(1.4 - p*y2**2) + 4*u,
    -J + u + y1/6
])

sm.pprint(eom)

# %%
# Define and Solve the Optimization Problem
# -----------------------------------------
num_nodes = 1001
t0, tf = 0.0, 4.5
interval_value = (tf - t0) / (num_nodes - 1)

state_symbols = (y1, y2, J)
unkonwn_input_trajectories = (u,)

# %%
# Specify the objective function and form the gradient.
objective = sm.Integral(u**2 + y1**2, t)

obj, obj_grad = create_objective_function(
    objective,
    state_symbols,
    unkonwn_input_trajectories,
    tuple(),
    num_nodes,
    interval_value
)

# %%

instance_constraints = (
    y1.func(t0) + 5,
    y2.func(t0) + 5,
)

# %%
# Here I bound J <= 0.0
limit_value = np.inf
bounds = {
   J: (-limit_value, 0.0),
}

# %%
# Iterate
# -------

prob = Problem(obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        instance_constraints= instance_constraints,
        bounds=bounds,
)

prob.add_option('max_iter', 1000)

initial_guess = np.ones(prob.num_free) * 0.1
# Find the optimal solution.
for i in range(1):
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    print(f'Objective value achieved: {info['obj_val']:.4f}, as per the book '+
        f'it is {44.8044433}, so the improvement of the value in the book is is: '
        f'{(-info['obj_val'] + 44.8044433)/44.8044433*100:.3f} % ')
    print('\n')


# %%
# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# %%
# Plot the objective function.
prob.plot_objective_value()


# %%
# Is the inequality constraint satisfied at all points in time?
max_J = np.max(solution[2*num_nodes: 3*num_nodes-1])
if max_J <= 0.0:
    print(f"Minimal value of the J\u1D62 is: {max_J:.3e} <= 0.0, so satisfied")
else:
    print(f"Minimal value of the J\u1D62 is: {max_J:.3e} > 0.0, so not satisfied")

# %%
# sphinx_gallery_thumbnail_number = 2
