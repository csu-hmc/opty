"""
Hypersensitive Control
======================

This is example 10.7 from Betts' Test Problems, in the book:
"Practical Methods for Optimal Control Using Nonlinear Programming", 3rd edition,
Chapter 10, by John T. Betts.

**States**

- y : state variable
- uy: its speed

**Specifieds**

- u : control variable

Note: the state variable uy is needed because opt currently needs minimum two
differential equations in the equations of motion. Mathematically it is not
needed.

"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
from opty.utils import create_objective_function
import matplotlib.pyplot as plt

# %%
# Equations of motion.

t = me.dynamicsymbols._t
y, uy, u = me.dynamicsymbols('y uy, u')

eom = sm.Matrix([ uy - y.diff(t), -uy - y**3 + u])
sm.pprint(eom)

# %%
# I packed the optimization problem and its solution into a function, so that
# I can easily call it with different parameters.

def solve_optimization(nodes, tf):
    t0, tf = 0.0, tf
    num_nodes = nodes
    interval_value = (tf - t0)/(num_nodes - 1)

    # Provide some reasonably realistic values for the constants.

    state_symbols = (y, uy)
    specified_symbols = (u,)


    # Specify the objective function and form the gradient.
    obj_func = sm.Integral(y**2 + u**2, t)
    sm.pprint(obj_func)
    obj, obj_grad = create_objective_function(obj_func,
                                          state_symbols,
                                          specified_symbols,
                                          tuple(),
                                          num_nodes,
                                          node_time_interval=interval_value)


    # Specify the symbolic instance constraints, as per the example
    instance_constraints = (
        y.func(t0) - 1,
        y.func(tf) - 1.5,
    )


    # Create the optimization problem and set any options.
    prob = Problem(obj, obj_grad, eom, state_symbols,
               num_nodes, interval_value,
               instance_constraints=instance_constraints,
    )

    prob.add_option('nlp_scaling_method', 'gradient-based')

    # Give some rough estimates for the trajectories.
    initial_guess = np.zeros(prob.num_free)

    # Find the optimal solution.
    solution, info = prob.solve(initial_guess)
    print(info['status_msg'])
    print(f'Objective value achieved: {info['obj_val']:.4f}, as per the book ' +
        f'it is {6.7241}, so the error is: '
        f'{(info['obj_val'] - 6.7241)/6.7241*100:.3f} % ')

    # Plot the optimal state and input trajectories.
    prob.plot_trajectories(solution)

    # Plot the constraint violations.
    prob.plot_constraint_violations(solution)

    # Plot the objective function as a function of optimizer iteration.
    prob.plot_objective_value()

# %%
# As per the example tf = 10000

tf = 10000
num_nodes = 501
solve_optimization(num_nodes, tf)

# %%
# With the value of tf = 10000 above, opty converged to a locally optimal point,
# but the objective value is far from the one given in the book.
# As per the plot of the solution y(t) it seems, that most of the time y(t) = 0,
# only at the very beginning and the very end it is different from 0.
# So, it may make sense to use a smaller tf.
# Also increasing num_nodes may help.

tf = 8.0
num_nodes = 10001
solve_optimization(num_nodes, tf)

# %%

# sphinx_gallery_thumbnail_number = 4

plt.show()
