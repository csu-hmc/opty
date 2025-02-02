"""
Stiff Set of Differential Equations
===================================

This is example 10.1 from `[Betts2010]_`, 3rd edition, Chapter 10: Test Problems.
More details are in sectionn 4.11.7 of the book.

This is a set of stiff differential equations without any 'physical' meaning.
It was proposed to test optimization algorithms.
The inequality constraint :math:`y^Ty \\geq function(time, parameters)` forces
the norm of the state vector to the larger than a wildly fluctuating function.
The objective to be minimized forces :math:`y^Ty` to be close to the
constraint.
As a stiff system wants to move 'straight' and here it is forced to turn wildly,
this is a good system to test optimization algorithms.

The main issue as far as *opty* is concerned is this inequality constraint for
the state variables:
:math:`y_1^2 + y_2^2 + y_3^2 + y_4^2 \\geq function(time, parameters)`

As presently *opty* does not support inequality constraints, I introduced a new
state variable ``fact`` to take care of the inequality. The inequality is then
reformulated as
:math:`y_1^2 + y_2^2 + y_3^2 + y_4^2 = fact \\cdot function(time, parameters)`.

and I restrict fact :math:`\\geq 1.0`.

**States**

- :math:`y_1, .., y_4` : state variables as per the book
- :math:`fact`: additional state variable to take care of the inequality

**Specifieds**

- :math:`u_1, u_2` : control variables

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
#
t = me.dynamicsymbols._t
y1, y2, y3, y4 = me.dynamicsymbols('y1 y2 y3 y4')
fact = me.dynamicsymbols('fact')
u1, u2 = me.dynamicsymbols('u1 u2')

# %%
# As the time occurs explicitly in the equations of motion, I use a function
# to represent it symbolically. This will be a known_trajectory_map in the
# optimization problem.

T = sm.symbols('T', cls=sm.Function)

# %%
def p(t, a, b):
    return sm.exp(-b*(t - a)**2)

rhs = (3.0*(p(T(t), 3, 12) + p(T(t), 6, 10) + p(T(t), 10, 6)) +
                8.0*p(T(t), 15, 4) + 0.01)

eom = sm.Matrix([
                -y1.diff(t) - 10*y1 + u1 + u2,
                -y2.diff(t) - 2*y2 + u1 - 2*u2,
                -y3.diff(t) - 3*y3 + 5*y4 + u1 - u2,
                -y4.diff(t) + 5*y3 - 3*y4 + u1 + 3*u2,
                y1**2 + y2**2 + y3**2 + y4**2 - fact*rhs,
])
sm.pprint(eom)

# %%
# Set up and Solve the Optimization Problem
# -----------------------------------------

t0, tf = 0.0, 20.0
num_nodes = 601
interval_value = (tf - t0)/(num_nodes - 1)
times = np.linspace(t0, tf, num_nodes)

state_symbols = (y1, y2, y3, y4, fact)
specified_symbols = (u1, u2)
integration_method = 'backward euler'

# Specify the objective function and form the gradient.
obj_func = sm.Integral(100*(y1**2 + y2**2 + y3**2 + y4**2)
            + 0.01*(u1**2 + u2**2), t)

obj, obj_grad = create_objective_function(
        obj_func,
        state_symbols,
        specified_symbols,
        tuple(),
        num_nodes,
        node_time_interval=interval_value,
        integration_method=integration_method,
)

# Specify the symbolic instance constraints, as per the example.
instance_constraints = (
        y1.func(t0) - 2.0,
        y2.func(t0) - 1.0,
        y3.func(t0) - 2.0,
        y4.func(t0) - 1.0,
        y1.func(tf) - 2.0,
        y2.func(tf) - 3.0,
        y3.func(tf) - 1.0,
        y4.func(tf) + 2.0,
)

bounds = {fact: (1.0 , 10000 )}

# %%
# Create the optimization problem and set any options.
prob = Problem(
        obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        instance_constraints=instance_constraints,
        known_trajectory_map={T(t): times},
        bounds=bounds,
        integration_method=integration_method,
)

prob.add_option('max_iter', 7000)

# %%
# Give some rough estimates for the trajectories.
# In order to speed up the optimization, I used the solution from a previous
# run as the initial guess.
initial_guess = np.zeros(prob.num_free)
initial_guess = np.load('betts_10_1_opty_solution.npy')

# %%
# Find the optimal solution.
for _ in range(1):
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    print(f'Objective value achieved: {info['obj_val']:.4f}, as per the book ' +
        f'it is {2030.85609}, so the difference is: '
        f'{(info['obj_val'] - 2030.85609)/2030.85609*100:.3f} % ')

# %%
# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# %%
# Plot the objective function as a function of optimizer iteration.
prob.plot_objective_value()

# sphinx_gallery_thumbnail_number = 2

# %%
# Verify that the inequality is always kept, i.e. that fact >= 1.0 always.
print(f'minimum value of fact = {np.min(solution[4*num_nodes: 5*num_nodes]):.4e}')
