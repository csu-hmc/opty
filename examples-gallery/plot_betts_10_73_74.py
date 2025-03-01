"""
Linear Tangent Steering
=======================

These are example 10.73 and 10.74 from Betts' book "Practical Methods for
Optimal Control Using NonlinearProgramming", 3rd edition,
Chapter 10: Test Problems.
They describe the **same** problem, but are **formulated differently**.
More details in section 5.6. of the book.



Note: I simply copied the equations of motion, the bounds and the constants
from the book. I do not know their meaning.

"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
import matplotlib.pyplot as plt

# %%
# Example 10.74
# -------------
#
# **States**
#
# - :math:`x_0, x_1, x_2, x_3` : state variables
#
# **Controls**
#
# - :math:`u` : control variable
#
# Equations of motion.
#
t = me.dynamicsymbols._t

x = me.dynamicsymbols('x:4')
u = me.dynamicsymbols('u')
h = sm.symbols('h')

#Parameters
a = 100.0

eom = sm.Matrix([
    -x[0].diff(t) + x[2],
    -x[1].diff(t) + x[3],
    -x[2].diff(t) + a * sm.cos(u),
    -x[3].diff(t) + a * sm.sin(u),
])
sm.pprint(eom)

# %%
# Define and Solve the Optimization Problem
# -----------------------------------------
num_nodes = 301

t0, tf = 0*h, (num_nodes - 1) * h
interval_value = h

state_symbols = x
unkonwn_input_trajectories = (u, )

# Specify the objective function and form the gradient.
def obj(free):
    return free[-1]

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad

instance_constraints = (
    x[0].func(t0),
    x[1].func(t0),
    x[2].func(t0),
    x[3].func(t0),
    x[1].func(tf) - 5.0,
    x[2].func(tf) - 45.0,
    x[3].func(tf),
)

bounds = {
        h: (0.0, 0.5),
        u: (-np.pi/2, np.pi/2),
}

# %%
# Create the optimization problem and set any options.
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

# Give some rough estimates for the trajectories.
initial_guess = np.ones(prob.num_free) * 0.1

# Find the optimal solution.
for i in range(1):
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    print(f'Objective value achieved: {info['obj_val']*(num_nodes-1):.4f}, ' +
          f'as per the book it is {0.554570879}, so the error is: ' +
        f'{(info['obj_val']*(num_nodes-1) - 0.554570879)/0.554570879 :.3e} ')

solution1 = solution

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
# Example 10.73
# -------------
#
# There is a boundary condition at the end of the interval, at :math:`t = t_f`:
# :math:`1 + \lambda_0 x_2 + \lambda_1 x_3 + \lambda_2 a \hspace{2pt} \text{cosu} + \lambda_3 a \hspace{2pt} \text{sinu} = 0`
#
# where :math:`sinu = - \lambda_3 / \sqrt{\lambda_2^2 + \lambda_3^2}` and
# :math:`cosu = - \lambda_2 / \sqrt{\lambda_2^2 + \lambda_3^2}` and a is a constant
#
# As *opty* presently does not support such instance constraints, I introduce
# a new state variable and an additional equation of motion:
#
# :math:`hilfs = 1 + \lambda_0 x_2 + \lambda_1 x_3 + \lambda_2 a \hspace{2pt} \text{cosu} + \lambda_3 a \hspace{2pt} \text{sinu}`
#
# and I use the instance constraint :math:`hilfs(t_f) = 0`.
#
#
# **states**
#
# - :math:`x_0, x_1, x_2, x_3` : state variables
# - :math:`lam_0, lam_1, lam_2, lamb_3` : state variables
# - :math:`hilfs` : state variable
#
#
# Equations of Motion
# -------------------
#
t = me.dynamicsymbols._t

x = me.dynamicsymbols('x:4')
lam = me.dynamicsymbols('lam:4')
hilfs = me.dynamicsymbols('hilfs')
h = sm.symbols('h')

# Parameters
a = 100.0

cosu = - lam[2] / sm.sqrt(lam[2]**2 + lam[3]**2)
sinu = - lam[3] / sm.sqrt(lam[2]**2 + lam[3]**2)

eom = sm.Matrix([
    -x[0].diff(t) + x[2],
    -x[1].diff(t) + x[3],
    -x[2].diff(t) + a * cosu,
    -x[3].diff(t) + a * sinu,
    -lam[0].diff(t),
    -lam[1].diff(t),
    -lam[2].diff(t) - lam[0],
    -lam[3].diff(t) - lam[1],
    -hilfs + 1 + lam[0]*x[2] + lam[1]*x[3] + lam[2]*a*cosu + lam[3]*a*sinu,
])
sm.pprint(eom)

# %%
# Define and Solve the Optimization Problem
# -----------------------------------------

state_symbols = x + lam + [hilfs]

t0, tf = 0*h, (num_nodes - 1) * h
interval_value = h

# %%
# Specify the objective function and form the gradient.
def obj(free):
    return free[-1]

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad

instance_constraints = (
    x[0].func(t0),
    x[1].func(t0),
    x[2].func(t0),
    x[3].func(t0),
    x[1].func(tf) - 5.0,
    x[2].func(tf) - 45.0,
    x[3].func(tf),
    lam[0].func(t0),
    hilfs.func(tf),
)

bounds = {
        h: (0.0, 0.5),
}
# %%
# Create the optimization problem and set any options.
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

# %%
# Give some estimates for the trajectories. This example only converged with
# very close initial guesses.
i1 = list(solution1[: -1])
i2 = [0.0 for _ in range(4*num_nodes)]
i3 = [0.0]
i4 = solution1[-1]
initial_guess = np.array(i1 + i2 + i3 + i4)

# %%
# Find the optimal solution.
for i in range(1):
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    print(f'Objective value achieved: {info['obj_val']*(num_nodes-1):.4f}, ' +
          f'as per the book it is {0.55457088}, so the error is: ' +
        f'{(info['obj_val']*(num_nodes-1) - 0.55457088)/0.55457088:.3e}')

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
# Compare the two solutions.
difference = np.empty(4*num_nodes)
for i in range(4*num_nodes):
    difference[i] = solution1[i] - solution[i]

fig, ax = plt.subplots(4, 1, figsize=(8, 6), constrained_layout=True)
names = ['x0', 'x1', 'x2', 'x3']
times = np.linspace(t0, (num_nodes-1)*solution[-1], num_nodes)
for i in range(4):
    ax[i].plot(times, difference[i*num_nodes:(i+1)*num_nodes])
    ax[i].set_ylabel(f'difference in {names[i]}')
ax[0].set_title('Difference in the state variables between the two solutions')
ax[-1].set_xlabel('time');
prevent_print = 1

# %%
# sphinx_gallery_thumbnail_number = 5
