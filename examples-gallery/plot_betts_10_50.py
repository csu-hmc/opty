"""
Delay Equation (Göllmann, Kern, and Maurer)
===========================================

This is example 10.50 from Betts' book "Practical Methods for Optimal Control
Using NonlinearProgramming", 3rd edition, Chapter 10: Test Problems.

Let :math:`t_0, t_f` be the starting time and final time. There are instance
constraints: :math:`x_i(t_f) = x_j(t_0), i \\neq j`.
As presently *opty* does not support such instance constraints, I iterate a
few times, hoping it will converge - which it does in this example.

There are also inequalities: :math:`x_i(t) + u_i(t) \\geq 0.3`.
To handle them, I define additional  state variables :math:`q_i(t)` and use
additional EOMs:

:math:`q_i(t) = x_i(t) + u_i(t)`, and then use bounds on :math:`q_i(t)`.

**States**

- :math:`x_1, x_2, x_3. x_4. x_5, x_6` : state variables
- :math:`q_1, q_2, q_3. q_4. q_5, q_6` : state variables for the inequalities

**Controls**

- :math:`u_1, u_2, u_3, u_4, u_5, u_6` : control variables

Note: I simply copied the equations of motion, the bounds and the constants
from the book. I do not know their meaning.

"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
from opty.utils import create_objective_function
import matplotlib.pyplot as plt

# %%
# Equations of Motion
# -------------------
t = me.dynamicsymbols._t

x1, x2, x3, x4, x5, x6 = me.dynamicsymbols('x1, x2, x3, x4, x5, x6')
q1, q2, q3, q4, q5, q6 = me.dynamicsymbols('q1, q2, q3, q4, q5, q6')
u1, u2, u3, u4, u5, u6 = me.dynamicsymbols('u1, u2, u3, u4, u5, u6')

#Parameters
x0 = 1.0
u_minus_1, u0 = 0.0, 0.0

eom = sm.Matrix([
    -x1.diff(t) + x0*u_minus_1,
    -x2.diff(t) + x1*u0,
    -x3.diff(t) + x2*u1,
    -x4.diff(t) + x3*u2,
    -x5.diff(t) + x4*u3,
    -x6.diff(t) + x5*u4,
    -q1 + u1 + x1,
    -q2 + u2 + x2,
    -q3 + u3 + x3,
    -q4 + u4 + x4,
    -q5 + u5 + x5,
    -q6 + u6 + x6,
])
sm.pprint(eom)

# %%
# Define and Solve the Optimization Problem
# -----------------------------------------
num_nodes = 101
t0, tf = 0.0, 1.0
interval_value = (tf - t0) / (num_nodes - 1)

state_symbols = (x1, x2, x3, x4, x5, x6, q1, q2, q3, q4, q5, q6)
unkonwn_input_trajectories = (u1, u2, u3, u4, u5, u6)

# %%
# Specify the objective function and form the gradient.
objective = sm.Integral(x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2 + u1**2
        + u2**2 + u3**2 + u4**2 + u5**2 + u6**2, t)

obj, obj_grad = create_objective_function(
    objective,
    state_symbols,
    unkonwn_input_trajectories,
    tuple(),
    num_nodes,
    interval_value
)

# %%
# Specify the instance constraints and bounds. I use the solution from a
# previous run as the initial guess to save running time in this example.
# It took 8 iterations to get it.
initial_guess = np.random.rand(18*num_nodes) * 0.1
np.random.seed(123)
initial_guess = np.load('betts_10_50_solution.npy')*(1.0 +
                                                0.001*np.random.rand(1))

instance_constraints = (
    x1.func(t0) - 1.0,

    x2.func(t0) - initial_guess[num_nodes-1],
    x3.func(t0) - initial_guess[2*num_nodes-1],
    x4.func(t0) - initial_guess[3*num_nodes-1],
    x5.func(t0) - initial_guess[4*num_nodes-1],
    x6.func(t0) - initial_guess[5*num_nodes-1],

    q1.func(t0) - 0.5,
    q2.func(t0) - 0.5,
    q3.func(t0) - 0.5,
    q4.func(t0) - 0.5,
    q5.func(t0) - 0.5,
    q6.func(t0) - 0.5,
    )

limit_value = np.inf
bounds = {
    q1: (0.3, limit_value),
    q2: (0.3, limit_value),
    q3: (0.3, limit_value),
    q4: (0.3, limit_value),
    q5: (0.3, limit_value),
    q6: (0.3, limit_value),
}

# %%
# Iterate
# -------
# Here I iterate *loop* times, and use the solution to set the instance
# constraints for the next iteration.
loop = 2
for i in range(loop):

    prob = Problem(obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        instance_constraints= instance_constraints,
    )

    prob.add_option('max_iter', 1000)


# Find the optimal solution.
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    print(f'Objective value achieved: {info['obj_val']:.4f}, as per the book '+
        f'it is {3.10812211}, so the error is: '
        f'{(info['obj_val'] - 3.10812211)/3.10812211*100:.3f} % ')
    print('\n')

    instance_constraints = (
        x1.func(t0) - 1.0,

        x2.func(t0) - solution[num_nodes-1],
        x3.func(t0) - solution[2*num_nodes-1],
        x4.func(t0) - solution[3*num_nodes-1],
        x5.func(t0) - solution[4*num_nodes-1],
        x6.func(t0) - solution[5*num_nodes-1],
    )

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
# Are the instance constraints satisfied?
delta = [solution[0] - 1.0]
for i in range(1, 6):
    delta.append(solution[i*num_nodes] - solution[i*num_nodes-1])

labels = [r'$x_1(t_0) - 1$', r'$x_1(t_f) - x_2(t_0)$', r'$x_2(t_f) - x_3(t_0)$',
          r'$x_3(t_f) - x_4(t_0)$', r'$x_4(t_f) - x_5(t_0)$',
          r'$x_5(t_f) - x_6(t_0)$']
fig, ax = plt.subplots(figsize=(8, 2), constrained_layout=True)
ax.bar(labels, delta)
ax.set_title('Instance constraint violations')
prevent_print = 1

# %%
# Are the inequality constraints satisfied?
min_q = np.min(solution[7*num_nodes: 12*num_nodes-1])
if min_q >= 0.3:
    print(f"Minimal value of the q\u1D62 is: {min_q:.3f} >= 0.3, so satisfied")
else:
    print(f"Minimal value of the q\u1D62 is: {min_q:.3f} < 0.3, so not satisfied")

# %%
# sphinx_gallery_thumbnail_number = 4
