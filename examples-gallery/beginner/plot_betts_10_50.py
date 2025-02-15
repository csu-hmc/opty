r"""
Delay Equation (Göllmann, Kern, and Maurer)
===========================================

This is example 10.50 from [Betts2010]_.

There are inequalities: :math:`x_i(t) + u_i(t) \geq 0.3`. To handle them,
additional state variables :math:`q_i(t)` and additional equations of motion
can be added: :math:`q_i(t) = x_i(t) + u_i(t)`, and then use bounds on
:math:`q_i(t)`.

**States**

- :math:`x_1, x_2, x_3. x_4. x_5, x_6` : state variables
- :math:`q_1, q_2, q_3. q_4. q_5, q_6` : state variables for the inequalities

**Controls**

- :math:`u_1, u_2, u_3, u_4, u_5, u_6` : control variables

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

x1, x2, x3, x4, x5, x6 = me.dynamicsymbols('x1, x2, x3, x4, x5, x6')
q1, q2, q3, q4, q5, q6 = me.dynamicsymbols('q1, q2, q3, q4, q5, q6')
u1, u2, u3, u4, u5, u6 = me.dynamicsymbols('u1, u2, u3, u4, u5, u6')

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
num_nodes = 501
t0, tf = 0.0, 1.0
interval_value = (tf - t0)/(num_nodes - 1)

state_symbols = (x1, x2, x3, x4, x5, x6, q1, q2, q3, q4, q5, q6)
unkonwn_input_trajectories = (u1, u2, u3, u4, u5, u6)

# %%
# Specify the objective function and form the gradient.
objective = sm.Integral(
    x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2 +
    u1**2 + u2**2 + u3**2 + u4**2 + u5**2 + u6**2, t)

obj, obj_grad = create_objective_function(
    objective,
    state_symbols,
    unkonwn_input_trajectories,
    tuple(),
    num_nodes,
    interval_value,
    time_symbol=t,
)

# %%
# Specify the instance constraints and bounds

instance_constraints = (
    x1.func(t0) - 1.0,
    x2.func(t0) - x1.func(tf),
    x3.func(t0) - x2.func(tf),
    x4.func(t0) - x3.func(tf),
    x5.func(t0) - x4.func(tf),
    x6.func(t0) - x5.func(tf),
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
# Solve the Optimization Problem
# ------------------------------

prob = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    num_nodes,
    interval_value,
    instance_constraints=instance_constraints,
    bounds=bounds,
    time_symbol=t,
)

prob.add_option('max_iter', 1000)

initial_guess = np.random.rand(18*num_nodes) * 0.1
for i in range(6*num_nodes, 12*num_nodes):
    initial_guess[i] = max(0.3, initial_guess[i])

for _ in range(1):
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    print(f'Objective value achieved: {info['obj_val']:.4f}, as per the book '
          f'it is {3.10812211}, so the error is: '
          f'{(info['obj_val'] - 3.10812211)/3.10812211*100:.3f} % ')
    print('\n')

# %%
# Plot the optimal state and input trajectories.
_ = prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
_ = prob.plot_constraint_violations(solution)

# %%
# Plot the objective function.
_ = prob.plot_objective_value()

# %%
# Are the inequality constraints satisfied?
min_q = np.min(solution[7*num_nodes:12*num_nodes-1])
if min_q >= 0.3:
    print(f"Minimal value of the q\u1D62 is: {min_q:.12f} >= 0.3, "
          f"so satisfied.")
else:
    print(f"Minimal value of the q\u1D62 is: {min_q:.12f} < 0.3, "
          f"so not satisfied.")

# %%
# sphinx_gallery_thumbnail_number = 2
