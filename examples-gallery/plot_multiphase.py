"""
Multiphase Collision
====================

A block is sliding on a surface with Coulomb friction. Push it with a (limited)
rightward force until it hits a wall 1m on the right. When it hits the wall
enforce a Newtonian collision with a coefficient of restitution e so that the
block bounces off the wall and slides backwards to its original location. Apply
this force such that the block reaches its original location in the minimal
amount of time and that each phase has the same duration.

"""

import numpy as np
import sympy as sm
from opty import Problem
import matplotlib.pyplot as plt

# %%
# Start with defining the fixed duration and number of nodes.
num_nodes = 400

# %%
# Symbolic equations of motion, note that we make two sets: one before the #
# collision and one after.
m, e, mu, g, t, h = sm.symbols('m, e, mu, g, t, h', real=True)
xr, vr, xl, vl, F = sm.symbols('x_r, v_r, x_l, v_l, F', cls=sm.Function)

state_symbols = (xr(t), vr(t), xl(t), vl(t))
constant_symbols = (m, e, mu, g)
specified_symbols = (F(t),)

eom = sm.Matrix([
    xr(t).diff(t) - vr(t),
    m*vr(t).diff(t) + mu*m*g - F(t),
    xl(t).diff(t) - vl(t),
    m*vl(t).diff(t) - mu*m*g,
])
sm.pprint(eom)

# %%
# Specify the known system parameters.
par_map = {
    m: 1.0,
    e: 0.8,
    mu: 0.6,
    g: 9.81,
}


# %%
# Specify the objective function and it's gradient.
def obj(free):
    """Return h (always the last element in the free variables)."""
    return free[-1]


def obj_grad(free):
    """Return the gradient of the objective."""
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad


# %%
# Specify the symbolic instance constraints, i.e. initial and end conditions
# using node numbers 0 to N - 1
dur = (num_nodes - 1)*h
instance_constraints = (
    xr(0*h) - 0.0,
    xr(dur) - 1.0,
    vr(0*h) - 0.0,
    # TODO : figure out how to let the collision happen at any time, not just
    # the halfway point in time
    vl(0*h) + e*vr(dur),
    xl(0*h) - 1.0,
    xl(dur) - 0.0,
)

bounds = {
    F(t): (0.0, 160.0),
    h: (0.0, 0.5),
    vr(t): (0.0, np.inf),
    vl(t): (-np.inf, 0.0),
}

# %%
# Create an optimization problem.
prob = Problem(obj, obj_grad, eom, state_symbols, num_nodes, h,
               known_parameter_map=par_map,
               instance_constraints=instance_constraints,
               time_symbol=t,
               bounds=bounds)

# %%
# Use a zero as an initial guess.
initial_guess = np.zeros(prob.num_free)

# %%
# Find the optimal solution.
solution, info = prob.solve(initial_guess)
print(info['status_msg'])
print(info['obj_val'])

# %%
# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# %%
# Plot the objective function as a function of optimizer iteration.
prob.plot_objective_value()

plt.show()
