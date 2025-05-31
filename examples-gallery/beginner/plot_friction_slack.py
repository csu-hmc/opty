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
num_nodes = 501

# %%
# Symbolic equations of motion, note that we make two sets: one before the #
# collision and one after.
m, mu, g, t, h = sm.symbols('m, mu, g, t, h', real=True)
x, v, psi, Fp, Fn, F = sm.symbols('x, v, psi, Fp, Fn, F', cls=sm.Function)

state_symbols = (x(t), v(t))
constant_symbols = (m, mu, g)
specified_symbols = (F(t), Fn(t), Fp(t), psi(t))

eom = sm.Matrix([
    x(t).diff(t) - v(t),
    m*v(t).diff(t) - Fp(t) + Fn(t) - F(t),
    # following two lines ensure: psi >= abs(v)
    psi(t) + v(t),  # >= 0
    psi(t) - v(t),  # >= 0
    # mu*m*g >= Fp + Fn
    mu*m*g - Fp(t) - Fn(t),  # >= 0
    # mu*m*g*psi = (Fp + Fn)*psi -> mu*m*g = Fn v > 0 & mu*m*g = Fp if v < 0
    (mu*m*g - Fp(t) - Fn(t))*psi(t),
    # Fp*psi = -Fp*v -> Fp is zero if v > 0
    Fp(t)*(psi(t) + v(t)),
    # Fn*psi = Fn*v -> Fn is zero if v < 0
    Fn(t)*(psi(t) - v(t)),
])
sm.pprint(eom)

# %%
# Specify the known system parameters.
par_map = {
    m: 1.0,
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
t0, tf = 0*h, (num_nodes - 1)*h
instance_constraints = (
    x(t0) - 0.0,
    v(t0) - 0.0,
    #x(tf/2) - 10.0,
    #v(tf/2) - 0.0,
    x(tf) - 10.0,
    v(tf) - 0.0,
)

bounds = {
    F(t): (0.0, 200.0),
    h: (0.0, 1.0),
}

eom_bounds = {
    2: (0.0, np.inf),
    3: (0.0, np.inf),
    4: (0.0, np.inf),
}

# %%
# Create an optimization problem.
prob = Problem(obj, obj_grad, eom, state_symbols, num_nodes, h,
               known_parameter_map=par_map,
               instance_constraints=instance_constraints,
               time_symbol=t,
               bounds=bounds, eom_bounds=eom_bounds,
               backend='numpy')

# %%
# Use a zero as an initial guess.
initial_guess = 0.02*np.ones(prob.num_free)
initial_guess[-1] = 0.01

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
