"""
Coulomb Friction with Slack Variables
=====================================

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
from opty.utils import MathJaxRepr
import matplotlib.pyplot as plt

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
MathJaxRepr(eom)

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
# %%
# Start with defining the fixed duration and number of nodes.
N = 400

t0, tm, tf = 0*h, (N//2)*h, (N - 1)*h
instance_constraints = (
    x(t0) - 0.0,
    v(t0) - 0.0,
    x(tm) - 10.0,
    v(tm) - 0.0,
    x(tf) + 0.0,
    v(tf) - 0.0,
)

bounds = {
    F(t): (-400.0, 400.0),
    h: (0.0, 1.0),
}

eom_bounds = {
    2: (0.0, np.inf),
    3: (0.0, np.inf),
    4: (0.0, np.inf),
}

# %%
# Create an optimization problem.
prob = Problem(obj, obj_grad, eom, state_symbols, N, h,
               known_parameter_map=par_map,
               instance_constraints=instance_constraints,
               time_symbol=t,
               bounds=bounds, eom_bounds=eom_bounds,
               backend='numpy',
               )

prob.add_option('max_iter', 10000)

# %%
# Use a zero as an initial guess.
half = N//2
initial_guess = np.zeros(prob.num_free)

initial_guess[0*N:1*N - half] = np.linspace(0.0, 10.0, num=half)  # x
initial_guess[1*N - half:1*N] = np.linspace(10.0, 0.0, num=half)  # x

initial_guess[1*N:2*N - half] = 10.0  # v
initial_guess[2*N - half:2*N] = -10.0  # v

initial_guess[2*N:3*N - half] = 5.0  # F
initial_guess[3*N - half:3*N] = -5.0  # F

initial_guess[3*N:4*N - half] = 0.6  # Fn
initial_guess[4*N - half:4*N] = 0.0  # Fn

initial_guess[4*N:5*N - half] = 0.0  # Fp
initial_guess[5*N - half:5*N] = 0.6  # Fp

initial_guess[5*N:6*N - half] = 10.0  # psi
initial_guess[6*N - half:6*N] = 10.0  # psi

initial_guess[-1] = 0.01

# %%
# Plot the optimal state and input trajectories.
prob.plot_trajectories(initial_guess)

# %%
# Find the optimal solution.
solution, info = prob.solve(initial_guess)
print(info['status_msg'])
print(info['obj_val'])

# %%
# Plot the objective function as a function of optimizer iteration.
prob.plot_objective_value()

# %%
# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# %%
# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

plt.show()
