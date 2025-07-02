# %%
r"""
Coulomb Friction with Slack Variables
=====================================

Objectives
-----------

- Demonstrate how slack variables and inequality constraints can be used to
  manage discontinuties in the equations of motion.
- Show how to use a differentiable approximation to get good initial guesses
  for the original problem.

Description
-----------

A block of mass :math:`m` is being pushed with force :math:`F(t)` along on a
surface. Coulomb friction acts between the block and the surface. Find a
minimal time solution to push the block 10 meters and then back to the original
position.

Notes
-----

- Good initial guesses are needed in this example.

**States**

- :math:`x(t)` - position of the block
- :math:`v(t)` - velocity of the block

**Inputs**

- :math:`F(t)` - force applied to the block
- :math:`F_{fp}(t)` - positive friction force
- :math:`F_{fn}(t)` - negative friction force
- :math:`\psi(t)` - slack variable to handle discontinuities in the equations
  of motion.

**Parameters**

- :math:`m` - mass of the block
- :math:`\mu` - coefficient of friction
- :math:`g` - gravitational acceleration

"""

import numpy as np
import sympy as sm
from opty import Problem
from opty.utils import MathJaxRepr
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_number = 6

# %%
# Coulomb Friction with Step Function
# ===================================
#
# A differentiable approximation of the Coulomb friction force is used to
# get good initial guesses for the original problem.


def smooth_step(x, steepness=10.0):
    """Return a smooth step function, with step at x = 0.0"""
    return 0.5 * (1 + sm.tanh(steepness * x))


# Symbolic equations of motion.
m, mu, g, t, h, Fr = sm.symbols('m, mu, g, t, h, Fr', real=True)
x, v, psi, Ffp, Ffn, F = sm.symbols('x, v, psi, Ffp, Ffn, F', cls=sm.Function)
h = sm.symbols('h', real=True)

state_symbols = (x(t), v(t))
constant_symbols = (m, mu, g)
specified_symbols = (F(t), Ffn(t), Ffp(t), psi(t))

eom = sm.Matrix([
    # equations of motion with positive and negative friction force
    x(t).diff(t) - v(t),
    m*v(t).diff(t) - Ffp(t) + Ffn(t) - F(t),
    Ffp(t) - Fr * smooth_step(v(t)),
    Ffn(t) - Fr * smooth_step(-v(t)),
])

MathJaxRepr(eom)

# %%
# Specify the known system parameters.

par_map = {
    m: 1.0,
    mu: 0.6,
    g: 9.81,
}
Freib = par_map[m] * par_map[g] * par_map[mu]  # Max. friction force
par_map.update({Fr: Freib})


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
# Start with defining the fixed duration and number of nodes.
# N must be even so the solution of the hump approximation may be used as an
# initial guess for the slack variable problem.
N = 100
if N % 2 != 0:
    raise ValueError("N must be even for this example.")

t0, tm, tf = 0*h, (N // 2) * h, (N - 1)*h
instance_constraints = (
    x(t0) - 0.0,
    v(t0) - 0.0,
    Ffp(t0) - 0.0,
    Ffn(t0) - 0.0,
    x(tm) - 10.0,
    v(tm) - 0.0,
    x(tf) + 0.0,
    v(tf) - 0.0,
)

bounds = {
    F(t): (-400.0, 400.0),  # Force
    h: (0.0, 0.2),
}

# %%
# Create an optimization problem.
prob = Problem(obj, obj_grad, eom, state_symbols, N, h,
               known_parameter_map=par_map,
               instance_constraints=instance_constraints,
               time_symbol=t,
               bounds=bounds)

prob.add_option('max_iter', 5000)

# %%
# Use a random initial guess.
np.random.seed(42)
initial_guess = np.random.randn(prob.num_free)
initial_guess[-1] = 0.005

# %%
# Plot the initial guess.
_ = prob.plot_trajectories(initial_guess)

# %%
# Find the optimal solution.
solution, info = prob.solve(initial_guess)
initial_guess = solution
print(info['status_msg'])
print(f"Interval value h = {info['obj_val']:.5f} s")
# %%
# Plot the objective function as a function of optimizer iteration.
_ = prob.plot_objective_value()

# %%
# Plot the constraint violations.
_ = prob.plot_constraint_violations(solution)

# %%
# Plot the optimal state and input trajectories.
_ = prob.plot_trajectories(solution)

# %%
# Plot the friction force.
xs, rs, _, _ = prob.parse_free(solution)
ts = prob.time_vector(solution)
fig, ax = plt.subplots()
ax.plot(ts, -rs[1] + rs[2])
ax.set_ylabel(r'$F_f$ [N]')
ax.set_xlabel('Time [s]')
ax.set_title('Friction Force with Smooth Step Function')
plt.show()

# %%
# Coulomb Friction with Slack Variables
# =====================================
#
# This is the original Problem using slack variables.

# %%
# Symbolic equations of motion.

eom = sm.Matrix([
    # equations of motion with positive and negative friction force
    x(t).diff(t) - v(t),
    m*v(t).diff(t) - Ffp(t) + Ffn(t) - F(t),
    # following two lines ensure: psi >= abs(v)
    psi(t) + v(t),  # >= 0
    psi(t) - v(t),  # >= 0
    # mu*m*g*psi = (Ffp + Ffn)*psi -> mu*m*g = Ffn v > 0 & mu*m*g = Ffp
    # if v < 0
    (mu*m*g - Ffp(t) - Ffn(t))*psi(t),
    # Ffp*psi = -Ffp*v -> Ffp is zero if v > 0
    Ffp(t)*(psi(t) + v(t)),
    # Ffn*psi = Ffn*v -> Ffn is zero if v < 0
    Ffn(t)*(psi(t) - v(t)),
])

MathJaxRepr(eom)

# %%
# Adjust parameters and bounds to the slack variable problem.
del par_map[Fr]

bounds.update({Ffn(t): (0.0, Freib)})  # Negative friction force
bounds.update({Ffp(t): (0.0, Freib)})  # Positive friction force

eom_bounds = {
   2: (0.0, np.inf),
   3: (0.0, np.inf),
}

# %%
# Create an optimization problem.
prob = Problem(obj, obj_grad, eom, state_symbols, N, h,
               known_parameter_map=par_map,
               instance_constraints=instance_constraints,
               time_symbol=t,
               bounds=bounds,
               eom_bounds=eom_bounds)

prob.add_option('max_iter', 5000)

# %%
# Take the solution of the differentiable approximation as initial guess. Some
# noise is added.
initial_guess = (np.zeros(prob.num_free) +
                 np.random.normal(loc=1.0, scale=1.0, size=prob.num_free))
better_guess = (solution +
                np.random.normal(loc=1.0, scale=1.0, size=len(solution)))
initial_guess[0: 5*N] = better_guess[0: 5*N]
initial_guess[5*N: 6*N] = np.abs(initial_guess[1*N: 2*N])
initial_guess[-1] = solution[-1]

# %%
# Plot the initial guess.
_ = prob.plot_trajectories(initial_guess)

# %%
# Find the optimal solution.
solution, info = prob.solve(initial_guess)
initial_guess = solution
print(info['status_msg'])
print(f"Interval value h = {info['obj_val']:.5f} s")

# %%
# Plot the objective function as a function of optimizer iteration.
_ = prob.plot_objective_value()

# %%
# Plot the constraint violations.
_ = prob.plot_constraint_violations(solution)

# %%
# Plot the optimal state and input trajectories.
ax = prob.plot_trajectories(solution)
t_v0 = (N // 2) * solution[-1]
for i in range(len(ax)):
    ax[i].axvline(t_v0, color='k', linestyle='--')
_ = ax[1].axhline(0.0, color='k', linestyle='--')

# %%
# Plot the friction force.
xs, rs, _, _ = prob.parse_free(solution)
ts = prob.time_vector(solution)
fig, ax = plt.subplots()
ax.plot(ts, -rs[1] + rs[2])
ax.set_ylabel(r'$F_f$ [N]')
ax.set_xlabel('Time [s]')
ax.set_title('Friction Force with Slack Variables')

plt.show()
