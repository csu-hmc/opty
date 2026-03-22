"""
Coulomb Friction with Slack Variables
=====================================

A block of mass :math:`m` is being pushed with force :math:`F(t)` along on a
horizontal surface. Coulomb friction acts between the block and the surface.
Find a minimal time solution to push the block 10 meters from being stationary
and then back to the original stationary position.

Objectives
----------

- Demonstrate how slack variables and inequality constraints can be used to
  manage discontinuties in the equations of motion.

"""

import numpy as np
import sympy as sm
from opty import Problem
from opty.utils import MathJaxRepr
import matplotlib.pyplot as plt

# %%
# In general, the Coulomb friction force is a piecewise function (ignoring the
# state at :math:`v=0`):
#
# .. math::
#
#    F_f = \begin{cases}
#            \phantom{-}\mu m g & \textrm{if }  v < 0  \\
#            -\mu m g & \textrm{if }  v > 0  \\
#          \end{cases}
#
# If :math:`F_f = F_f^+ - F_f^-` where there are two positive components of
# friction. Then the sum of the two positive valued fricton components must
# always be less than or equal than the Coulomb magnitude (both could be zero):
#
# .. math::
#
#    \mu m g \geq F_f^+ + F_f^-
#
# The slack variable :math:`\psi` is introduced and constrained to enforce it
# to be greater than :math:`|v|`, so :math:`\psi` is always positive:
#
# .. math::
#
#    \psi \geq -v \\
#    \psi \geq  v
#
# This should do the same thing as the above two (looks like the Fischer-B for
# a single variable):
#
# .. math::
#
#    \psi \geq \sqrt{v^2}
#
# Using :math:`\psi`, the following two constraints then ensures that
# :math:`F_f^+` is zero if :math:`v > 0` and :math:`F_f^-` is zero if :math:`v
# < 0`:
#
# .. math::
#
#    F_f^+ \psi = -F_f^+ v \\
#    F_f^- \psi = F_f^- v
#
# Again using :math:`\psi`, the following constraint ensures that :math:`\mu mg
# = F_f^-` if  :math:`v > 0` and :math:`\mu m g = F_f^+` and if :math:`v < 0`:
#
# .. math::
#
#    \mu m g \psi = (F_f^+ + F_f^-)\psi
#
# TODO : One issue seems to be that Ff can be anything at v = 0.
#
# Some possible alternative equations:
#
# TODO : This linear complimentarity constraint would enforce Ff always being
# opposite of v:
#
# .. math::
#
#    -F_f v >= 0
#
# This linear complimentarity constraint ensures that only one component of
# friction can be greater than zero at a time:
#
# .. math::
#
#    F_f^+  F_f^- = 0
#
# Fischer-Burmeister equation for this:
#
# .. math::
#
#    \sqrt{{}^+F_f^2 + {}^-F_f^2} - {}^+F_f  - {}^-F_f = 0
#

# %%
# Symbolic equations of motion.
m, mu, g, t, h = sm.symbols('m, mu, g, t, h', real=True)
x, v, psi, Ffp, Ffn, F = sm.symbols('x, v, psi, Ffp, Ffn, F', cls=sm.Function)

state_symbols = (x(t), v(t))
constant_symbols = (m, mu, g)
specified_symbols = (F(t), Ffn(t), Ffp(t), psi(t))

eom = sm.Matrix([
    # equations of motion with positive and negative friction force
    x(t).diff(t) - v(t),
    m*v(t).diff(t) - Ffp(t) + Ffn(t) - F(t),
    # following two lines ensure: psi >= abs(v)
    psi(t) + v(t),  # >= 0
    psi(t) - v(t),  # >= 0
    # mu*m*g >= Ffp + Ffn
    mu*m*g - Ffp(t) - Ffn(t),  # >= 0
    # mu*m*g*psi = (Ffp + Ffn)*psi -> mu*m*g = Ffn v > 0 & mu*m*g = Ffp if v < 0
    (mu*m*g - Ffp(t) - Ffn(t))*psi(t),
    # Ffp*psi = -Ffp*v -> Ffp is zero if v > 0
    Ffp(t)*(psi(t) + v(t)),
    # Ffn*psi = Ffn*v -> Ffn is zero if v < 0
    Ffn(t)*(psi(t) - v(t)),
])


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
N = 40

t0, tm, tf = 0*h, (N//2)*h, (N - 1)*h
instance_constraints = (
    x(t0) - 0.0,
    v(t0) - 0.0,
    x(tm) - 10.0,
    v(tm) - 0.0,
    x(tf) + 0.0,
    v(tf) - 0.0,
    # It is indeterminant what the friction force shoudl be at v = 0, so we
    # just force it to be zero.
    Ffp(t0),
    Ffn(t0),
    Ffp(tm),
    Ffn(tm),
    Ffp(tf),
    Ffn(tf),
)

bounds = {
    F(t): (-400.0, 400.0),
    Ffn(t): (0.0, np.inf),
    Ffp(t): (0.0, np.inf),
    h: (0.0, 0.2),
    psi(t): (0.0, np.inf),
    x(t): (0.0, 10.0),
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
               backend='numpy')

prob.add_option('max_iter', 10000)

# %%
# Use a zero as an initial guess.
half = N//2
initial_guess = np.zeros(prob.num_free)

initial_guess[0*N:1*N - half] = np.linspace(0.0, 10.0, num=half)  # x
initial_guess[1*N - half:1*N] = np.linspace(10.0, 0.0, num=half)  # x

initial_guess[1*N:2*N - half] = 10.0  # v
initial_guess[2*N - half:2*N] = -10.0  # v

initial_guess[2*N:3*N - half] = 100.0  # F
initial_guess[3*N - half:3*N] = -100.0  # F

initial_guess[3*N:4*N - half] = 5.0  # Ffn
initial_guess[4*N - half:4*N] = 0.0  # Ffn

initial_guess[4*N:5*N - half] = 0.0  # Ffp
initial_guess[5*N - half:5*N] = 5.0  # Ffp

initial_guess[5*N:6*N - half] = 10.0  # psi
initial_guess[6*N - half:6*N] = 10.0  # psi

initial_guess[-1] = 0.05

# %%
# Plot the initial guess.
_ = prob.plot_trajectories(initial_guess)

# %%
# Find the optimal solution.
solution, info = prob.solve(initial_guess)
print(info['status_msg'])
print(info['obj_val'])

# %%
# Plot the objective function as a function of optimizer iteration.
_ = prob.plot_objective_value()

# %%
# Plot the constraint violations.
_ = prob.plot_constraint_violations(solution)

# %%
# Plot the optimal state and input trajectories.
axes = prob.plot_trajectories(solution, skip_first=True)
for ax in axes:
    lines = ax.get_lines()
    ax.axvline(N//2*solution[-1], color='black')
    for line in lines:
        line.set_marker('o')

# %%
# Plot the friction force.
xs, rs, _, _ = prob.parse_free(solution)
ts = prob.time_vector(solution)
fig, ax = plt.subplots()
ax.plot(ts, -rs[1] + rs[2], marker='o')
ax.set_ylabel(r'$F_f$ [N]')
ax.set_xlabel('Time [s]')

plt.show()
