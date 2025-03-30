# %%
r"""
Multiphase Collision
====================

Objective
---------

- Show how to use DAEs in the equations of motion to allow some instance to
  be determined by ``opty``

Introduction
------------

A block is sliding on a surface with Coulomb friction. Push it with a (limited)
rightward force until it hits a wall 1m on the right. When it hits the wall
enforce a Newtonian collision with a coefficient of restitution e so that the
block bounces off the wall and slides backwards to its original location. Apply
this force such that the block reaches its original location in the minimal
amount of time.

The goal is to find the optimal time of the collision so that the total
duration is minimized.
Prsently opty does not support 'free' times in the instance constraints. So,
to achieve this one can add these conditions to the eoms:

- collision takes place when :math:`x_r(t) = x_{\textrm{impact}}`
- :math:`v_l(t) = -e \cdot v_l(t)` at the time of collision


Notes
-----

- Good initial conditions are always helpdful, particularly in this case.
- One has to use suitable boundary conditions to it converges within the
  constraints imposed by the physical problem.


**States**

- :math:`x_r(t)`: position when the block moves to the right
- :math:`v_r(t)`: velocity when the block moves to the right
- :math:`x_l(t)`: position when the block moves to the left
- :math:`v_l(t)`: velocity when the block moves to the left
- :math:`T(t)`: needed only to record the time of the collision
- :math:`aux_1(t)`: auxiliary state variable
- :math:`aux_2(t)`: auxiliary state variable


**Inputs**

- :math:`F(t)`: force applied to the block [N]


**Fixed Parameters**

- :math:`m`: mass of the block [kg]
- :math:`e`: coefficient of restitution [-]
- :math:`\mu`: coefficient of friction [-]
- :math:`g`: gravity [m/s^2]
- :math:`x_{\textrm{impact}}`: position of the wall [m]


**Free Parameters**

:math:`h`: variable time intervall [s]

"""

import numpy as np
import sympy as sm
from opty import Problem
import matplotlib.pyplot as plt

# %%
# Set Up Good Initial Guess
# -------------------------
# Define differentiable hump and step functions and plot them.
def hump_diff(x, a, b, steep):
    return 0.5 * (sm.tanh(steep * (x - a)) - sm.tanh(steep * (x - b)))

def step_diff(x, a, steep):
    """differentiable step function."""
    return 0.5 * (1 + sm.tanh(steep * (x - a)))

a, b, x, steep = sm.symbols('a, b, x, steep', real=True)

hump_lamb = sm.lambdify((x, a, b, steep), hump_diff(x, a, b, steep))
step_lam = sm.lambdify((x, a, steep), step_diff(x, a, steep))
XX = np.linspace(-5.0, 5.0, 500)
a = -2.0
b = 2.0

steep = 15.0        # steepness of the function

hump = hump_lamb(XX, a, b, steep)
step = step_lam(XX, a, steep)
fig, ax = plt.subplots(2, 1, figsize=(6.5, 2.5), layout='constrained')
ax[0].plot(XX, hump)
ax[0].set_title('differentiable hump function')
ax[0].axvline(a, color='red', linestyle='--', lw=1.0)
ax[0].axvline(b, color='red', linestyle='--', lw=1.0)
ax[1].plot(XX, step)
ax[1].set_title('differentiable Heavyside function')
ax[1].axvline(a, color='red', linestyle='--', lw=1.0)
plt.show()

# Symbolic equations of motion, note that we make two sets: one before the #
# collision and one after.
m, e, mu, g, t, h = sm.symbols('m, e, mu, g, t, h', real=True)
x_impact = sm.symbols('x_impact', real=True)
xr, vr, xl, vl, F, T = sm.symbols('x_r, v_r, x_l, v_l, F, T', cls=sm.Function)

eom = sm.Matrix([
    xr(t).diff(t) - vr(t),
    m*vr(t).diff(t) + mu*m*g - F(t)*(1-step_diff(xr(t), x_impact, steep)),
    xl(t).diff(t) - vl(t),
    m*vl(t).diff(t) - mu*m*g,
    T(t).diff(t) - (1-step_diff(xr(t), x_impact, steep)),
])
sm.pprint(eom)

# %%
state_symbols = (xr(t), vr(t), xl(t), vl(t), T(t))
constant_symbols = (m, e, mu, g)
specified_symbols = (F(t),)

num_nodes = 301

# Specify the known system parameters.
par_map = {
    m: 1.0,
    e: 0.8,
    mu: 0.6,
    g: 9.81,
    x_impact: 1.0
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
# *teiler* is the ratio of the time before and after the collision.

teiler = 1.667
zs = int((num_nodes - 1)/teiler ) * h
dur = (num_nodes - 1) * h

instance_constraints = (
    xr(0*h) - 0.0,
    xr(zs) - 1.0,
    vr(0*h) - 0.0,
        # TODO : figure out how to let the collision happen at any time, not just
        # the halfway point in time
    vl(zs) + e*vr(zs),
    xl(zs) - x_impact,
    xl(dur) - 0.0,
)

bounds = {
    F(t): (0.0, 160.0),
    h: (0.0001, 0.5),
    vr(t): (0.0, np.inf),
    vl(t): (-np.inf, 0.0),
}

# Create an optimization problem.
prob = Problem(obj, obj_grad, eom, state_symbols, num_nodes, h,
        known_parameter_map=par_map,
        instance_constraints=instance_constraints,
        time_symbol=t,
        bounds=bounds)

# Use a zero as an initial guess.

initial_guess = np.zeros(prob.num_free)

# Find the optimal solution.
solution, info = prob.solve(initial_guess)
print(info['status_msg'])
print(f"Smallest variable time interval {info['obj_val']:.4e} sec")

# Plot the optimal state and input trajectories.
ax = prob.plot_trajectories(solution)
for i in range(len(ax)):
    ax[i].axvline(int((num_nodes-1)/teiler)*solution[-1], color='k',
                  linestyle='--')
    ax[-1].set_xlabel((f'time [s] \n The vertical dashed lines give the time '
                       f'of impact'))

# %%
# Plot the constraint violations.
_ = prob.plot_constraint_violations(solution, subplots=True)

# %%
# Plot the objective function as a function of optimizer iteration.
_ = prob.plot_objective_value()
duration1 = solution[-1] * (num_nodes - 1)
plt.show()

# =============================================================================
# ============================================================================

# %%
# Set Up the System with 'free' Collision Time
# --------------------------------------------
aux1, aux2 = sm.symbols('aux1, aux2', cls=sm.Function)

state_symbols = (xr(t), vr(t), xl(t), vl(t), aux1(t), aux2(t), T(t))
constant_symbols = (m, e, mu, g)
specified_symbols = (F(t),)

epsilon = 1e-4
hump1 = hump_diff(xr(t), x_impact-epsilon, x_impact+epsilon, steep)
eom = sm.Matrix([
    xr(t).diff(t) - vr(t),
    m*vr(t).diff(t) + mu*m*g - F(t)*(1-step_diff(xr(t), x_impact, steep)),
    xl(t).diff(t) - vl(t),
    m*vl(t).diff(t) - mu*m*g,
    aux1(t)- (vl(t) +  e*vr(t))*hump1 - ((vl(t) - vl(t))*(1-hump1)),
    aux2(t) -  (xl(t) -  xr(t))*hump1 - ((xl(t) - xl(t))*(1-hump1)),
    T(t).diff(t) - 1.0*(1-step_diff(xr(t), x_impact, steep)) + 1.0*step_diff(xr(t), x_impact, steep),
])
sm.pprint(eom)


# %%
# Set instance constraints and add to the bounds defined above.
instance_constraints = (
        xr(0*h) - 0.0,
        T(0*h) - 0.0,
        vr(0*h) - 0.0,
        xl(dur) - 0.0,
)

delta = 0.001
bounds = bounds | {aux1(t): (-delta, delta), aux2(t): (-delta, delta)}

# Create an optimization problem.
prob = Problem(obj, obj_grad, eom, state_symbols, num_nodes, h,
        known_parameter_map=par_map,
        instance_constraints=instance_constraints,
        time_symbol=t,
        bounds=bounds)

prob.add_option('max_iter', 40000)

initial_guess = np.zeros(prob.num_free)
initial_guess[: 4*num_nodes] = solution[: 4*num_nodes]
initial_guess[4*num_nodes:6*num_nodes] = np.zeros(2*num_nodes)
initial_guess[6*num_nodes:7*num_nodes] = solution[4*num_nodes:5*num_nodes]
initial_guess[-1] = solution[-1]

# Find the optimal solution.
solution, info = prob.solve(initial_guess)
print(info['status_msg'])
print(f"Smallest variable time interval {info['obj_val']:.4e} sec")
duration2 = solution[-1] * (num_nodes - 1)

# %%
# Plot the constraint violations.
_ = prob.plot_constraint_violations(solution, subplots=True)

# %%
# Plot the objective function as a function of optimizer iteration.
_ = prob.plot_objective_value()

# %%
ax = prob.plot_trajectories(solution)
for i in range(6*num_nodes, 7*num_nodes-1):
    try:
        if solution[i+1] < solution[i]:
            wert = i
            raise IndexError
    except IndexError:
        break

for i in range(len(ax)):
        ax[i].axvline((wert-6*num_nodes)*solution[-1], color='k', linestyle='--')
ax[-1].set_xlabel((f'time [s] \n The vertical dashed lines give the time '
                   f'of impact'))
print(f'Improvement over (good) initial guess: {(duration1 - duration2)/duration1*100:.2f} %')
plt.show()