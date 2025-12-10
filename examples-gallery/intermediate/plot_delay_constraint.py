# %%

r"""
Delay Constraint
================

Objective
---------

- Show how to handle a somewhat unusual constraint using numerical
  trajectories

Description
-----------

A particle of mass :math:`m` should move from 0 to 10 in minimum time.
It is pushed by a force :math:`P`, and there is speed dependent friction, also.
There is a constraint of the amount of energy that may be used in any
``time_delay`` period, that is
:math:`\int_{t-time_{\textrm{delay}}}^{t} Power(\tau) d\tau \leq E_{max}`,
e.g. the battery of the vehicle can only
deliver so much energy in a given time period without overheating. As
breaking does not cool the battery, only positive power is considered.

Explanation
-----------

- The function :math:`\frac{d}{dt} W= \frac{1}{s} \ln(e^{s \cdot
  \textrm{Power}(t)} + 1)
  \approx \max(0, \textrm{Power}(t))` is introduced to only count positive
  power, where :math:`\textrm{Power(t)} = P(t) \cdot u_x` in the case on hand.
  ``s`` is a sharpness factor to tune how tight the bend is of the softplus
  function, see some plots below.
- :math:`W_{delay}(t)` Is a copy of :math:`W` with a delay of
  :math:`time_{\textrm{delay}}` seconds. This will be a numerical trajectory.
  :math:`W_{diff}(t)` is the amount of energy that is used in the last
  :math:`time_{\textrm{delay}}` seconds.
  :math:`W_{diff}(t) = W(t) - W_{delay}(t)`
- The derivative of :math:`W_{\textrm{delay}}` with respect to :math:`P`
  is needed. (For a numerical trajectory the derivative w.r.t. the relevant
  variable must be provided). This may be done as follows:

  - Set :math:`WdtdP : = \dfrac{\partial}{\partial{t}} \dfrac{\partial}
    {\partial{P}}W(t) = \dfrac{\partial}{\partial{P}} \left[ \frac{1}{s}
    \ln(e^{s\cdot P(t) \cdot u_x(t)} + 1)\frac{1}{s}  \ln(e^{s\cdot P(t)
    \cdot u_x(t)} + 1)\right]`
  - Set :math:`\dfrac{d}{dt} WdP = WdtdP`
  - then :math:`WdP = \dfrac{\partial}{\partial{P}}W(t)`, which is needed.

Notes
-----

- This example is inspired by a problem from Neville Nieman. Neville
  also had the basic idea with ``softplus`` to only count positive power.
- Convergence is difficult, e.g. with ``sharpness > 1.0`` it does not converge
  well.
- By definition: :math:`W_{diff}(t) \geq 0.0`. However, setting this as a
  lower bound makes convergence worse. It seems that some "playing around"
  with "obvious" things is needed often with nonlinear optimization.

"""
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from opty import Problem
from opty.utils import MathJaxRepr

# sphinx_gallery_thumbnail_number = 4

# %%
# Equations of Motion
# -------------------
x, ux = me.dynamicsymbols('x ux')
P, F = me.dynamicsymbols('P, F')

W = me.dynamicsymbols('W')
W_delay = sm.symbols('W_delay', cls=sm.Function)(P)
W_diff = sm.symbols('W_diff', cls=sm.Function)

WdtdP = sm.symbols('WdtdP', cls=sm.Function)(P)
WdP = me.dynamicsymbols('WdP')

h = sm.symbols('h')
m, m0, c = sm.symbols('m, m0, c')
t = me.dynamicsymbols._t
reibung = c * sm.sin(x) + c

# %%
# *sharpness* defines how tight the softplus function is.
#
# Softplus demo available at: https://www.desmos.com/calculator/sveirxlszn
sharpness = 1.0
ddt_W_func = 1 / sharpness * sm.ln(sm.exp(sharpness * (P * ux)) + 1)
ddt_W_numeric = sm.lambdify((P, ux), ddt_W_func, "numpy")

# %%
# Free order is used in the functions defining the numerical trajectories.
# It must match the order used in the Problem definition. This is
# verified further down.
free_order = [x, ux, W, W_diff(t), WdP, WdtdP, P, h]

# %%
# Plot softplus and its derivatives w.r.t. P for different sharpness values
sharpness1 = sm.symbols('sharpness1')
ddt_W_func1 = 1 / sharpness1 * sm.ln(sm.exp(sharpness1 * (P * ux)) + 1)
ddt_W_numeric1 = sm.lambdify((P, ux, sharpness1), ddt_W_func1, cse=True)

XX = np.linspace(-10, 10, 400)
YY = np.ones_like(XX)
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for sharpness2 in [1.0, 1.5, 10.0]:
    ZZ = ddt_W_numeric1(XX, YY, sharpness2)
    ax[0].plot(XX, ZZ, label=f'sharpness={sharpness2}')
    ax[0].set_title('softplusfor different sharpness values')
ax[0].legend()

ddt_W_func1 = ((1 / sharpness1 *
                sm.ln(sm.exp(sharpness1 * (P * ux)) + 1)).diff(P))
ddt_W_numeric1 = sm.lambdify((P, ux, sharpness1), ddt_W_func1, cse=True)
for sharpness2 in [1.0, 1.5, 10.0]:
    ZZ = ddt_W_numeric1(XX, YY, sharpness2)
    ax[1].plot(XX, ZZ, label=f'sharpness={sharpness2}')
ax[1].set_title(f'derivative of softplus \n'
                'for different sharpness values')
ax[1].legend()

ddt_W_func1 = ((1 / sharpness1 *
                sm.ln(sm.exp(sharpness1 * (P * ux)) + 1)).diff(P, 2))
ddt_W_numeric1 = sm.lambdify((P, ux, sharpness1), ddt_W_func1, cse=True)
for sharpness2 in [1.0, 1.5, 10.0]:
    ZZ = ddt_W_numeric1(XX, YY, sharpness2)
    ax[2].plot(XX, ZZ, label=f'sharpness={sharpness2}')
ax[2].set_title(f'second derivative of softplus \n'
                'for different sharpness values')
_ = ax[2].legend()

# %%
# Form :math:`\dfrac{\partial}{\partial{P}} \left[ \frac{1}{s}
# \ln(e^{s\cdot P(t) \cdot u_x(t)} + 1)\frac{1}{s}  \ln(e^{s\cdot P(t)
# \cdot u_x(t)} + 1)\right]`

ddtdP_W_func = ((1 / sharpness *
                 sm.ln(sm.exp(sharpness * (P * ux)) + 1)).diff(P))
ddtdP_W_numeric = sm.lambdify((P, ux), ddtdP_W_func, cse=True)
# %%
# Equations of Motion
eom = sm.Matrix([
    # physical equations of motion
    ux - x.diff(t),
    (P / m - reibung * ux) / m - ux.diff(t),
    # functions needed for the delay.
    W.diff(t) - ddt_W_func,
    W_diff(t) - (W - W_delay),
    WdtdP - ddtdP_W_func,
    WdP.diff(t) - ddtdP_W_func,
])

MathJaxRepr(eom)

# %%
# Set the values, dictionaries, functions needed for Problem.

state_symbols = (x, ux, W, W_diff(t), WdP, WdtdP)
num_nodes = 201
t0, tf = 0.0 * h, (num_nodes - 1) * h
time_delay = 0.5

par_map = {
    m: 1.0,
    c: 0.1,
}

instance_constraints = (
    W.func(t0) - 0.0,
    W_diff(t0) - 0.0,
    x.func(t0) - 0.0,
    ux.func(t0) - 0.0,
    x.func(tf) - 10.0,
    ux.func(tf) - 0.0,
)

bounds = {
    h: (0.0, 0.5),
    P: (-10, 10.0),
    W_diff(t): (-np.inf, 20.0),
}


def obj(free):
    return free[-1]


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad

# %%
# Create the known trajectories as explained above.


def W_delay_traj(free):
    time = np.linspace(0, free[-1] * (num_nodes - 1), num_nodes)
    delayed_time = np.clip(time - time_delay, 0, None)
    W_index = free_order.index(W)
    W_arr = free[W_index * num_nodes: (W_index + 1) * num_nodes]
    W_delay_arr = np.interp(delayed_time, time, W_arr)
    return W_delay_arr


def Wdt_delay_traj(free):
    time = np.linspace(0, free[-1] * (num_nodes - 1), num_nodes)
    delayed_time = np.clip(time - time_delay, 0, None)
    WdP_index = free_order.index(WdP)
    WdP_arr = free[WdP_index * num_nodes: (WdP_index + 1) * num_nodes]
    Wdt_delay_arr = np.interp(delayed_time, time, WdP_arr)
    return Wdt_delay_arr


known_traj_map = {
    W_delay: W_delay_traj,
    W_delay.diff(P): Wdt_delay_traj,
}

# %%
# Just to get prob.num_free, correct order.

prob = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    num_nodes,
    h,
    known_parameter_map=par_map,
    instance_constraints=instance_constraints,
    time_symbol=t,
    bounds=bounds,
    known_trajectory_map=known_traj_map,
    backend="numpy",
)

# %%
# Verify the free order matches extraction indices.
if list(prob._extraction_indices.keys()) != free_order:
    print(f"used: {free_order}")
    print(f"Correct: \n{list(prob._extraction_indices.keys())}")
    raise ValueError("Free order does not match extraction indices.")
else:
    print(f"Correct free_order:\n{free_order}")

# %%
# Find the solution

initial_guess = np.ones(prob.num_free)

prob = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    num_nodes,
    h,
    known_parameter_map=par_map,
    instance_constraints=instance_constraints,
    time_symbol=t,
    bounds=bounds,
    known_trajectory_map=known_traj_map,
    backend="cython",
)
prob.add_option("max_iter", 5000)
prob.add_option("mumps_mem_percent", 16000)

solution, info = prob.solve(initial_guess)
print(info["status_msg"])
print(info["obj_val"])

# %%
# Plot objective value
_ = prob.plot_objective_value()

# %%
# Plot the constraint violations
_ = prob.plot_constraint_violations(solution, subplots=True)

# %%
# Plot the trajectories
fig, ax = plt.subplots(9, 1, figsize=(6.4, 12), layout='constrained',
                       sharex=True)
_ = prob.plot_trajectories(solution, show_bounds=True, axes=ax)
