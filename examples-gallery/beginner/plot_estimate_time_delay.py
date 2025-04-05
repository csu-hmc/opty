# %%
r"""
Time Delay Estimation
=====================

Objective
---------

- Show how a time delay in a driving force of a mechanical system may be
  estimated from noisy measurements by introducing a state variable which
  mimicks the time.


Introduction
------------

A simple pendulum is driven by a torque of the form:
:math:`F \sin(\omega (t - \delta))`, and :math:`F, \omega, \tau`
are to be erstimated based on noisy measurements of the angle.

The driving force is set to zero for :math:`t < \delta`.

**States**

- :math:`q` : angle of the pendulum [rad]
- :math:`u` : angular velocity of the pendulum [rad/s]
- :math:`T(t)` : variable which mimicks the time t [s]


**Known parameters**

- :math:`m` : mass of the pendulum [kg]
- :math:`g` : gravity [m/s^2]
- :math:`l` : length of the pendulum [m]
- :math:`I_{zz}` : inertia of the pendulum [kg*m^2]
- :math:`steep` : steepness of the differentiable step function [1/s]


**Unknown parameters**

- :math:`F` : strength of the driving torque [Nm]
- :math:`\omega` : frequency of the driving torque [rad/s]
- :math:`\delta` : time delay of the driving torque [s]

"""
import numpy as np
import sympy as sm
from scipy.integrate import solve_ivp
import sympy.physics.mechanics as me
from opty import Problem
from opty.utils import MathJaxRepr
import matplotlib.pyplot as plt

# %%
# Set Up the System
# -----------------

N, A = sm.symbols('N A', cls=me.ReferenceFrame)
O, P = sm.symbols('O P', cls=me.Point)
q, u = me.dynamicsymbols('q u')
O.set_vel(N, 0)
t = me.dynamicsymbols._t

m, g, l, iZZ = sm.symbols('m g l iZZ', real=True)
F, omega, delta = sm.symbols('F, omega, delta', real=True)

A.orient_axis(N, q, N.z)
A.set_ang_vel(N, u*N.z)

P. set_pos(O, -l*A.y)
P.v2pt_theory(O, N, A)

inert = me.inertia(A, 0, 0, iZZ)
bodies = [me.RigidBody('body', P, A, m, (inert, P))]
# Driving force set to zero for t < delta
torque = F * sm.sin(omega*(t - delta)) * sm.Heaviside(t - delta)
forces = [(P, -m*g*N.y), (A, torque*A.z)]
kd = sm.Matrix([q.diff(t) - u])

KM = me.KanesMethod(N, q_ind=[q], u_ind=[u], kd_eqs=kd)
fr, frstar = KM.kanes_equations(bodies, forces)

MM = KM.mass_matrix_full
force = KM.forcing_full

# %%
# Convert sympy functions to numpy functions.
qL = [q, u]
pL = [m, g, l, iZZ, F, omega, delta, t]

MM_lam = sm.lambdify(qL + pL, MM, cse=True)
force_lam = sm.lambdify(qL + pL, force, cse=True)
torque_lam = sm.lambdify(qL + pL, torque, cse=True)

# %%
# Integrate numerically to get the measurements.

m1 = 1.0
g1 = 9.81
l1 = 1.0
iZZ1 = 1.0
F1 = 0.25
omega1 = 2.0
delta1 = 1.0

t1 = 0.0

q10 = 0.0
u10 = 0.0

interval = 7.5
schritte = 8000

pL_vals = [m1, g1, l1, iZZ1, F1, omega1, delta1, t1]
y0 = [q10, u10]

times = np.linspace(0, interval, schritte)
t_span = (0., interval)


def gradient(t, y, args):
    args[-1] = t
    vals = np.concatenate((y, args))
    sol = np.linalg.solve(MM_lam(*vals), force_lam(*vals))
    return np.array(sol).T[0]


resultat1 = solve_ivp(gradient, t_span, y0, t_eval=times, args=(pL_vals,))
resultat = resultat1.y.T
print('resultat shape', resultat.shape, '\n')

# %%
# Plot the results.
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(times, resultat[:, 0], label='angle q')
torque_print = []
for i in range(len(times)):
    pL_vals[-1] = times[i]
    torque_print.append(torque_lam(resultat[i, 0], resultat[i, 1], *pL_vals))
ax.plot(times, torque_print, label='torque applied')
ax.set_xlabel('time [s]')
ax.set_ylabel('angle [rad], driving torque [Nm]')
ax.set_title((f'Pendulum with torque, delta = {delta1}, $\\omega$ = {omega1}, '
              f'strength = {F1}'))
_ = ax.legend()

# %%
# Adapt the eoms for opty.

steep = 10.0
T = sm.symbols('T', cls=sm.Function)


def step_diff(x, a, steep):
    return 0.5 * (1.0 + sm.tanh(steep * (x - a)))


# %%
# Replace the nondifferentiable Heaviside function with a differentiable
# approximation.
# Add the eom, suitab le to make T(t) mimick the time t.
torque = F * sm.sin(omega*(T(t) - delta)) * step_diff(T(t), delta, steep)
forces = [(P, -m*g*N.y), (A, torque*A.z)]
kd = sm.Matrix([q.diff(t) - u])

KM = me.KanesMethod(N, q_ind=[q], u_ind=[u], kd_eqs=kd)
fr, frstar = KM.kanes_equations(bodies, forces)
eom = kd.col_join(fr + frstar)
eom = eom.col_join(sm.Matrix([T(t).diff(t) - 1.0]))
MathJaxRepr(eom)

# %%
# Set Up the Estimation Problem for opty
# --------------------------------------
np.random.seed(123)

state_symbols = [q, u, T(t)]

num_nodes = schritte
t0, tf = 0.0, interval
interval_value = (tf - t0) / (num_nodes - 1)

par_map = {}
par_map[m] = m1
par_map[g] = g1
par_map[l] = l1
par_map[iZZ] = iZZ1

# %%
# Create the noisy measurements.
measurements = (resultat.T.flatten() + np.random.normal(0, 1.0, 2 * num_nodes)
                * 0.01)


def obj(free):
    return np.sum([(measurements[i] - free[i])**2 for i in range(num_nodes)])


def obj_grad(free):
    grad = np.zeros_like(free)
    for i in range(num_nodes):
        grad[i] = -2 * (measurements[i] - free[i])
    return grad


instance_constraints = (
    q.func(t0) - q10,
    u.func(t0) - u10,
    T(t0) - t0,
)

# Give rough bounds for the parameters to be erstimated. This speeds up the
# convergence a lot.
bounds = {
    delta: (0.1, 2.0),
    omega: (1.0, 3.0),
    F: (0.1, 0.5),
}

prob = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    num_nodes,
    interval_value,
    known_parameter_map=par_map,
    instance_constraints=instance_constraints,
    bounds=bounds,
    time_symbol=t,
    backend='numpy',
)

print('Sequence of unknown parameters in solution',
      prob.collocator.unknown_parameters, '\n')

initial_guess = np.random.rand(prob.num_free)

solution, info = prob.solve(initial_guess)
print(info['status_msg'])

# %%
# Plot the trajectories
_ = prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
_ = prob.plot_constraint_violations(solution)

# %%
# Plot the objective value as a function the the iterations.
_ = prob.plot_objective_value()

# %%
# Print the results.
print((f'estimated \u03C9 = {solution[-2]:.3f}, given value = {delta1}, '
       f'hence error = {(solution[-2] - delta1)/delta1*100:.3f} %'))
print((f'estimated \u03B4 = {solution[-1]:.3f}, given value = {omega1},'
       f' hence error = {(solution[-1] - omega1)/omega1*100:.3f} %'))
print((f'estimated F = {solution[-3]:.3f}, given value = {F1},'
       f' hence error = {(solution[-3] - F1)/F1*100:.3f} %'))

# %%
# sphinx_gallery_thumbnail_number = 3
