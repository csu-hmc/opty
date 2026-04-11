# %%
r"""
Particle Through Box
====================

Objectives
----------

- Show how to use variable bounds.
- Show how to use the `eom_bounds` to restrict a combination of states and
  inputs.

Description
-----------

In the horizonal X/Y planeA particle must move from the point (0, 0) to the
point (10, 10) as fast a possible. The controls are forces in x / y direction,
:math:`f_x` and :math:`f_y`. *At least* during some predescribed time interval,
the particle must be in a box defined by
:math:`x \in [x_{\min}, x_{\max}]` and :math:`y \in [y_{\min}, y_{\max}]`.
This is achieved by using variable bounds, which are fully open outside the
time interval and are restricted to the box inside the time interval.

The particle is subject to speed dependent friction.

The norm of the control force :math:`| \vec{F}| = |(f_x, f_y)|`
is limited by adding :math:`\sqrt{f_x^2 + f_y^2}` as an additional equation
of motion and using eom_bounds to restrict it.

**States**

- :math:`x` : x position of the particle
- :math:`y` : y position of the particle
- :math:`u_x` : x velocity of the particle
- :math:`u_y` : y velocity of the particle

**Controls**

- :math:`f_x` : x component of the force acting on the particle
- :math:`f_y` : y component of the force acting on the particle

**Parameters**

- :math:`m` : mass of the particle
- :math:`\mu` : friction coefficient
- :math:`g` : gravitational acceleration

"""

import os
import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
import matplotlib.pyplot as plt
from opty import Problem
from opty.utils import MathJaxRepr
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation


# %%
# Equations of Motion
# -------------------

N = me.ReferenceFrame('N')
O, P = me.Point('O'), me.Point('P')
O.set_vel(N, 0)
t = me.dynamicsymbols._t
x, y, ux, uy = me.dynamicsymbols('x y ux uy')
fx, fy = me.dynamicsymbols('fx fy')
P.set_pos(O, x * N.x + y * N.y)
P.set_vel(N, ux * N.x + uy * N.y)

m, mu, g = sm.symbols('m mu g')
Pa = me.Particle('Pa', P, m)
bodies = [Pa]
forces = [(P, (fx - mu * m * g) * N.x + (fy - mu * m * g) * N.y)]

kd = sm.Matrix([
    ux - x.diff(t),
    uy - y.diff(t)
])

kane = me.KanesMethod(N, q_ind=[x, y], u_ind=[ux, uy], kd_eqs=kd)
fr, frstar = kane.kanes_equations(bodies, forces)
eom = kd.col_join(fr + frstar)
eom = eom.col_join(sm.Matrix([sm.sqrt(fx**2 + fy**2)]))
MathJaxRepr(eom)

# %%
# Set up The Optimization Problem
# -------------------------------

h = sm.symbols('h')
state_symbols = [x, y, ux, uy]
num_nodes = 501
t0, tf = 0.0, h * (num_nodes - 1)
interval_value = h

par_map = {}
par_map[m] = 1.0
par_map[mu] = 0.5
par_map[g] = 9.81

# %%
# The objective is to minimize the total duration.


def obj(free):
    return free[-1]


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad


# %%
# Set the instance constraints.
instance_constraints = [
    x.func(t0) - 0.0,
    y.func(t0) - 0.0,
    ux.func(t0) - 0.0,
    uy.func(t0) - 0.0,
    x.func(tf) - 10.0,
    y.func(tf) - 10.0,
    ux.func(tf) - 0.0,
    uy.func(tf) - 0.0,
]

# %%
# Limit the norm of the control force to 25 N.
eom_bounds = {4: (0.0, 25.0)}

# %%
# Set the variable bounds for x and y.
#
# The center of the time interval is t = teil * h.
teil = int(num_nodes / 2)

# %%
# The width of the time interval where the particle must be in the box
# is 2 * delta.
delta = max(int(num_nodes / 20), 1)

# %%
# Outside the time interval, the bounds are fully open. Inside the time
# interval,the bounds are as :math:`x \in [low_x, hi_x]` and
# :math:`y \in [low_y, hi_y]`.

grenze1 = np.inf
low_x = 10.0
hi_x = 13.0
low_y = 1.0
hi_y = 3.0

low_bound_x = np.array([-grenze1 for _ in range(0, teil - delta)] +
                       [low_x for _ in range(teil - delta, teil + delta)] +
                       [-grenze1 for _ in range(teil + delta, num_nodes)])
up_bound_x = np.array([grenze1 for _ in range(0, teil - delta)] +
                      [hi_x for _ in range(teil - delta, teil + delta)] +
                      [grenze1 for _ in range(teil + delta, num_nodes)])

low_bound_y = np.array([-grenze1 for _ in range(0, teil - delta)] +
                       [low_y for _ in range(teil - delta, teil + delta)] +
                       [-grenze1 for _ in range(teil + delta, num_nodes)])
up_bound_y = np.array([grenze1 for _ in range(0, teil - delta)] +
                      [hi_y for _ in range(teil - delta, teil + delta)] +
                      [grenze1 for _ in range(teil + delta, num_nodes)])

bounds = {
    h: (0.0, 1.0),
    x: (low_bound_x, up_bound_x),
    y: (low_bound_y, up_bound_y)
}

# %%
# Create the Problem.
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
    eom_bounds=eom_bounds,
    backend='numpy',
    time_symbol=t,
)


# %%
# Solve the Problem.
# Use the solution of a previous iteration as initial guess if available.

fname = f"particle_through_box_{num_nodes}_nodes_solution.csv"
if os.path.exists(fname):
    initial_guess = np.loadtxt(fname)
else:
    np.random.seed(0)
    initial_guess = np.random.rand(prob.num_free)
    initial_guess[-1] = 0.01

solution, info = prob.solve(initial_guess)
print(info['status_msg'])

# %%
# Plot the results. Note the orange dotted lines, which show the active
# bounds for x and y.

_ = prob.plot_trajectories(solution, show_bounds=True)

# %%
# Plot the constraint violations.

_ = prob.plot_constraint_violations(solution, subplots=True, show_bounds=True)


# %%
# Plot the objective value.

_ = prob.plot_objective_value()

# %%
# Animation
# ---------
fps = 15

resultat, inputs, *_, h_val = prob.parse_free(solution)
resultat = resultat.T
inputs = inputs.T
tf = h_val * (num_nodes - 1)
t_arr = np.linspace(t0, tf, num_nodes)
state_sol = interp1d(t_arr, resultat, kind='cubic', axis=0)
input_sol = interp1d(t_arr, inputs, kind='cubic', axis=0)

t_in = (teil - delta) * h_val
t_out = (teil + delta) * h_val

pL = [key for key in par_map.keys()]
pL_vals = [par_map[key] for key in par_map.keys()]
qL = [x, y, ux, uy] + [fx, fy]

# Define the end of the force_vector.
arrow_head = me.Point('arrow_head')
scale = 5.0
arrow_head.set_pos(P, fx / scale * N.x + fy / scale * N.y)

# Get the coordinates of the points in the inertial frame for plotting.
coords = P.pos_from(O).to_matrix(N)
coords = coords.row_join(arrow_head.pos_from(O).to_matrix(N))
coords_lam = sm.lambdify(qL + pL, coords, cse=True)


fig, ax = plt.subplots(figsize=(7, 7), layout='constrained')

min_x = min(low_x, np.min(resultat[:, 0])) - 1.0
max_x = max(hi_x, np.max(resultat[:, 0])) + 1
min_y = min(low_y, np.min(resultat[:, 1])) - 1.0
max_y = max(hi_y, np.max(resultat[:, 1])) + 1
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)

ax.scatter(0.0, 0.0, s=25, color='blue')
ax.scatter(10.0, 10.0, s=25, color='green')
ax.set_aspect('equal', adjustable='box')

arrow = FancyArrowPatch([0.0, 0.0], [0.0, 0.0],
                        arrowstyle='-|>',     # nicer arrow head
                        mutation_scale=20,    # makes head bigger
                        linewidth=1,
                        color='green')
ax.add_patch(arrow)

# the box the particle must go through
ax.vlines(low_x, low_y, hi_y, colors='grey')
ax.vlines(hi_x, low_y, hi_y, colors='grey')
ax.hlines(low_y, low_x, hi_x, colors='grey')
ax.hlines(hi_y, low_x, hi_x, colors='grey')

XX = np.linspace(low_x, hi_x, 100)
YY = [hi_y for _ in range(100)]
ax.fill_between(XX, YY, y2=low_y, color='black', alpha=0.25)


partic = ax.scatter([], [], s=50, color='red')
intime = ax.scatter([], [], s=25, color='red', marker='*')
followed_path, = ax.plot([], [], color='red', linewidth=0.5)

x_values, y_values = [], []
x_inbox, y_inbox = [], []


def update(frame):
    t = frame
    coords_vals = coords_lam(*state_sol(t)[0:], *input_sol(t)[0:2],
                             *pL_vals)
    ax.set_title(
        f"Running time: {t:.2f} s \n Particle "
        f"must be in the grey box at least from {t_in:.2f} sec "
        f"to {t_out:.2f} sec \n Green arrow shows the force vector, "
        f"norm = "
        f"{np.linalg.norm([coords_vals[0, 1], coords_vals[1, 1]]):.2f}"
        f" N \n The red stars are the path for {t_in:.2f} sec "
        f"to {t_out:.2f} sec"
    )

    arrow.set_positions(np.array([coords_vals[0, 0], coords_vals[1, 0]]),
                        np.array([coords_vals[0, 1], coords_vals[1, 1]]))
    partic.set_offsets([coords_vals[0, 0], coords_vals[1, 0]])

    x_values, y_values = [], []
    x_inbox, y_inbox = [], []
    for zeit in np.concatenate((np.arange(0, t, 1.0/fps), np.array([t]))):
        coords_vals = coords_lam(*state_sol(zeit)[0:], *input_sol(zeit)[0:2],
                                 *pL_vals)
        x_values.append(coords_vals[0, 0])
        y_values.append(coords_vals[1, 0])
        if t_in <= zeit <= t_out:
            x_inbox.append(coords_vals[0, 0])
            y_inbox.append(coords_vals[1, 0])
    followed_path.set_data(x_values, y_values)
    intime.set_offsets(np.column_stack((x_inbox, y_inbox)))

    return arrow, partic, followed_path, intime


ani = FuncAnimation(fig, update,
                    frames=np.concatenate((np.arange(0, tf, 1.0/fps),
                                           np.array([tf]))),
                    interval=1500/fps, blit=False)

plt.show()
