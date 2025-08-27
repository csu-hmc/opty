# %%
r"""
Particle around Obstructive Hole
================================

Objective
---------

- Show how to use implicitly known trajectories in connection with bounds on
  equations of motion.

Description
-----------

A particle is to move from point A to point B while avoiding an obstructive
hole, or impenetrable area, or something like this. It is driven by a force
:math:`F = \begin{pmatrix} f_x \\ f_y \end{pmatrix}`. It should move as fast
as possible with low energy consumption.

The border of the hole is assumed to be known only numerically.
In the simulation the measurements and their derivatives are created using
random points and connecting them with 4th order splines.

It is further assumed, that the terrain becomes harder as the particle moves
away from the hole in y direction. This is modeled as as speed and y dependent
force: :math:`F_{\textrm{terrain}} =
-\textrm{reibung} \cdot y(t)^2 \cdot \begin{pmatrix} \dfrac{dx(t)}{dt} \\
\dfrac{dy(t)}{dt} \end{pmatrix}`

Two undefined sympy functions ``top_edge`` and ``bottom_edge`` are introduced
to represent the upper and lower boundaries of the hole. They (arbitrarily)
meet at y = 0, and the y - coordinate of the particle is restricted by two
additional equations of motion which are bounded.

Notes
-----

- A :math:`4^{th}` order spline is used, as previous trials have shown that the
  convergence behavior of opty tends to be better.
- The boundaries of the hole are extended, so the bounded equations of motion
  are effectively not restricting the particle outside the area of the hole.
- Convergence is quite difficult. At the points on the x axis where the
  upper and lower half of the borders of the hole meet, the derivatives change
  discontinuously, unavoidably so. Maybe this is a reason for the difficult
  convergence.
- The initial guess given favors a clockwise direction of the particle around
  the hole.

**States**

- :math:`x` : x coordinate of the particle
- :math:`y` : y coordinate of the particle
- :math:`u_x` : x velocity of the particle
- :math:`u_y` : y velocity of the particle


**Inputs**

- :math:`f_x` : x component of the driving force
- :math:`f_y` : y component of the driving force

**Parameters**

- :math:`m` : mass of the particle
- :math:`\textrm{reibung}` : factor to describe the roughness of the terrain,
  see above.
- :math:`g` : gravitational acceleration
- :math:`\epsilon` : minimal allowed vertical distance of the particle
  from the hole

**Others**

- :math:`\textrm{weight}` : scalar giving the relative importance of speed
  vs. energy consumption in the objective function.

 """
import os
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline
from opty import Problem
from opty.utils import MathJaxRepr
from scipy.optimize import fsolve
from matplotlib import animation

# sphinx_gallery_thumbnail_number=1

# %%
# Set the hole to be avoided.
#
# In real applications, these would be measured values, connected by higher
# order splines, like done here.

x_meas = np.linspace(-21, 21, 500)

np.random.seed(4)
XX = np.linspace(-21, 21, 14)

sicher = 100
Y_top = np.concatenate((np.array([-sicher, -sicher]),
                        np.random.uniform(5, 10, 10),
                        np.array([-sicher, -sicher])))

inter_top = UnivariateSpline(XX, Y_top, k=4, s=0)

Y_bottom = np.concatenate((np.array([sicher, sicher]),
                           np.random.uniform(-10, -5, 10),
                           np.array([sicher, sicher])))
inter_bottom = UnivariateSpline(XX, Y_bottom, k=4, s=0)

y_top_meas = inter_top(x_meas)
ydt_top_meas = inter_top.derivative()(x_meas)

y_bottom_meas = inter_bottom(x_meas)
ydt_bottom_meas = inter_bottom.derivative()(x_meas)

fig, ax = plt.subplots(2, 1, figsize=(6.4, 8), layout='constrained')
ax[0].set_xlim(-25, 25)
ax[0].set_ylim(-25, 25)
ax[0].scatter([0, 10], [-11, 15], color='red', s=100)
ax[0].axhline(0, color='black', lw=0.5)
ax[0].plot(XX, Y_top, 'o', label='data points')
ax[0].plot(XX, Y_bottom, 'o', label='data points bottom')
ax[0].plot(x_meas, y_top_meas, label='Cubic Spline')
ax[0].plot(x_meas, y_bottom_meas, label='Cubic Spline Bottom')
ax[0].fill_between(x_meas, y_top_meas, y_bottom_meas,
                   where=(y_top_meas > y_bottom_meas),
                   interpolate=True, color='gray', alpha=0.5)
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[1].set_xlabel('x')
ax[1].set_ylabel('dy/dx')
ax[0].legend()
ax[0].set_title('Terrain Representation')
ax[1].set_title('Terrain Derivatives')
ax[1].plot(x_meas, ydt_top_meas, label="derivative of top curve")
ax[1].plot(x_meas, ydt_bottom_meas, label="derivative of bottom curve")
_ = ax[1].legend()

# %%
# Find the approx. x coordinates where the splines meet the x-axis
# Needed for plotting only.


def Null_bottom_x(x):
    return inter_bottom(x)


x0 = 15.0
loesung = fsolve(Null_bottom_x, x0)

x0 = -15.0
loesung = fsolve(Null_bottom_x, x0)


def Null_top_x(x):
    return inter_top(x)


x0 = 15.0
loesung_r = fsolve(Null_top_x, x0)

x0 = -15.0
loesung_l = fsolve(Null_top_x, x0)

# for plotting the 'artificial' measurement points away from the hole in a
# lighter shade.

x_meas_left = []
x_meas_right = []
x_meas_between = []
for ort in x_meas:
    if ort <= loesung_l:
        x_meas_left.append(ort)
    elif ort >= loesung_r:
        x_meas_right.append(ort)
    else:
        x_meas_between.append(ort)

x_meas_left = x_meas_left + [x_meas_between[0]]
x_meas_right = [x_meas_between[-1]] + x_meas_right

# %%
# Equations of Motion
# -------------------


def smooth_step(x, a, steepness=50):
    """returns 1 for x >= a, 0 for x < a"""
    return 0.5 * (sm.tanh(steepness * (x - a)) + 1.0)


N = me.ReferenceFrame('N')
O, P = me.Point('O'), me.Point('P')
t = me.dynamicsymbols._t
O.set_vel(N, 0)

x, y, ux, uy = me.dynamicsymbols('x y u_x u_y')
fx, fy = me.dynamicsymbols('f_x f_y')

# Functions for the edges of the hole
top_edge = sm.Function('top_edge')(x)
top_edge_dx = sm.Function('top_edge_dx')(x)
bottom_edge = sm.Function('bottom_edge')(x)
bottom_edge_dx = sm.Function('bottom_edge_dx')(x)

m, epsilon, reibung = sm.symbols('m epsilon reibung')

P.set_pos(O, x * N.x + y * N.y)
P.set_vel(N, ux * N.x + uy * N.y)
body = me.Particle('body', P, m)
bodies = [body]

forces = [(P, fx * N.x + fy * N.y - reibung * y**2 * (ux * N.x + uy * N.y))]

kd = sm.Matrix([ux - x.diff(t), uy - y.diff(t)])
KM = me.KanesMethod(N, q_ind=[x, y], u_ind=[ux, uy], kd_eqs=kd)
fr, frstar = KM.kanes_equations(bodies, forces)
eom = kd.col_join(fr + frstar)

# %%
# Add the equations which enforce that the particle does not fall into the
# hole. smooth_step is needed, so only one of them is active.

eom = eom.col_join(sm.Matrix([
    (bottom_edge - y - epsilon) * smooth_step(-y, 0),
    (y - top_edge - epsilon) * smooth_step(y, 0),
]))

MathJaxRepr(eom)

# %%
# Set up the Problem and Solve it
# -------------------------------

state_symbols = [x, y, ux, uy]
num_nodes = 1501
h = sm.symbols('h')
interval_value = h
t0, tf = 0.0, h * (num_nodes - 1)

par_map = {}
par_map[m] = 1.0
par_map[epsilon] = 1.0
par_map[reibung] = 0.01

instance_constraints = [
    x.func(t0),
    y.func(t0) + 11.0,
    ux.func(t0),
    uy.func(t0),
    x.func(tf) - 10.0,
    y.func(tf) - 15.0,
    ux.func(tf),
    uy.func(tf)
]

loc_lim = 20.0
limit = 50.0
bounds = {
    x: (-loc_lim, loc_lim),
    y: (-loc_lim, loc_lim),
    fx: (-limit, limit),
    fy: (-limit, limit),
    h: (0.0, 1.0),
}

eom_bounds = {
    4: (0.0, 25.0),   # bottom edge constraint
    5: (0.0, 25.0)    # top edge constraint
}

# weight sets the relative importance of saving energy vs. speed
weight = 500


def obj(free):
    summe = (np.sum(free[4*num_nodes:6*num_nodes]**2) * free[-1] +
             weight * free[-1])
    return summe


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[4*num_nodes:6*num_nodes] = (2 * free[4*num_nodes:6*num_nodes] *
                                     free[-1])
    grad[-1] = np.sum(free[4*num_nodes:6*num_nodes]**2) + weight
    return grad


# %%
# Set the measured values of the hole.


def calc_top_edge(free):
    x = free[0: num_nodes]
    return np.interp(x, x_meas, y_top_meas)


def calc_bottom_edge(free):
    x = free[0: num_nodes]
    return np.interp(x, x_meas, y_bottom_meas)


def calc_top_edge_dx(free):
    x = free[0: num_nodes]
    return np.interp(x, x_meas, ydt_top_meas)


def calc_bottom_edge_dx(free):
    x = free[0: num_nodes]
    return np.interp(x, x_meas, ydt_bottom_meas)


# %%
# Form the Problem

# %%
prob = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    num_nodes,
    interval_value,
    known_parameter_map=par_map,
    known_trajectory_map={
        top_edge: calc_top_edge,
        bottom_edge: calc_bottom_edge,
        top_edge.diff(x): calc_top_edge_dx,
        bottom_edge.diff(x): calc_bottom_edge_dx,
    },
    instance_constraints=instance_constraints,
    bounds=bounds,
    eom_bounds=eom_bounds,
    time_symbol=t,
    backend='numpy'
)

# %%
# Solve the problem
#
# Use the existing solution if available, else solve the problem.

fname = f'particle_avoid_hole_{num_nodes}_nodes_solution.csv'
if os.path.exists(fname):
    # Use existing solution
    solution = np.loadtxt(fname)

else:
    # calculate the solution using a reasonable initial guess

    initial_guess = np.ones(prob.num_free) * 0.1
    x_guess = np.concatenate([np.linspace(0.0, -19.0, num_nodes // 2),
                              np.linspace(-19.0, 10, num_nodes -
                                          num_nodes // 2)])
    y_guess = np.concatenate((
        np.linspace(-11.0, -19.0, num_nodes // 3),
        np.linspace(-19.0, 19.0, num_nodes // 3),
        np.linspace(19.0, 15, num_nodes - (num_nodes // 3) - (num_nodes // 3))
    ))
    initial_guess[0: num_nodes] = x_guess
    initial_guess[num_nodes: 2 * num_nodes] = y_guess
    initial_guess[-1] = 0.01
    prob.add_option('max_iter', 80000)
    for _ in range(2):
        solution, info = prob.solve(initial_guess)
        print(info['status_msg'])
        initial_guess = solution
    prob.plot_objective_value()
    np.savetxt(fname, solution, fmt='%.12f')

# %%
# Plot the results
fig, ax = plt.subplots(10, 1, figsize=(6.5, 15), layout='constrained')
prob.plot_trajectories(solution, axes=ax)
ax[-1].axhline(0, color='black', lw=0.5, linestyle='--')
_ = ax[-2].axhline(0, color='black', lw=0.5, linestyle='--')

# %%
# Plot the constraint violations
_ = prob.plot_constraint_violations(solution)


# %%
# Animation
# ---------

fps = 0.5

state, input, _, h_val = prob.parse_free(solution)

t0, tf = 0.0, h_val * (num_nodes - 1)
t_arr = np.linspace(t0, tf, num_nodes)
state_sol = CubicSpline(t_arr, state.T)
input_sol = CubicSpline(t_arr, input.T)

f_head = me.Point('f_head')
fx1, fy1 = sm.symbols('f_x1 f_y1')
f_head.set_pos(O, fx1 * N.x + fy1 * N.y)

coordinates = P.pos_from(O).to_matrix(N)
coordinates = coordinates.row_join(f_head.pos_from(O).to_matrix(N))


coords_lam = sm.lambdify(state_symbols + [fx1, fy1],
                         coordinates, cse=True)


def init_plot():
    fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
    ax.set_xlim(np.min(state.T[:, 0])-1, np.max(state.T[:, 0])+1)
    ax.set_ylim(np.min(state.T[:, 1])-1, np.max(state.T[:, 1])+1)
    ax.set_xlim(-21, 21)
    ax.set_ylim(-21, 21)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.scatter([10], [15], color='green', s=20)
    ax.scatter([0], [-11], color='red', s=20)

    # Plot the hole
    ax.plot(x_meas_left, inter_bottom(x_meas_left), color='black', lw=0.5,
            linestyle='--')
    ax.plot(x_meas_right, inter_bottom(x_meas_right), color='black', lw=0.5,
            linestyle='--')
    ax.plot(x_meas_between, inter_bottom(x_meas_between), color='black',
            lw=1.0)

    ax.plot(x_meas_left, inter_top(x_meas_left), color='black', lw=0.5,
            linestyle='--')
    ax.plot(x_meas_right, inter_top(x_meas_right), color='black', lw=0.5,
            linestyle='--')
    ax.plot(x_meas_between, inter_top(x_meas_between), color='black', lw=1.0)
    ax.fill_between(x_meas, y_top_meas, y_bottom_meas,
                    where=(y_top_meas > y_bottom_meas),
                    interpolate=True, color='gray', alpha=0.5)

    # Iitiate the artists.
    line1, = ax.plot([], [], color='red', lw=0.5)
    point = ax.scatter([], [], color='red', s=100, marker='o')
    pfeil = ax.quiver([], [], [], [], color='green', scale=90, width=0.004)
    return fig, ax, point, pfeil, line1


fig, ax, point, pfeil, line1 = init_plot()


def update(t):

    message = (
        f'Running time {t:.2f} sec \n'
        f'The light lines show the extension of the terrain \n'
        f'to make the eom_bounds work everywhere \n'
        f'Video about 10 times natural speed'
        )

    ax.set_title(message, fontsize=10)

    coords = coords_lam(*state_sol(t), *input_sol(t))
    point.set_offsets([coords[0, 0], coords[1, 0]])
    pfeil.set_offsets([coords[0, 0], coords[1, 0]])
    pfeil.set_UVC([coords[0, 1] - coords[0, 0]],
                  [coords[1, 1] - coords[1, 0]])

    # Draw the tracing line
    old_x, old_y = [], []
    for zeit in np.arange(0.0, t_arr[-1], 1/fps):
        if zeit <= t:
            coords = coords_lam(*state_sol(zeit), *input_sol(zeit))
            old_x.append(coords[0, 0])
            old_y.append(coords[1, 0])
        else:
            break
    line1.set_data(old_x, old_y)


ani = animation.FuncAnimation(fig, update,
                              frames=np.arange(0.0, t_arr[-1], 1/fps),
                              interval=100/fps, blit=False)

plt.show()
