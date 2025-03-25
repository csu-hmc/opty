r"""
Light Diffraction
=================

Objectives
----------

- Show how to solve a variational problem with opty.
- Show how to use differentiable functions to approximate their non-
  differentiable counterparts.

Introduction
------------

The bending of light is explained by the assumption that its speed is different
in different media, while it 'tries' to minimize the time it takes to travel.
More details may be found here:

https://en.wikipedia.org/wiki/Refractive_index

A particle is moved from the origin to the final point using a large force, so
the required speeds can be reached in negligible time.
The maximum speed is limited to different values in different regions.
This is accomplished by using a smooth approximation of a 'hump' function.

Notes
-----

As opty cannot use non-differentiable functions, one has to take care that with
their differentiable counter parts 'edge effects' are avoided. Here it is of
advantage to bound :math:`u_x` to be larger than zero to avoid possible
vertical movement along the edge of a region.


**states**

- :math:`x, y` : position of the particle
- :math:`u_x, u_y` : velocity of the particle
- :math:`f_{max}` : needed to bound the maximum force that can be applied
- :math:`\textrm{speed}_1, \textrm{speed}_2, \textrm{speed}_3` : needed to
  bound the speed in different regions

**controls**

- :math:`f_x, f_y` : force applied to the particle

**parameters**

- :math:`m` : mass of the particle [kg]
- :math:`a_1, a_2, a_3` : boundaries of the regions [m]
- :math:`\mu_1, \mu_2, \mu_3` : maximum speed in the regions [m/s]
- :math:`\textrm{steepness}` : steepness of the 'hump' function. The larger it
  is the steeper the sides get

"""

import os
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty import Problem
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %%
# Define the smooth bump function and plot it.
a, b, xt, k = sm.symbols('a b xt k')
steepness = 20.0


def smooth_bump(x, a, b, k):
    """gives approx 1.0 for a < xt < b, 0.0 otherwise"""
    return 0.5 * (sm.tanh(k * (x - a)) - sm.tanh(k * (x - b)))


eval_bump = sm.lambdify((xt, a, b, k), smooth_bump(xt, a, b, k))
x_vals = np.linspace(-1, 1, 100)
fig, ax = plt.subplots(figsize=(6.4, 1.75))
a = -0.5
b = 0.75
ax.plot(x_vals, eval_bump(x_vals, -0.5, 0.75, 500))
ax.set_title((f'Smooth bump function with steepness = {steepness}, min = {a}, '
             f'max = {b}'))

# Set Up the Equations of Motion
# ------------------------------
N = me.ReferenceFrame('N')
O, P = sm.symbols('O P', cls=me.Point)
P.set_vel(N, 0)
t = me.dynamicsymbols._t

x, y, ux, uy = me.dynamicsymbols('x y u_x u_y')
speed1, speed2, speed3 = me.dynamicsymbols('speed_1 speed_2 speed_3')

fx, fy = me.dynamicsymbols('f_x f_y')
f_max = me.dynamicsymbols('f_max')
m, a1, a2, a3 = sm.symbols('m a_1 a_2 a_3')
mu1, mu2, mu3 = sm.symbols('mu_1 mu_2 mu_3')

P.set_pos(O, x * N.x + y * N.y)
P.set_vel(N, ux * N.x + uy * N.y)
bodies = [me.Particle('P', P, m)]

forces = [(P, fx * N.x + fy * N.y)]

kd = sm.Matrix([ux - x.diff(t), uy - y.diff(t)])

KM = me.KanesMethod(N, q_ind=[x, y], u_ind=[ux, uy], kd_eqs=kd)
fr, frstar = KM.kanes_equations(bodies, forces)
eom = kd.col_join(fr + frstar)
# %%
# Add the constraints.
hump1 = smooth_bump(x, -1.0, a1, steepness)
hump2 = smooth_bump(x, a1, a2, steepness)
hump3 = smooth_bump(x, a2, a3, steepness)
speed_mag = sm.sqrt(ux**2 + uy**2)

eom1 = sm.Matrix([speed1 - speed_mag * hump1,
                  speed2 - speed_mag * hump2,
                  speed3 - speed_mag * hump3])
eom = eom.col_join(eom1)
eom = eom.col_join(sm.Matrix([f_max - sm.sqrt(fx**2 + fy**2)]))
print((f'The equations of motion have {sm.count_ops(eom)} operations and have'
      f' shape {eom.shape}'))

# %%
# Set Up the Optimization Problem
# -------------------------------
h = sm.symbols('h')
interval = h
num_nodes = 501
t0, tf = 0.0, h * (num_nodes - 1)

state_symbols = [x, y, ux, uy, f_max, speed1, speed2, speed3]

par_map = {m: 1.0,
           a1: 3.0,
           a2: 7.0,
           a3: 11.0,
           mu1: 5.0,
           mu2: 1.0,
           mu3: 5.0,
           }


def obj(free):
    """Objective function for the optimization problem."""
    return free[-1]


def obj_grad(free):
    """Gradient of the objective function."""
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad


instance_constraints = (
    x.func(t0) - 0.0,
    y.func(t0) - 0.0,
    ux.func(t0) - 4.0,
    uy.func(t0) - 3.0,
    x.func(tf) - 10.0,
    y.func(tf) - 10.0,
)

limit = 500.0
bounds = {
    h: (0.0, 1.0),
    x: (0.0, 10.0),
    y: (0.0, 10.0),
    ux: (0.5, np.inf),
    uy: (0.0, np.inf),
    fx: (-limit, limit),
    fy: (-limit, limit),
    f_max: (0.0, limit),
    speed1: (0.0, par_map[mu1]),
    speed2: (0.0, par_map[mu2]),
    speed3: (0.0, par_map[mu3]),
}

prob = Problem(
        obj,
        obj_grad,
        eom, state_symbols,
        num_nodes,
        interval,
        instance_constraints=instance_constraints,
        known_parameter_map=par_map,
        bounds=bounds,
        time_symbol=t,
        backend='numpy'
)

prob.add_option('max_iter', 5000)

# %%
# Use existing solution if available, else solve the problem.
fname = f'light_diffraction_{num_nodes}_nodes_solution.csv'

if os.path.exists(fname):
    # Use existing solution.
    solution = np.loadtxt(fname)
else:
    # Solve the problem. Pick a reasonable initial guess.
    initial_guess = np.ones(prob.num_free) * 0.5
    x_values = np.linspace(0, 10, num_nodes)
    y_values = np.linspace(0, 10, num_nodes)
    initial_guess[:num_nodes] = x_values
    initial_guess[num_nodes:2*num_nodes] = y_values

    for _ in range(2):
        solution, info = prob.solve(initial_guess)
        initial_guess = solution
        print(info['status_msg'])
    _ = prob.plot_objective_value()
# %%
# Plot the solution.
_ = prob.plot_trajectories(solution)
# %%
# Plot the constraint violations.
_ = prob.plot_constraint_violations(solution, subplots=True)

# %%
# Animate the Motion
# ------------------
fps = 20


def add_point_to_data(line, x, y):
    # to trace the path of the point. Copied from Timo.
    old_x, old_y = line.get_data()
    line.set_data(np.append(old_x, x), np.append(old_y, y))


state_vals, input_vals, _, h_val = prob.parse_free(solution)
tf = h_val * (num_nodes - 1)
t_arr = np.linspace(t0, tf, num_nodes)
state_sol = CubicSpline(t_arr, state_vals.T)
input_sol = CubicSpline(t_arr, input_vals.T)

# create additional point for the speed vector
arrow_head = sm.symbols('arrow_head', cls=me.Point)
arrow_head.set_pos(P, ux * N.x + uy * N.y)

coordinates = P.pos_from(O).to_matrix(N)
coordinates = coordinates.row_join(arrow_head.pos_from(O).to_matrix(N))

pL, pL_vals = zip(*par_map.items())
coords_lam = sm.lambdify((*state_symbols, fx, fy, *pL), coordinates,
                         cse=True)


def init():
    # needed to give the picture the right size.
    xmin, xmax = -1.0, 11.
    ymin, ymax = -1.0, 11.

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid()

    ax.axvline(xmin, color='black', lw=1)
    ax.axvline(par_map[a1], color='black', lw=1)
    ax.axvline(par_map[a2], color='black', lw=1)
    ax.axvline(par_map[a3], color='black', lw=1)
    ax.scatter(0.0, 0.0, color='red', s=20)
    ax.scatter(10.0, 10.0, color='green', s=20)

    yy = np.linspace(ymin, ymax, 100)
    ax.fill_betweenx(yy, xmin, par_map[a1], color='gray',
                     alpha=1.0/par_map[mu1])
    ax.fill_betweenx(yy, par_map[a1], par_map[a2], color='gray',
                     alpha=1.0/par_map[mu2])
    ax.fill_betweenx(yy, par_map[a2], par_map[a3], color='gray',
                     alpha=1.0/par_map[mu3])

    # Initialize the block
    line1 = ax.scatter([], [], color='blue', s=100)
    line2, = ax.plot([], [], color='red', lw=0.5)
    pfeil = ax.quiver([], [], [], [], color='green', scale=25, width=0.004,
                      headwidth=8)

    return fig, ax, line1, line2, pfeil


fig, ax, line1, line2, pfeil = init()


def update(t):
    message = (f'running time {t:.2f} sec \n Darker shade is slower maximum'
               f' speed \n The speed is the green arrow')
    ax.set_title(message, fontsize=12)

    coords = coords_lam(*state_sol(t), *input_sol(t), *pL_vals)

    line1.set_offsets([coords[0, 0], coords[1, 0]])
    add_point_to_data(line2, coords[0, 0], coords[1, 0])
    pfeil.set_offsets([coords[0, 0], coords[1, 0]])
    pfeil.set_UVC(coords[0, 1] - coords[0, 0], coords[1, 1] - coords[1, 0])


delta = int(num_nodes / (fps * (tf - t0)))
frames = prob.time_vector(solution)[::delta]
animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)

# %%
# A frame from the animation.
# sphinx_gallery_thumbnail_number = 4

plt.show()
