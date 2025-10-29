# %%

r"""
Light Diffraction
=================

Objectives
----------

- Show how to use inequalities on the equations of motion.
- Show how to use differentiable functions to approximate their non-
  differentiable counterparts.
- Show that values of the parameter map may be changed in a loop without having
  to build ``Problem`` again.

Introduction
------------

A particle should move from the origin to the final point in minimum time.
The maximum speed is limited to different values in different regions.
This is accomplished by using a smooth approximation of a 'hump' function in
connection with inequality constraints on the equations of motion.

Notes
-----

- The bending of light on surfaces is explained by the assumption that its
  speed is different in different media, while it 'tries' to minimize the time
  it takes to travel. More details may be found here.

  https://en.wikipedia.org/wiki/Refractive_index
  So, this may be considered as a simulation of light diffraction.

- Here differentiable hump functions are used instead of their non-
  differentiable counterparts. This may create unwanted 'edge effects' in the
  solution of the problem. Here :math:`u_x > 0.0` is enforced to avoid vertical
  motion.


**states**

- :math:`x, y` : position of the particle
- :math:`u_x, u_y` : velocity of the particle


**controls**

- :math:`f_x, f_y` : force applied to the particle

**parameters**

- :math:`m` : mass of the particle [kg]
- :math:`a_1, a_2, a_3` : boundaries of the regions [m]
- :math:`\mu_1, \mu_2, \mu_3` : maximum speed in the regions [m/s]
- :math:`\textrm{steepness}` : steepness of the 'hump' function

"""

import os
from opty.utils import MathJaxRepr
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty import Problem
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %%
# Define the smooth hump function.


def smooth_hump(x, a, b, k):
    """gives approx 1.0 for a < x < b, 0.0 otherwise.
    The larger k, the steeper the transition"""
    return 0.5 * (sm.tanh(k * (x - a)) - sm.tanh(k * (x - b)))


# %%
# Set Up the Equations of Motion, Kane's Method
# ---------------------------------------------
N = me.ReferenceFrame('N')
O, P = sm.symbols('O P', cls=me.Point)
P.set_vel(N, 0)
t = me.dynamicsymbols._t

x, y, ux, uy = me.dynamicsymbols('x y u_x u_y')
fx, fy = me.dynamicsymbols('f_x f_y')

m, a1, a2, a3 = sm.symbols('m a_1 a_2 a_3')
mu1, mu2, mu3 = sm.symbols('mu_1 mu_2 mu_3')
steepness = sm.symbols('steepness')
friction = sm.symbols('friction')

P.set_pos(O, x * N.x + y * N.y)
P.set_vel(N, ux * N.x + uy * N.y)
bodies = [me.Particle('P', P, m)]

forces = [(P, fx * N.x + fy * N.y - ux * friction * N.x - uy * friction * N.y)]

kd = sm.Matrix([ux - x.diff(t), uy - y.diff(t)])

KM = me.KanesMethod(N, q_ind=[x, y], u_ind=[ux, uy], kd_eqs=kd)
fr, frstar = KM.kanes_equations(bodies, forces)
eom = kd.col_join(fr + frstar)
# %%
# Add the eoms to bound the speeds in the different regions.
hump1 = smooth_hump(x, -1.0, a1, steepness)
hump2 = smooth_hump(x, a1, a2, steepness)
hump3 = smooth_hump(x, a2, a3, steepness)
speed_mag = sm.sqrt(ux**2 + uy**2)

eom1 = sm.Matrix([speed_mag * hump1,
                  speed_mag * hump2,
                  speed_mag * hump3])
eom = eom.col_join(eom1)

# %%
# Bound the magnitude of the driving force, but not its direction.
eom = eom.col_join(sm.Matrix([sm.sqrt(fx**2 + fy**2)]))
print((f'The equations of motion have {sm.count_ops(eom)} operations and have'
      f' shape {eom.shape}'))
MathJaxRepr(eom)
# %%
# Set Up the Optimization Problem
# -------------------------------
h = sm.symbols('h')
interval = h
num_nodes = 501
t0, tf = 0.0, h * (num_nodes - 1)

state_symbols = [x, y, ux, uy]

par_map = {
    m: 1.0,
    a1: 3.0,
    a2: 7.0,
    a3: 11.0,
    mu1: 5.0,
    mu2: 1.0,
    mu3: 5.0,
    steepness: 80.0,
    friction: 1.0,
}

# %%
# Plot the hump function.
a, b, xt, k = sm.symbols('a b xt k')
eval_hump = sm.lambdify((xt, a, b, k), smooth_hump(xt, a, b, k))
x_vals = np.linspace(-1, 1, 100)
k = par_map[steepness]
fig, ax = plt.subplots(figsize=(8, 1.5), layout='constrained')
ax.plot(x_vals, eval_hump(x_vals, -0.5, 0.75, k))
ax.axvline(-0.5, color='black', lw=0.5)
ax.axvline(0.75, color='black', lw=0.5)
ax.axhline(0.0, color='black', lw=0.5)
ax.axhline(1.0, color='black', lw=0.5)
_ = ax.set_title(f"Smooth hump function with steepness = {k}")

# %%
# Build ``Problem``.


def obj(free):
    """minimize the variable time interval."""
    return free[-1]


def obj_grad(free):
    """Gradient of the objective function."""
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad


instance_constraints = (
    x.func(t0) - 0.0,
    y.func(t0) - 0.0,
    x.func(tf) - 10.0,
    y.func(tf) - 10.0,
)

bounds = {
    h: (0.0, 1.0),
    x: (0.0, 10.0),
    y: (0.0, 10.0),
    ux: (0.5, np.inf),
    uy: (0.1, np.inf),
}

limit = 400
eom_bounds = {
    4: (0.0, par_map[mu1]),
    5: (0.0, par_map[mu2]),
    6: (0.0, par_map[mu3]),
    7: (0.0, limit),
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
        eom_bounds=eom_bounds,
        time_symbol=t,
        backend='numpy',
)

prob.add_option('max_iter', 15000)

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

    it0 = 0
    for i in range(15):
        # Values of par_map may be changed in a loop without having to build
        # Problem again.
        par_map[steepness] = 10 + 5 * i

        solution, info = prob.solve(initial_guess)
        initial_guess = solution
        print(info['status_msg'])
        it1 = len(prob.obj_value)
        print('Iterations needed', it1 - it0)
        it0 = it1
    _ = prob.plot_objective_value()
    print(f'Total iterations needed were: {len(prob.obj_value)}')

# %%
# Plot trajectories.
_ = prob.plot_trajectories(solution, show_bounds=True)

# %%
# PLot the constraint violations.
_ = prob.plot_constraint_violations(solution)

# %%
# Animate the Motion
# ------------------
fps = 15

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
    pfeil = ax.quiver([], [], [], [], color='green', scale=25,
                      width=0.004, headwidth=8)

    return fig, ax, line1, line2, pfeil


fig, ax, line1, line2, pfeil = init()


def update(t):
    message = (f'running time {t:.2f} sec \n Darker shade is slower maximum'
               f' speed \n The speed is the green arrow')
    ax.set_title(message, fontsize=12)

    coords = coords_lam(*state_sol(t), *input_sol(t), *pL_vals)

    koords_x = []
    koords_y = []
    for t1 in np.linspace(t0, tf, int(fps * (tf - t0))):
        if t1 <= t:
            coords = coords_lam(*state_sol(t1), *input_sol(t1), *pL_vals)
            koords_x.append(coords[0, 0])
            koords_y.append(coords[1, 0])
    line2.set_data(koords_x, koords_y)
    line1.set_offsets([coords[0, 0], coords[1, 0]])
    pfeil.set_offsets([coords[0, 0], coords[1, 0]])
    pfeil.set_UVC(coords[0, 1] - coords[0, 0], coords[1, 1] - coords[1, 0])


frames = np.linspace(t0, tf, int(fps * (tf - t0)))
animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
# %%
# A frame from the animation.
# sphinx_gallery_thumbnail_number = 4
fig, ax, line1, line2, pfeil = init()
update(4.15)

plt.show()
