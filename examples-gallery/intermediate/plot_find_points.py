# %%
r"""
Hit Points at Open Times
========================

Objective
---------

- Show how to use the objective function to hit given points at times
  determined by ``opty``.

Description
-----------

A particle of mass :math:`m` is moved by a force :math:`f_x, f_y`. It must
move from the origin to a final point as fast as possibe and must hit given
points along the way.
The particle moves in the horizontal X/Y plane so gravity is not considered.

Method
------

The objective function is defined that at each point in time :math:`t_0 = 0,
t_1 = h, t_2 = 2h, .., t_f = (\textrm{num_nodes} - 1) * h` the distance from
the x - coordinate of the particle to a given point is considered. Say, the
minimal distance happens at :math:`t_m` Then only
:math:`(x_{\textrm{particle}}(t_m) -x_{\textrm{point}})^2 +
(y_{\textrm{particle}}(t_m) - y_{\textrm{point}})^2`
will be returned. This way for each point to be hit.

The gradient is formed accordingly.

The value of the variable time interval :math:`h` is added to ensure that the
speed is maximized.

Notes
-----

- This method crucially utilizes the fact, that ``opty`` looks at the whole
  trajectory simultaneously when trying to find the optimal solution.
- The way the objective function is defined, this entails a trade off between
  speed and accuracy of hitting the points.

**States**:

- :math:`x, y` - position of the particle in the X/Y plane
- :math:`u_x, u_y` - velocity of the particle in the X/Y plane

**Inputs**:

- :math:`f_x, f_y` - force acting on the particle in the X/Y plane

**Parameters**:

- :math:`m` - mass of the particle

"""
import os
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty import Problem
from opty.utils import MathJaxRepr

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# sphinx_gallery_thumbnail_number = 4


# %%
# Equations of Motion.
N = me.ReferenceFrame('N')
O, P = me.Point('O'), me.Point('P')
t = me.dynamicsymbols._t
O.set_vel(N, 0)

x, y, ux, uy = me.dynamicsymbols('x, y, u_x, u_y')
fx, fy = me.dynamicsymbols('f_x, f_y')

m = sm.symbols('m')

P.set_pos(O, x * N.x + y * N.y)
P.set_vel(N, ux * N.x + uy * N.y)

Pa = me.Particle('P', P, m)
bodies = [Pa]
forces = [(P, fx * N.x + fy * N.y)]

kd = sm.Matrix([ux - x.diff(t), uy - y.diff(t)])

KM = me.KanesMethod(N, q_ind=[x, y], u_ind=[ux, uy], kd_eqs=kd)
fr, frstar = KM.kanes_equations(bodies, forces)

eom = kd.col_join(fr + frstar)
MathJaxRepr(eom)

# %%
# Set Up the Optimization Problem and Solve it
# --------------------------------------------

h = sm.symbols('h')
num_nodes = 100

t0, tf = 0 * h, (num_nodes - 1) * h
interval_value = h

state_symbols = sm.Matrix([x, y, ux, uy])

# %%
# Set up the objective function and its gradient as explained above.

# Coordinates of points to be reached on the journey
x1, y1, x2, y2, x3, y3, x4, y4 = 1.0, 2.0, 4.0, 8.0, 8.0, 4.0, 5.0, 4.0

# Relative importance between speed and accuracy of hitting the points.
weight = 10.0


def obj(free):
    X1 = (free[0: num_nodes] - x1)**2
    X2 = (free[0: num_nodes] - x2)**2
    X3 = (free[0: num_nodes] - x3)**2
    X4 = (free[0: num_nodes] - x4)**2

    Y1 = (free[num_nodes: 2 * num_nodes] - y1)**2
    Y2 = (free[num_nodes: 2 * num_nodes] - y2)**2
    Y3 = (free[num_nodes: 2 * num_nodes] - y3)**2
    Y4 = (free[num_nodes: 2 * num_nodes] - y4)**2

    minx1 = np.argmin(X1)
    minx2 = np.argmin(X2)
    minx3 = np.argmin(X3)
    minx4 = np.argmin(X4)
    return (X1[minx1] + X2[minx2] + X3[minx3] + X4[minx4] + Y1[minx1] +
            Y2[minx2] + Y3[minx3] + Y4[minx4] + free[-1] * weight)


def obj_grad(free):
    X1 = (free[0: num_nodes] - x1)**2
    X2 = (free[0: num_nodes] - x2)**2
    X3 = (free[0: num_nodes] - x3)**2
    X4 = (free[0: num_nodes] - x4)**2
    minx1 = np.argmin(X1)
    minx2 = np.argmin(X2)
    minx3 = np.argmin(X3)
    minx4 = np.argmin(X4)

    grad = np.zeros_like(free)
    grad[minx1] = 2 * (free[minx1] - x1)
    grad[minx2] = 2 * (free[minx2] - x2)
    grad[minx3] = 2 * (free[minx3] - x3)
    grad[minx4] = 2 * (free[minx4] - x4)
    grad[minx1 + num_nodes] = 2 * (free[minx1 + num_nodes] - y1)
    grad[minx2 + num_nodes] = 2 * (free[minx2 + num_nodes] - y2)
    grad[minx3 + num_nodes] = 2 * (free[minx3 + num_nodes] - y3)
    grad[minx4 + num_nodes] = 2 * (free[minx4 + num_nodes] - y4)
    grad[-1] = weight
    return grad


par_map = {m: 1.0}

instance_constraints = (
    x.func(t0),
    y.func(t0),
    ux.func(t0),
    uy.func(t0),
    x.func(tf) - 10.0,
    y.func(tf),
    ux.func(tf),
    uy.func(tf)
)

limit = 7.5
bounds = {
    h: (0.0, 1.0),
    fx: (-limit, limit),
    fy: (-limit, limit),
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

fname = f'plot_find_points_{num_nodes}_nodes_solution.csv'
if os.path.exists(fname):
    solution = np.loadtxt(fname)
else:
    initial_guess = np.zeros(prob.num_free)
    initial_guess[0: num_nodes] = np.linspace(0.0, 10.0, num_nodes)

    for _ in range(5):
        solution, info = prob.solve(initial_guess)
        initial_guess = solution
        print(info['status_msg'])
    _ = prob.plot_objective_value()
# %%
# Plot the trajectories.
_ = prob.plot_trajectories(solution)

# %%
# Plot the constraint violations
_ = prob.plot_constraint_violations(solution)

# %%
# Animate the Simulation
# ----------------------

fps = 20


def add_point_to_data(line, x, y):
    # to trace the path of the point. Copied from Timo.
    old_x, old_y = line.get_data()
    line.set_data(np.append(old_x, x), np.append(old_y, y))


state_vals, input_vals, _, h_sol = prob.parse_free(solution)

tf = h_sol*(num_nodes - 1)
t_arr = prob.time_vector(solution=solution)
state_sol = CubicSpline(t_arr, state_vals.T)
input_sol = CubicSpline(t_arr, input_vals.T)

coordinates = P.pos_from(O).to_matrix(N)

pl, pl_vals = zip(*par_map.items())
coords_lam = sm.lambdify((*state_symbols, fx, fy, *pl), coordinates, cse=True)

width, height, radius = 0.5, 0.5, 0.5


def init_plot():
    xmin, xmax = -1.0, 11.0
    ymin, ymax = -2.0, 10.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('X-axis [m]')
    ax.set_ylabel('Y-axis [m]')

    ax.scatter(x1, y1, color='blue', s=25)
    ax.scatter(x2, y2, color='blue', s=25)
    ax.scatter(x3, y3, color='blue', s=25)
    ax.scatter(x4, y4, color='blue', s=25)
    ax.scatter(0.0, 0.0, color='red', s=25)
    ax.scatter(10.0, 0.0, color='green', s=25)

    point = ax.scatter([], [], color='black', s=100)
    line, = ax.plot([], [], color='black', lw=0.5, alpha=0.5)
    pfeil = ax.quiver([], [], [], [], color='green', scale=55, width=0.002,
                      headwidth=8)

    return fig, ax, point, line, pfeil


fig, ax, point, line, pfeil = init_plot()
plt.close(fig)


def update(t):
    message = (f'running time {t:0.2f} sec '
               f'\n The green arrow shows the force.')
    ax.set_title(message, fontsize=12)

    coords = coords_lam(*state_sol(t), *input_sol(t), *pl_vals)

    point.set_offsets([coords[0, 0], coords[1, 0]])
    add_point_to_data(line, coords[0, 0], coords[1, 0])
    pfeil.set_offsets([coords[0, 0], coords[1, 0]])
    pfeil.set_UVC(input_sol(t)[0], input_sol(t)[1])
    return point, line, pfeil


# %%
# Create the animation.
fig, ax, point, line, pfeil = init_plot()
anim = FuncAnimation(fig, update,
                     frames=np.arange(t0, tf, 1/fps),
                     interval=1/fps*1000)

plt.show()
