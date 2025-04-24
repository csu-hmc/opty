# %%

r"""

Particle Through Gates
======================

Objective
---------

- Show how the current inability of opty to allow instance constraints of the
  form :math:`x(t_i) = (a, b)` may be overcome.

Introduction
------------

A particle is moving in the horizontal X/Y plane. It is driven by a force
:math:`F = \begin{pmatrix} f_x \\ f_y \end{pmatrix}` and must pass through
three gates. Anywhere insinde each gate is allowed, and opty should find the
best path.

To covercome opty's current inability one may do as follows:

- introduce a differentiable hump function :math:`H(x, a, b,
  \textrm{steepness})` that is 1 for :math:`x \in (a, b)` else 0.
  :math:`\textrm{steepness}` controls the slopes of the 'walls'.
- Introduce state variables :math:`\textrm{gate}_i`
- add :math:`\textrm{gate}_i - H(x, a_i, b_i, \textrm{steepness}) \cdot
  \textrm{aux}_i` to the equations of motion.
- :math:`\textrm{aux}_i \in (0.6, 1.0)` are free input parameters.
- Set :math:`\textrm{gate}_i = 0.95` in the instance constraints.


Notes
-----

- :math:`\textrm{steepness} \in (5.0, 15.0)` is a good range. If it is too
  large, convergence seems more difficult.
- :math:`\textrm{aux}_i` help convergence. Not totally clear, why this is so.


**States**

- :math:`x, y` are the coordinates of the particle.
- :math:`u_x, u_y` are the velocities of the particle.
- :math:`\textrm{gate}_i` auxiliary variables described above.


**Known Parameters**

- :math:`m` is the mass of the particle [kg].
- :math:`\nu` is the friction coefficient [kg/s].
- :math:`a_i, b_i` are the coordinates of the gates [m].
- :math:`\textrm{steepness}` is the steepness of the gates [1/m].


**Free Parameters**

- :math:`h` is the time step [s].
- :math:`\textrm{aux}_i` are the auxiliary parameters described above.


**Specifieds**

- :math:`f_x, f_y` are the forces acting on the particle [N].

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

# %%
# Define the differentiable hump function and plot it.


def hump_diff(var, a, b, steepness):
    """returns a differentiable function that is 1 between a and b"""
    return 0.5 * (sm.tanh(steepness * (var - a)) + sm.tanh(steepness *
                                                           (b - var)))


a, b, var, c = sm.symbols('a b var c')
hump_lam = sm.lambdify((var, a, b, c), hump_diff(var, a, b, c), cse=True)

# ``steepness``  wil be set to ``c`` below.
c = 5.0
XX = np.linspace(-5.0, 5.0, 500)
YY_hump = hump_lam(XX, -1.0, 1.0, c)
fig, ax = plt.subplots(figsize=(6.8, 1.5), layout='constrained')
ax.plot(XX, YY_hump)
ax.set_title(f'Differentiable Heaviside function, steepness ={c}')
ax.axvline(x=-1.0, color='black', linestyle='--', lw=0.5)
ax.axvline(x=1.0, color='black', linestyle='--', lw=0.5)

for i in (0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.4):
    print(f'value of hump at x = {i} is {hump_lam(i, -1.0, 1.0, c):.5f}')

# %%
# Set Up the Equations of Motion
# -----------------------------
N = me.ReferenceFrame('N')
O, P = sm.symbols('O P', cls=me.Point)

O.set_vel(N, 0)
t = me.dynamicsymbols._t

x, y, ux, uy = me.dynamicsymbols('x y ux uy')
gate_1, gate_2, gate_3 = me.dynamicsymbols('gate_1 gate_2 gate_3')
fx, fy = me.dynamicsymbols('fx fy')

m, nu = sm.symbols('m nu')
steepness = sm.symbols('steepness')
a1l, a1r, a2b, a2t, a3l, a3r = sm.symbols('a1l a1r a2b a2t a3l a3r')
gate_1y, gate_2x, gate_3y = sm.symbols('gate_1y gate_2x gate_3y')
aux1, aux2, aux3 = sm.symbols('aux1, aux2, aux3')

P.set_pos(O, x * N.x + y * N.y)
P.set_vel(N, ux * N.x + uy * N.y)

body = me.Particle('body', P, m)
bodies = [body]
forces = [(P, fx * N.x + fy * N.y - nu * P.vel(N))]
kd = sm.Matrix([ux - x.diff(t), uy - y.diff(t)])

kanes = me.KanesMethod(N, q_ind=[x, y], u_ind=[ux, uy], kd_eqs=kd)
fr, frstar = kanes.kanes_equations(bodies, forces)
eom = kd.col_join(fr + frstar)

# %%
# Add the gate conditions to the equations of motion and print them.
eom_gates = sm.Matrix([
    gate_1 - hump_diff(x, a1l, a1r, steepness) * aux1,
    gate_2 - hump_diff(y, a2b, a2t, steepness) * aux2,
    gate_3 - hump_diff(x, a3l, a3r, steepness) * aux3,
])

eom = eom.col_join(eom_gates)
MathJaxRepr(eom)

# %%
# Define the Problem and Solve It
# -------------------------------
h = sm.symbols('h')
num_nodes = 501
state_symbols = (x, y, ux, uy, gate_1, gate_2, gate_3)
t0, t1, t2, t3, tf = (0.0, int(num_nodes/4) * h, int(num_nodes/2) * h,
                      int(3*num_nodes/4) * h, (num_nodes-1) * h)
interval_value = h

par_map = {}
par_map[steepness] = c
par_map[m] = 1.0
par_map[nu] = 0.0
par_map[a1l] = 0.0
par_map[a1r] = 2.0
par_map[a2b] = 6.0
par_map[a2t] = 8.0
par_map[a3l] = 7.0
par_map[a3r] = 9.0
par_map[gate_1y] = 2.0
par_map[gate_2x] = 5.0
par_map[gate_3y] = 1.0

instance_constraints = (
    x.func(t0) - 2.0,
    y.func(t0) - 0.0,
    ux.func(t0) - 0.0,
    uy.func(t0) - 0.0,

    gate_1.func(t1) - 0.95,
    y.func(t1) - gate_1y,
    gate_2.func(t2) - 0.95,
    x.func(t2) - gate_2x,
    gate_3.func(t3) - 0.95,
    y.func(t3) - gate_3y,
    # At the final time particle to be at rest at its starting position.
    x.func(tf) - 2.0,
    y.func(tf) - 0.0,
    ux.func(tf) - 0.0,
    uy.func(tf) - 0.0,
)

limit = 10
bounds = {
    h: (0.0, 0.5),
    fx: (-limit, limit),
    fy: (-limit, limit),
    x: (0.0, 15.0),
    y: (0.0, 15.0),
    aux1: (0.6, 1.0),
    aux2: (0.6, 1.0),
    aux3: (0.6, 1.0),
    }


def obj(free):
    return free[-1]


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad


prob = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    num_nodes,
    interval_value,
    known_parameter_map=par_map,
    instance_constraints=instance_constraints,
    time_symbol=t,
    bounds=bounds,
    backend='numpy',
)

prob.add_option('max_iter', 5000)
# %%
# Create a good initial guess.

initial_guess = np.ones(prob.num_free) * 0.0
z1 = int(num_nodes/4)

x1 = np.linspace(2.0, (par_map[a1r] + par_map[a1l])/2, z1)
y1 = np.linspace(0.0, par_map[gate_1y], z1)

x2 = np.linspace((par_map[a1r] + par_map[a1l])/2, par_map[gate_2x], z1)
y2 = np.linspace(par_map[gate_1y], (par_map[a2t] + par_map[a2b])/2, z1)

x3 = np.linspace(par_map[gate_2x], (par_map[a3r] + par_map[a3l])/2, z1)
y3 = np.linspace((par_map[a2t] + par_map[a2b])/2, par_map[gate_3y], z1)

x4 = np.linspace((par_map[a3r] + par_map[a3l])/2, 2.0, z1)
y4 = np.linspace(par_map[gate_3y], 0.0, z1)

x_total = np.concatenate((x1, x2, x3, x4))
y_total = np.concatenate((y1, y2, y3, y4))

initial_guess[0: 8*z1] = np.concatenate((x_total, y_total))

# %%
# Use the existing solution if available, else solve the problem.
# The solution, if it exists is saved in ``fname``.
fname = f'particle_through_gates_{num_nodes}_nodes_solution.csv'

if os.path.exists(fname):
    # Solution exists.
    solution = np.loadtxt(fname)
else:
    # Calculate the solution.
    for _ in range(2):
        solution, info = prob.solve(initial_guess)
        initial_guess = solution
        print(info['status_msg'])
        _ = prob.plot_objective_value()

# %%
# Plot the trajectories.
_ = prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
_ = prob.plot_constraint_violations(solution, subplots=True)

# %%
# Print the values of the auxiliary variables.
print(f'value of aux1 is {solution[-4]:.3f}')
print(f'value of aux2 is {solution[-3]:.3f}')
print(f'value of aux3 is {solution[-2]:.3f}')

# np.savetxt(fname, solution, fmt='%.12f')
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
    xmin, xmax = -1.0, 10.0
    ymin, ymax = -1.0, 10.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('X-axis [m]')
    ax.set_ylabel('Y-axis [m]')

    ax.scatter(par_map[a1l], par_map[gate_1y], color='red', s=25)
    ax.scatter(par_map[a1r], par_map[gate_1y], color='red', s=25)
    ax.plot([par_map[a1l], par_map[a1r]], [par_map[gate_1y], par_map[gate_1y]],
            color='red', lw=0.5)

    ax.scatter(par_map[gate_2x], par_map[a2b], color='blue', s=25)
    ax.scatter(par_map[gate_2x], par_map[a2t], color='blue', s=25)
    ax.plot([par_map[gate_2x], par_map[gate_2x]], [par_map[a2b], par_map[a2t]],
            color='blue', lw=0.5)

    ax.scatter(par_map[a3l], par_map[gate_3y], color='green', s=25)
    ax.scatter(par_map[a3r], par_map[gate_3y], color='green', s=25)
    ax.plot([par_map[a3l], par_map[a3r]], [par_map[gate_3y], par_map[gate_3y]],
            color='green', lw=0.5)

    ax.plot(x_total, y_total, color='black', lw=0.25, linestyle='--')

    line, = ax.plot([], [], color='black', lw=0.5, linestyle='--')
    point = ax.scatter([], [], color='black', s=100)
    pfeil = ax.quiver([], [], [], [], color='green', scale=55, width=0.002,
                      headwidth=8)

    return fig, ax, point, line, pfeil


fig, ax, point, line, pfeil = init_plot()
plt.close(fig)


def update(t):
    message = (f'running time {t:0.2f} sec \n The gree arrow shows the force.'
               f'\n The light grey line is the initial guess.')
    ax.set_title(message, fontsize=12)

    coords = coords_lam(*state_sol(t), *input_sol(t), *pl_vals)

    point.set_offsets([coords[0, 0], coords[1, 0]])
    add_point_to_data(line, coords[0, 0], coords[1, 0])
    pfeil.set_offsets([coords[0, 0], coords[1, 0]])
    pfeil.set_UVC(input_sol(t)[0], input_sol(t)[1])
    return point, pfeil


# %%
# Create the animation.
# sphinx_gallery_thumbnail_number = 4
fig, ax, point, line, pfeil = init_plot()
anim = FuncAnimation(fig, update,
                     frames=np.arange(t0, tf, 1/fps),
                     interval=1/fps*1000)

plt.show()
