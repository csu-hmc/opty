# %%
r"""
Classic Brachistochrone
=======================

Objective
---------

- Show how to solve a variational problem using opty with variable time
  interval.

Introduction
------------

A particle is to slide without friction under the constant force of gravity on
a curve from A to B in the shortest time. B must not be directly under A (else
the problem is trivial). The solution curve is called the Brachistochrone_
which is a cycloid.

.. _Brachistochrone: https://en.wikipedia.org/wiki/Brachistochrone_curve

Explanation of the Approach Taken
---------------------------------

Let the curve to be determined be called :math:`b(x)`.
If the particle is at :math:`(x(t), b(x(t))` then its speed :math:`\bar{v(t)}`
must be tangent to the curve. If :math:`\beta(t)` is the angle of the tangent
of the curve with the horizontal, then the velocity components must satisfy
:math:`\dot{x} = v(t) \cdot \cos(\beta),
\quad \dot{y} = v(t) \cdot \sin(\beta)`. The acceleration due to gravity lies
along the tangent as well: :math:`\dot{v(t)} = g \cdot \sin(\beta)`
Without loss of generality the starting point is (0, 0).

Notes
-----

- Obviously, the mass of the particle does not matter.
- It is well known that the motion will start vertically downwards.
  Hence setting up the equations of motion the 'standard way' will lead to a
  singularity at the initial time, when the optimization is performed by opty.
  The equations used here avoid this singularity.

**States**

- :math:`x`: x-coordinate of the particle
- :math:`y`: y-coordinate of the particle, opposite in sign to the
  gravitational force
- :math:`v`: the speed

**Inputs**

- :math:`\beta`: angle of the tangent of the curve with the horizontal

**Known Parameters**

- :math:`g`: acceleration due to gravity
- :math:`b_1`: x coordinate of the final point
- :math:`b_2`: y coordinate of the final point

**Free Parameters**

- :math:`h`: time interval

"""
from opty.utils import MathJaxRepr
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from opty import Problem
from scipy.optimize import root
from scipy.interpolate import CubicSpline
from matplotlib.animation import FuncAnimation

# %%
# Equations of Motion
# -------------------
#
t = me.dynamicsymbols._t

g = sm.symbols('g')
x, y, v = me.dynamicsymbols('x y v')
beta = me.dynamicsymbols('beta')

state_symbols = (x, y, v)


eom = sm.Matrix([
    x.diff(t) - v * sm.cos(beta),
    y.diff(t) - v * sm.sin(-beta),
    v.diff(t) - g * sm.sin(beta)
])
MathJaxRepr(eom)

# %%
# Set up the Optimization Problem and Solve It
# --------------------------------------------
#
# The goal is to minimize the time to reach the second point. So make the time
# interval a variable :math:`h`.
h = sm.symbols('h')
num_nodes = 201
t0, tf = 0*h, h*(num_nodes - 1)


def obj(free):
    return free[-1]


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad


# %%
# (b1, b2) is the final point and (0, 0) is the starting point.
b1 = 10.0
b2 = -6.0

# %%
# Initial and final constraints for the states.
instance_constraint = (
    x.func(t0),
    y.func(t0),
    v.func(t0),
    x.func(tf) - b1,
    y.func(tf) - b2,
)

# %%
# Bounding h to be positive avoids non-physical solutions.
bounds = {
    h: (0.0, 1.0),
    beta: (-np.pi/2, np.pi),
}

# %%
# Only one known parameter, the gravity.
par_map = {g: 9.81}


# %%
# Set up an instance of Problem.

prob = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    num_nodes,
    h,
    time_symbol=t,
    known_parameter_map=par_map,
    instance_constraints=instance_constraint,
    bounds=bounds,
    backend='numpy'
)

# %%
# Pick an reasonable initial guess.
x_guess = np.linspace(0, b1, num_nodes)
y_guess = np.linspace(0, b2, num_nodes)
v_guess = np.linspace(0, np.sqrt(-2*par_map[g]*b2), num_nodes)
beta_guess = beta_guess = np.linspace(np.pi/2, 0.0, num_nodes)
initial_guess = np.concatenate([x_guess, y_guess, v_guess, beta_guess, [0.01]])

# %%
# Solve the problem.
solution, info = prob.solve(initial_guess)
print(info['status_msg'])

# %%
# Plot the objective value over iterations.
_ = prob.plot_objective_value()

# %%
# State and input trajectories:
_ = prob.plot_trajectories(solution)

# %%
# Constraint violations:
_ = prob.plot_constraint_violations(solution)

# %%
# Animate the Solution
# --------------------
fps = 20


# %%
# Calculate the Brachistochrone from (0, 0) to (b1, b2). It is very sensitive
# to the initial guess.


def func(x0):
    """Gives the Brachistochrome equation for the starting point at (0, 0) and
    the final point at (b1, b2)"""
    r, theta = x0[0], x0[1]
    return [r*theta - r*np.sin(theta) - b1, r*(1.0 - np.cos(theta)) - b2]


x0 = [1.0, 1.0]
resultat = root(func, x0)

times = np.linspace(0.0, resultat.x[1], num_nodes)
xx = resultat.x[0]*times - resultat.x[0]*np.sin(times)
yy = resultat.x[0]*(1.0 - np.cos(times))

# %%
# Set up the plot.
state_vals, input_vals, _, h_val = prob.parse_free(solution)
t_arr = prob.time_vector(solution=solution)
state_sol = CubicSpline(t_arr, state_vals.T)
input_sol = CubicSpline(t_arr, input_vals.T)

xmin = -1.0
xmax = b1 + 1.0
ymin = b2 - 1.0
ymax = 1.0

N = me.ReferenceFrame('N')
O, arrow_head, P = sm.symbols('O arrow_head P', cls=me.Point)
P.set_pos(O, x*N.x + y*N.y)
arrow_head.set_pos(P, v*sm.cos(beta)*N.x + v*sm.sin(-beta)*N.y)

coordinates = P.pos_from(O).to_matrix(N)
coordinates = coordinates.row_join(arrow_head.pos_from(O).to_matrix(N))

coords_lam = sm.lambdify(list(state_symbols) + [beta] + list(par_map.keys()),
                         coordinates, cse=True)


def init_plot():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)

    # Plot the Brachistochrone analytical solution.
    ax.plot(xx, yy, color='black', lw=0.5)
    ax.scatter([0], [0], color='red', s=10)
    ax.scatter([b1], [b2], color='green', s=10)

    # Set up the point and the speed arrow.
    punkt = ax.scatter([], [], color='red', s=50)
    # Draw the speed vector.
    pfeil = ax.quiver([], [], [], [], color='green', scale=25, width=0.004)

    return fig, ax, punkt, pfeil


# %%
# Function to update the plot for each animation frame.


def update(t):
    message = (f'Running time {t:.2f} sec \n'
               f'The green arrow is the speed \n'
               f'Slow motion video.')
    ax.set_title(message, fontsize=12)

    coords = coords_lam(*state_sol(t), input_sol(t), *par_map.values())
    punkt.set_offsets([coords[0, 0], coords[1, 0]])

    pfeil.set_offsets([coords[0, 0], coords[1, 0]])
    pfeil.set_UVC(coords[0, 1] - coords[0, 0], coords[1, 1] - coords[1, 0])


# %%
# Create the animation.
fig, ax, punkt, pfeil = init_plot()

# sphinx_gallery_thumbnail_number = 4

ani = FuncAnimation(
    fig,
    update,
    frames=np.arange(t0, (num_nodes + 1)*h_val, 1/fps),
    interval=3000/fps,
)

plt.show()
