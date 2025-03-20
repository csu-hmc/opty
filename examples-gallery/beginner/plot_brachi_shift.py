r"""
Brachistochrone
===============

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
On part of the path, there is a line that the particle must not cross.

.. _Brachistochrone: https://en.wikipedia.org/wiki/Brachistochrone_curve

Explanation of the Approach Taken
---------------------------------

Let the curve be called :math:`b(x)`. If the particle is at :math:`(x(t) ,
b(x(t))` then it cannot have a speed component normal to the tangent of the
curve at this point. This fact is used to set up the equations of motion.
Without loss of generality the starting point is (0, 0).
The interfering line is a holonomic inequality constraint. The inequality is
enforced with an additional state variable, which is then bound to be negative.

Notes
-----

The curve is parameterized by time and a nonholonomic constraint is introduced
to ensure the motion is always tangent to the curve. The nonholonomic
constraint is similar to the constraint present in the `Chaplygin Sleigh`_.

.. _Chaplygin Sleigh: https://en.wikipedia.org/wiki/Chaplygin_sleigh

**States**

- :math:`x`: x-coordinate of the particle
- :math:`y`: y-coordinate of the particle, opposite in sign to the
  gravitational force
- :math:`u_x`: x-component of the velocity of the particle
- :math:`u_y`: y-component of the velocity of the particle
- math:`aux`: auxiliary variable to enforce the holonomic inequality constraint

**Inputs**

- :math:`\beta`: angle of the tangent of the curve with the horizontal

**Known Parameters**

- :math:`m`: mass of the particle
- :math:`g`: acceleration due to gravity
- :math:`b_1`: x coordinate of the final point
- :math:`b_2`: y coordinate of the final point
- :math:`h_{\textrm{shift}}`: shift of the curve in the -y direction
- :math:`\text{steepness}`: slope of the interfering line

**Free Parameters**

- :math:`h`: time interval

"""
import os
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
# The equations of motion represent a particle :math:`P` of mass :math:`m`
# moving in a uniform gravitational field. The point moves on a surface that
# applies a normal force to it and the angle of the surface relative to the
# horizontal is :math:`\beta(t)`. A nonholonomic constraint to ensure the
# particle's velocity only has a component tangent to the surface is added to
# the equations of motion. In addition, a holonomic inequality is added,
# making it a set of differential algebraic equations.
N, A = me.ReferenceFrame('N'), me.ReferenceFrame('A')
O, P = sm.symbols('O P', cls=me.Point)
O.set_vel(N, 0)
t = me.dynamicsymbols._t

m, g, h_shift, steepness = sm.symbols('m, g, h_shift, steepness')
x, y, ux, uy = me.dynamicsymbols('x y u_x u_y')
beta, aux = me.dynamicsymbols('beta, aux')

state_symbols = (x, y, ux, uy, aux)

A.orient_axis(N, beta, N.z)

P.set_pos(O, x*N.x + y*N.y)
P.set_vel(N, ux*N.x + uy*N.y)

speed_constr = sm.Matrix([P.vel(N).dot(A.y)])

kd = sm.Matrix([ux - x.diff(t), uy - y.diff(t)])

bodies = [me.Particle('body', P, m)]
forces = [(P, -m*g*N.y)]

kane = me.KanesMethod(
    N,
    q_ind=[x, y],
    u_ind=[ux],
    u_dependent=[uy],
    kd_eqs=kd,
    velocity_constraints=speed_constr,
)
fr, frstar = kane.kanes_equations(bodies, loads=forces)

eom = kd.col_join(fr + frstar)
eom = eom.col_join(speed_constr)
eom = eom.col_join(sm.Matrix([aux - (-y-steepness*x-h_shift)]))
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
# Set up the constraints and bounds, objective function and its gradient for
# the problem. (b1, b2) is the final point and (0, 0) is the starting point is
# (0/0).
b1 = 10.0
b2 = -6.0

par_map = {
        m: 1.0, g: 9.81,
        h_shift: 0.75,
        steepness: 0.9,
}


instance_constraint = (
    x.func(t0),
    y.func(t0),
    ux.func(t0),
    uy.func(t0),
    x.func(tf) - b1,
    y.func(tf) - b2,
)

bounds = {
    h: (0.0, 1.0),
    beta: (-np.pi/2 + 1.e-5, 0.0),
    aux: (-100.0, 0.0),
}

# %%
# backend='numpy' is used to speed up the execution if a solution is available.
# If no solution is available, backend='cython' is recommended, as it takes a
# large number of iterations to reach the optimal solution.
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
    backend='numpy',
)

# %%
# Solve the problem. Use the given solution if available, else pick a
# reasonable initial guess and solve the problem. This problem takes a large
# number of iterations to reach the optimal solution.
fname = f'brachi_shift_{num_nodes}_nodes_solution.csv'
if os.path.exists(fname):
    solution = np.loadtxt(fname)
else:
    initial_guess = np.random.randn(prob.num_free)*0.1
    prob.add_option('max_iter', 55000)
    solution, info = prob.solve(initial_guess)
    print(info['status_msg'])
    _ = prob.plot_objective_value()

#np.savetxt(fname, solution, fmt='%.12f')

# %%
# State and input trajectories:
_ = prob.plot_trajectories(solution)

# %%
# Constraint violations:
_ = prob.plot_constraint_violations(solution)

# %%
# Animate the Solution
# --------------------
fps = 35


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
t_arr = np.linspace(t0, num_nodes*solution[-1], num_nodes)
state_sol = CubicSpline(t_arr, state_vals.T)
input_sol = CubicSpline(t_arr, input_vals.T)

xmin = -1.0
xmax = b1 + 1.0
ymin = b2 - 1.0
ymax = 1.0

arrow_head = me.Point('arrow_head')
arrow_head.set_pos(P, ux*N.x + uy*N.y)

coordinates = P.pos_from(O).to_matrix(N)
coordinates = coordinates.row_join(arrow_head.pos_from(O).to_matrix(N))

coords_lam = sm.lambdify(list(state_symbols) + [beta] + list(par_map.keys()),
                         coordinates, cse=True)

# %%
def init_plot():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)

    # Plot the Brachistochrone analytical solution.
    ax.plot(xx, yy, color='red', lw=0.5)
    ax.scatter([0], [0], color='red', s=10)
    ax.scatter([b1], [b2], color='green', s=10)


    # plot the restrictive line
    start_x = -0.5
    ax.plot([start_x, -(b2 + par_map[h_shift])/par_map[steepness]],
            [-par_map[steepness]*start_x-par_map[h_shift], b2],
            color='black', lw=0.5)

    # shade the 'forbidden triangle
    triangle_x = [start_x, -(b2 + par_map[h_shift])/par_map[steepness],
                 start_x]
    triangle_y = [-par_map[steepness]*start_x-par_map[h_shift], b2, b2]
    ax.fill(triangle_x, triangle_y, color='gray', alpha=0.5)
    ax.plot(triangle_x, triangle_y, color='black', lw=0.5)

    ax.annotate('Brachistochrone', xy=(1.8, -3),
                arrowprops=dict(arrowstyle='->',
                        connectionstyle='arc3, rad=-.2', color='green',
                        lw=0.25),
                xytext=(4.0, -2.0), fontsize=10)

    # Set up the point and the speed arrow.
    punkt = ax.scatter([], [], color='red', s=50)
    # Draw the speed vector.
    pfeil = ax.quiver([], [], [], [], color='green', scale=25, width=0.004)

    return fig, ax, punkt, pfeil
# %%
# Function to update the plot for each animation frame
fig, ax, punkt, pfeil = init_plot()


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
animation = FuncAnimation(
    fig,
    update,
    frames=np.arange(t0, (num_nodes + 2)*h_val, 1/fps),
    interval=3000/fps,
)


# A frame from the animation.
# sphinx_gallery_thumbnail_number = 4
plt.show()
