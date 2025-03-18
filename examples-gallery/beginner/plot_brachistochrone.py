r"""
Brachistochrone
===============

Objective
---------

- Show how to solve variational problems using opty with variable time interval.

Introduction
------------

A particle is to slide without friction on a curve from A to B in the shortest
time. B must not be directly under A (else the problem is trivial).
The solution curve is called the **Brachistochrone**. It is a cycloid. More
about this classic problem may be found here:

https://en.wikipedia.org/wiki/Brachistochrone_curve


Explanation of the Approach Taken
---------------------------------

Let the curve be called b(x). If the particle is at (x(t) , b(x(t)) then it
cannot have a speed component normal to the tangent of the curve at this
point. This fact is used to set up the equations of motion.
Without loss of generality to starting point is (0, 0).


**States**

- :math:`x`: x-coordinate of the particle
- :math:`y`: y-coordinate of the particle
- :math:`u_x`: x-component of the velocity of the particle
- :math:`u_y`: y-component of the velocity of the particle

**Controls**

- :math:`\beta`: angle of the tangent of the curve with the horizontal

**Known Parameters**

- :math:`m`: mass of the particle
- :math:`g`: acceleration due to gravity
- :math:`b_1`: x coordinate of the final point
- :math:`b_2`: y coordinate of the final point

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
from opty.direct_collocation import Problem
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %%
# Equations of Motion
# -------------------
N, A = me.ReferenceFrame('N'), me.ReferenceFrame('A')
O, P = sm.symbols('O P', cls=me.Point)
O.set_vel(N, 0)
t = me.dynamicsymbols._t

m, g = sm.symbols('m g')
x, y, ux, uy = me.dynamicsymbols('x y u_x u_y')
beta = me.dynamicsymbols('beta')

A.orient_axis(N, beta, N.z)

P.set_pos(O, x*N.x + y*N.y)
P.set_vel(N, ux*N.x + uy*N.y)

body = [me.Particle('body', P, m)]
forces = [(P, -m*g*N.y)]
speed_constr =  sm.Matrix([P.vel((N)).dot(A.y)])

kd = sm.Matrix([ux - x.diff(t), uy - y.diff(t)])
kane = me.KanesMethod(
            N, q_ind=[x, y],
            u_ind=[ux],
            u_dependent=[uy],
            kd_eqs=kd,
            velocity_constraints=speed_constr)
fr, frstar = kane.kanes_equations(body, forces)

eom = kd.col_join(fr + frstar)
eom = eom.col_join(speed_constr)
MathJaxRepr(eom)

# %%
# Set up the Optimization Problem and Solve It
# --------------------------------------------
h = sm.symbols('h')
num_nodes = 201
t0, tf = 0.0, h*(num_nodes - 1)
interval = h

state_symbols = (x, y, ux, uy)

# B(b1/b2) is the final point, the starting point is (0/0), always
b1 = 10.0
b2 = -6.0

# %%
# Set up the constraints and bounds, objective function and its gradient
# for the problem.
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
}

par_map = {m: 1.0, g: 9.81}

def obj(free):
    return free[-1]

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad

# %%
# If a solution exists, backend = 'numpy' is better as it sets up ``Problem``
# much faster. If no solution is available, use 'cython' as the backend, as it
# is faster for solving the problem.
fname = f'brachistochrone_{num_nodes}_nodes_solution.csv'
if os.path.exists(fname):
    backend = 'numpy'
else:
    backend = 'cython'
prob = Problem(
        obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval,
        time_symbol=t,
        known_parameter_map=par_map,
        instance_constraints=instance_constraint,
        bounds=bounds,
        backend=backend,
)

# %%
# Solve the problem. Use the given solution if available, else pick a
# reasonable initial guess and solve the problem.

if os.path.exists(fname):
    # Take the given solution
    solution = np.loadtxt(fname)
else:
    # Solve the problem.
    initial_guess = np.random.randn(prob.num_free) * 0.1
    prob.add_option('max_iter', 15000)
    for _ in range(10):
        solution, info = prob.solve(initial_guess)
        initial_guess = solution
        print(info['status_msg'])

    _ = prob.plot_objective_value()

#np.savetxt(fname, solution, fmt='%.12f')
# %%
_ = prob.plot_trajectories(solution)
# %%
_ = prob.plot_constraint_violations(solution)
# %%
# Animate the Solution
# --------------------
fps = 35

#%%
# Calculate the Brachistochrone from (0, 0) to (b1, b2). It is very sensitive
# to the initial guess.
def func(X0):
    """gives the Brachistochrome equation for the starting point at (0/0) and
    the final point at (b1/b2)"""
    R = X0[0]
    theta = X0[1]
    return [R*theta - R*np.sin(theta) - b1, R*(1.0 - np.cos(theta)) - b2]

# These initial guesses do not work for all b1, b2. It is very sensitive to
# the initial guess..
X0 = [1.0, 1.0]
resultat = root(func, X0)

times = np.linspace(0.0, resultat.x[1], num_nodes)
XX = resultat.x[0]*times - resultat.x[0]*np.sin(times)
YY = resultat.x[0] * (1.0 - np.cos(times))
# %%
# Set up the plot.
state_vals, input_vals, *_ = prob.parse_free(solution)
t_arr = np.linspace(t0, num_nodes*solution[-1], num_nodes)
state_sol = CubicSpline(t_arr, state_vals.T)
input_sol = CubicSpline(t_arr, input_vals.T)

xmin = -1.0
xmax = b1 + 1.0
ymin = b2 - 1.0
ymax = 1.0

# additional points for the speed vector
arrow_head = me.Point('arrow_head')
arrow_head.set_pos(P, ux*N.x + uy*N.y)

coordinates = P.pos_from(O).to_matrix(N)
coordinates = coordinates.row_join(arrow_head.pos_from(O).to_matrix(N))

pL, pL_vals = zip(*par_map.items())
coords_lam = sm.lambdify(list(state_symbols) + [beta] + list(pL),
                coordinates, cse=True)

def init_plot():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)

    # Plot the Brachistochrone
    ax.plot(XX, YY, color='black', lw=0.5)
    ax.scatter([0], [0], color='red', s=10)
    ax.scatter([b1], [b2], color='green', s=10)

    # set up the point and the speed arrow.
    punkt = ax.scatter([], [], color='red', s=50)
    # draw the speed vektor
    pfeil = ax.quiver([], [], [], [], color='green', scale=25, width=0.004)

    return fig, ax, punkt, pfeil

# Function to update the plot for each animation frame
fig, ax, punkt, pfeil = init_plot()

def update(t):
    message = (f'running time {t:.2f} sec \n The green arrow is the speed \n'
               f'Slow motion video.')
    ax.set_title(message, fontsize=12)

    coords = coords_lam(*state_sol(t), input_sol(t), *pL_vals)
    punkt.set_offsets([coords[0, 0], coords[1, 0]])

    pfeil.set_offsets([coords[0, 0], coords[1, 0]])
    pfeil.set_UVC(coords[0, 1] - coords[0, 0] , coords[1, 1] - coords[1, 0])

# %%
# Create the animation.
fig, ax, punkt, pfeil = init_plot()
animation = FuncAnimation(fig, update, frames=np.arange(t0,
                (num_nodes+1)*solution[-1], 1 / fps), interval=3000/fps)

# %%
# A frame from the animation.
# sphinx_gallery_thumbnail_number = 4

plt.show()
