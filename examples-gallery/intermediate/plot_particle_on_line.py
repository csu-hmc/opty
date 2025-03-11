# %%
r"""
Particle Follows a Line
=======================

Objectives
----------

- Show how the equations of motion may be modified using differentiable
  functions.
- Show how suitable iterations from a simpler configuration to a more
  difficult one may achieve convergence.
- Show the use of the kwarg ``backend='numpy'`` in the Problem class.


Introduction
------------

The basic task it for a particle to reach its final destination as fast as
possible, while roughly staying within a 'band' of width ``radius`` around a
curve. Starting point and final point are on the curve.
The control variable is a force acting on the particle.


Detailed Description
--------------------

Calculate the distance of the particle from the curve. This is simple in 2D.
It leads to a nonlinear equation for the coordinate ``r`` of the curve where
it is closest to the particle. This equation is added to the equations of
motion.

1. Define a differentiable hump function that is one if the particle is inside
   the band and zero outside. When setting up the equations of motion,
   the forces acting on the particle are multiplied by this hump function.
   So, the particle may leave the band, but then no more controls are available.

2. Define an additional state variable ``dist`` = distance(curve to particle).
   This becomes an additional equation of motion. Now bound
   :math:`0 \leq` dist :math:`\leq` radius. This means the particle cannot
   leave the band.


Notes
-----

1. Both ways converge with amazing difficulty, given the apparent simplicity
   of the problem.
2. As instance constraints, bounds and parameters are changed in the iteration
   process, ``Problem`` must be inside the loop. The kwarg ``backend='numpy'``
   is crucial here to cut down the time needed for the iterations.
3. The method using bounds seems to iterate faster than the one using the hump
   - but not reliably so all the time.

**States**

- :math:`x` is the x-coordinate of the particle.
- :math:`y` is the y-coordinate of the particle.
- :math:`u_x` is the x-component of the velocity of the particle.
- :math:`u_y` is the y-component of the velocity of the particle.
- :math:`r` is the x-coordinate of the point on the curve that is closest to
   the particle.
- :math:`dist` is the distance of the particle from the curve.( only needed for
   method 2)

**Controls**

- :math:`F_x` is the x-component of the force acting on the particle.
- :math:`F_y` is the y-component of the force acting on the particle.

**Parameters**

- :math:`m` is the mass of the particle.
- :math:`g` is the acceleration due to gravity.
- :math:`friction` is the friction coefficient.
- :math:`radius` is the width of the band around the curve.
- :math:`steep` is a parameter that sets the steepness of the hump function.
- :math:`\alpha` and :math:`\beta` are parameters of the curve.

"""
import os
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty import Problem
from opty.utils import parse_free
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.interpolate import CubicSpline

# %%
# First Version
# -------------

# Define the curve:
x = sm.symbols('x')
alpha, beta = sm.symbols('alpha, beta')
def curve(x, alpha, beta):
    return beta * (1 - sm.exp(-alpha * x))

# Define differentiable hump function.
def hump(x, a, b, gr):
    # approx one for x in [a, b]
    # approx zero otherwise
    # the higher gr the closer the approximation
    return 1.0 - (1/(1 + sm.exp(gr*(x - a))) + 1/(1 + sm.exp(-gr*(x - b))))

# Plot the curve and the hump function.
alpha1, beta1, xx = sm.symbols('alpha1, beta1, xx')
a1, b1, gr1 = sm.symbols('a1, b1, gr1')
XX = np.linspace(-10, 10, 200)
XX1 = np.linspace(0.0, 10, 200)
curve_lam = sm.lambdify((xx, alpha1, beta1), curve(xx, alpha1, beta1))
hump_lam = sm.lambdify((xx, a1, b1, gr1), hump(xx, a1, b1, gr1))
fig, ax = plt.subplots(2, 1, figsize=(6.5, 3), layout='tight')
ax[0].plot(XX1, curve_lam(XX1, 1, 2))
ax[0].set_title('curve')
ax[1].plot(XX, hump_lam(XX, -1, 1, 10.0))
_ = ax[1].set_title('hump')
# %%
# Set the functions to calculate the closest point on the curve and the
# distance to the curve.
def closest_point_on_curve(r, x0, y0):
    """
    If (x0, y0) is the point, then the value or r such than (r, curve(r))
    is closest to the point is obtained by solving
    d/dr(distance of particle to curve)} = 0
    that is
    (r - x0) + (curve(r) - y0) * curve'(r) = 0.
    The l.h.s. of this equation will be returned
    """
    return ((r - x0) + (curve(r, alpha, beta) - y0)
            * curve(r, alpha, beta).diff(r))

def distance(r, x0, y0):
    """r ist the value such that (r, curve(r) is closest to (x0, y0)
    This distance is returned"""

    return sm.sqrt((r - x0)**2 + (curve(r, alpha, beta) - y0)**2)

# %%
# Set up the equations of motion.
N = me.ReferenceFrame('N')
O, Point = me.Point('O'), me.Point('Point')
O.set_vel(N, 0)
t = me.dynamicsymbols._t

x, y, ux, uy = me.dynamicsymbols('x, y, u_x, u_y')
Fx, Fy = me.dynamicsymbols('F_x, F_y')
r, dist = me.dynamicsymbols('r, dist')

m, g, friction, radius = sm.symbols('m, g, friction, radius')
steep = sm.symbols('steep')
body = me.Particle('body', Point, m)

Point.set_pos(O, x*N.x + y*N.y)
Point.set_vel(N, ux*N.x + uy*N.y)

# forces
distanz = distance(r, x, y)
hump1 = hump(distanz, -radius, radius, steep)
force =[(Point, -m*g*N.y + hump1*(Fx*N.x + Fy*N.y)
         - friction*(ux*N.x + uy*N.y))]

q_ind = [x, y]
u_ind = [ux, uy]
kd =sm.Matrix([ux - x.diff(t), uy - y.diff(t)])
kane = me.KanesMethod(N,
                      q_ind,
                      u_ind,
                      kd_eqs= kd,
)

fr, frstar = kane.kanes_equations([body], force)
eom = kd.col_join(fr + frstar)
eom = eom.col_join(sm.Matrix([closest_point_on_curve(r, x, y)]))
sm.pprint(eom)

# %%
# Set Up the Optimazation Problem
h = sm.symbols('h')
num_nodes = 101
t0, tf = 0.0, h*(num_nodes - 1)
interval = h

state_symbols = (x, y, ux, uy, r)
specified_symbols = (Fx, Fy)
# %%
# Set up the objective function, its gradient, bounds, the constraints,
# and the ``Problem``.
def obj(free):
    """Objective function to minimize duration. THe variable time interval
    is the last element in the free variables."""
    return free[-1]

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad

par_map = {}
par_map[m] = 1.0
par_map[g] = 9.81
par_map[friction] = 0.1
par_map[radius] = 5.0
par_map[steep] = 5.0
par_map[alpha] = 1.0
par_map[beta] = 2.0

instance_constraints = [
    x.func(t0) - 0.0,
    y.func(t0) - 0.0,
    ux.func(t0) - 0.0,
    uy.func(t0) - 0.0,
    r.func(t0) - 0.0,

    x.func(tf) - 10.0,
    y.func(tf) - par_map[beta],
    ux.func(tf) - 0.0,
    uy.func(tf) - 0.0,
]

limit = 175.0
bounds = {
    Fx: (-limit, limit),
    Fy: (-limit, limit),
    h: (0.0, 1.0),
    r: (0.0, 10.0),
}
# %%
# Set the initial guess, if no solution is available
fname = f'particle_follow_line_force_{num_nodes}_nodes_solution.csv'
# If as solution is available it is used. Esle long interatinos are needed.
if os.path.exists(fname):
    # Use existing solution.
    solution = np.loadtxt(fname)
    par_map[beta] = 5.7
    par_map[radius] = 1.24
    instance_constraints[-3] = y.func(tf) - par_map[beta]
    prob = Problem(obj,
               obj_grad,
               eom,
               state_symbols,
               num_nodes,
               interval,
               known_parameter_map=par_map,
               instance_constraints=instance_constraints,
               bounds=bounds,
               time_symbol=t,
               backend='numpy',
    )
else:
    # Solve the problem. One has to gradually increase the value of beta
    # and decrease the value of radius to get a solution.

    for i in range(48):
        par_map[beta] = 1.0 + 0.1*i
        par_map[radius] = 5.0 - 0.08*i
        instance_constraints[-3] = y.func(tf) - par_map[beta]
        prob = Problem(obj,
               obj_grad,
               eom,
               state_symbols,
               num_nodes,
               interval,
               known_parameter_map=par_map,
               instance_constraints=instance_constraints,
               bounds=bounds,
               time_symbol=t,
               backend='numpy',
        )

        x_values = np.linspace(0, 10, num_nodes)
        y_values = np.linspace(0, par_map[beta], num_nodes)
        initial_guess = np.ones(prob.num_free) * 0.01
        initial_guess[0:num_nodes] = x_values
        initial_guess[num_nodes:2*num_nodes] = y_values

        solution, info = prob.solve(initial_guess)
        initial_guess = solution
        print(f'{i + 1} - th iteration')
        print(info['status_msg'])

# %%
_ = prob.plot_trajectories(solution)
# %%
_ =prob.plot_constraint_violations(solution)

# %%
# Animate the simulation. It is made as a function, so it can easily be used
# for the second version.
def make_animation(state_symbols, simple=False):

    # Needed to plot the lines equidistant from the curve.
    xc, yc = sm.symbols('xc, yc')
    PC, PR, PL = me.Point('PC'), me.Point('PR'), me.Point('PL')
    PC.set_pos(O, xc*N.x + curve(xc, alpha, beta)*N.y)
    yc = -1 / curve(xc, alpha, beta).diff(xc)
    faktor = par_map[radius] / sm.sqrt(1 + yc**2)
    PR.set_pos(PC, faktor*N.x + faktor*yc*N.y)
    PL.set_pos(PC, -faktor*N.x - faktor*yc*N.y)
    coord_PR = PR.pos_from(O).to_matrix(N)
    coord_PL = PL.pos_from(O).to_matrix(N)
    coord_PR_lam = sm.lambdify((xc, alpha, beta), coord_PR, cse=True)
    coord_PL_lam = sm.lambdify((xc, alpha, beta), coord_PL, cse=True)

    h_sol = solution[-1]
    fps = 30

    tf = h_sol*(num_nodes - 1)
    state_vals, input_vals, _ = parse_free(solution, len(state_symbols),
                                       len(specified_symbols), num_nodes)
    t_arr = np.linspace(t0, tf, num_nodes)
    state_sol = CubicSpline(t_arr, state_vals.T)
    input_sol = CubicSpline(t_arr, input_vals.T)

    # tip of force arrow
    Tip = me.Point('Tip')
    distanz = distance(r, x, y)
    if simple:
        Tip.set_pos(Point, (Fx*N.x + Fy*N.y))
    else:
        Tip.set_pos(Point, hump(distanz, -radius, radius, steep) *
                    (Fx*N.x + Fy*N.y))

    coordinates = Point.pos_from(O).to_matrix(N)
    coordinates = coordinates.row_join(Tip.pos_from(O).to_matrix(N))

    pl, pl_vals = zip(*par_map.items())
    coords_lam = sm.lambdify((*state_symbols, *specified_symbols, *pl),
                    coordinates, cse=True)

    def init_plot():
        xmin, xmax = - par_map[radius] - 1, 11.0 + par_map[radius]
        ymin, ymax = -1.0, par_map[beta] + 2.0

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.set_xlabel('X-axis [m]')
        ax.set_ylabel('Y-axis [m]')

        xx = np.linspace(0, 10, 100)
        ax.plot(xx, curve_lam(xx, par_map[alpha], par_map[beta]), color='blue',
            lw=1)

        koordR = [coord_PR_lam(xxx, par_map[alpha], par_map[beta]) for xxx in
              np.linspace(0, 10, 100)]
        koordL = [coord_PL_lam(xxx, par_map[alpha], par_map[beta]) for xxx in
              np.linspace(0, 10, 100)]
        for i in range(99):
            ax.plot([koordR[i][0], koordR[i+1][0]], [koordR[i][1],
                 koordR[i+1][1]], color='blue', lw=0.5)
            ax.plot([koordL[i][0], koordL[i+1][0]], [koordL[i][1],
                 koordL[i+1][1]], color='blue', lw=0.5)


        line1 = ax.scatter([], [], color='red', marker='o', s=50)
        pfeil = ax.quiver([], [], [], [], color='green', scale=850,
                          width=0.002, headwidth=8)

        return fig, ax, line1, pfeil
    fig, ax, line1, pfeil = init_plot()

    def update(t):
        if simple:
            message = (f'Version with bounds \n Running time {t:0.2f} sec \n'
                f'The green arrow shows the force. \n Slow motion')
        else:
            message = (f'Version with bounded Force \n Running time {t:0.2f} '
                f'sec \n The green arrow shows the force. \n Slow motion')
        ax.set_title(message, fontsize=12)

        coords = coords_lam(*state_sol(t), *input_sol(t), *pl_vals)
    #    path.append(coords)
    #    ax.plot([path[i][0][0] for i in range(0, len(path)-1)],
    #            [path[i][1][0] for i in range(0, len(path)-1)], color='red',
    #            lw=0.25, alpha=0.5)

        line1.set_offsets([coords[0, 0], coords[1, 0]])
        pfeil.set_offsets([coords[0, 0], coords[1, 0]])
        pfeil.set_UVC(coords[0, 1], coords[1, 1])
        return line1, pfeil

    anim = animation.FuncAnimation(fig, update,
                               frames=np.arange(t0, tf, 1/fps),
                               interval=1/fps*1000*5)

    return anim, update, fig, ax, line1, pfeil

# %%
anim, *_ = make_animation(state_symbols, simple=False)
plt.show()
# %%
# Second Version
# --------------
# The only differences to the first version in the equations of motion are
#  - a state variable ``dist`` = distance(curve to particle) is added.
#  - the force acting on the particle is without the hump function.

dist = me.dynamicsymbols('dist')

# forces
force =[(Point, -m*g*N.y + Fx*N.x + Fy*N.y - friction*(ux*N.x + uy*N.y))]

kane = me.KanesMethod(N,
                      q_ind,
                      u_ind,
                      kd_eqs= kd,
)

fr, frstar = kane.kanes_equations([body], force)
eom = kd.col_join(fr + frstar)
eom = eom.col_join(sm.Matrix([closest_point_on_curve(r, x, y)]))
eom = eom.col_join(sm.Matrix([dist - distance(r, x, y)]))
sm.pprint(eom)

# %%
# Set Up the optimization problem.

state_symbols = (x, y, ux, uy, r, dist)
specified_symbols = (Fx, Fy)

# %%
# Set up an initial guess, if no solution is available.
fname = f'particle_follow_line_bound_{num_nodes}_nodes_solution.csv'
aaa = 10
if os.path.exists(fname):
    solution = np.loadtxt(fname)
    par_map[beta] = 5.7
    par_map[radius] = 1.24
    instance_constraints[-3] = y.func(tf) - par_map[beta]
    prob = Problem(obj,
               obj_grad,
               eom,
               state_symbols,
               num_nodes,
               interval,
               known_parameter_map=par_map,
               instance_constraints=instance_constraints,
               bounds=bounds,
               time_symbol=t,
               backend='numpy',
    )
else:
    # Solve the problem. Similar comments as above apply.
    for i in range(48):
        par_map[beta] = 1.0 + 0.1*i
        par_map[radius] = 5.0 - 0.08*i
        instance_constraints[-3] = y.func(tf) - par_map[beta]
        bounds[dist] = (0.0, par_map[radius])
        prob = Problem(obj,
               obj_grad,
               eom,
               state_symbols,
               num_nodes,
               interval,
               known_parameter_map=par_map,
               instance_constraints=instance_constraints,
               bounds=bounds,
               time_symbol=t,
               backend='numpy',
)
        x_values = np.linspace(0, 10, num_nodes)
        y_values = np.linspace(0, par_map[beta], num_nodes)
        initial_guess = np.ones(prob.num_free) * 0.01
        initial_guess[0:num_nodes] = x_values
        initial_guess[num_nodes:2*num_nodes] = y_values

        solution, info = prob.solve(initial_guess)
        print(f'{i + 1} - th iteration')
        initial_guess = solution
        print(info['status_msg'])

# %%
_ = prob.plot_trajectories(solution)
# %%
_ =prob.plot_constraint_violations(solution)

# %%
# Animate the Simulation.
anim, *_ = make_animation(state_symbols, simple=True)
plt.show()

# A frame from the animation.
# sphinx_gallery_thumbnail_number = 5
