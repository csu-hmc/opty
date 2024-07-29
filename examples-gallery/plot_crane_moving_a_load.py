 # %%
"""
Crane moving a load
===================

A load is moved by a crane. The load is rotably connected to the crane by a
massless rod. The goal is to move the load from the initial position to the
final position in the shortest possible time. The load must not 'overswing'
the final position. The load is moved by a force applied to its
suspension point. The jib of the crane is extended in the horizontal
X direction.

**Constants**

- l: length of the rod attaching the load to the crane [m]
- m1: mass of mover attached to the arm of the crane [kg]
- m2: mass of the load [kg]
- g: acceleration due to gravity [m/s²]

**States**

- xc: x-coordinate of the mover [m]
- xl: x-coordinate of the load [m]  dependent coordinate
- yl: y-coordinate of the load [m]  dependent coordinate
- q: angle of the rod [rad]
- uxc: velocity of the mover in x-direction [m/s]
- uxl: velocity of the load in x-direction [m/s] dependent speed
- uyl: velocity of the load in y-direction [m/s] dependent speed
- u: angular velocity of the rod [rad/s]

**Specifieds**

- F: force applied to the mover [N]

"""
import sympy.physics.mechanics as me
from collections import OrderedDict
import numpy as np
import sympy as sm
from scipy.interpolate import CubicSpline
from opty.direct_collocation import Problem
from opty.utils import parse_free
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches

# %%
# Set up Kane's equations of motion.
# Just the standard way of doing it.

N, A = sm.symbols('N A', cls = me.ReferenceFrame)
t = me.dynamicsymbols._t
O, P1, P2 = sm.symbols('O P1 P2', cls = me.Point)
O.set_vel(N, 0)
xc, xl, yl, q, uxc, uxl, uyl, u, F = me.dynamicsymbols('xc, xl, yl, q, uxc,' +
    'uxl, uyl, uq, F')
l, m1, m2, g = sm.symbols('l, m1, m2, g')

A.orient_axis(N, q, N.z)
A.set_ang_vel(N, u * N.z)

P1.set_pos(O, xc * N.x)
P1.set_vel(N, uxc * N.x)

P2.set_pos(P1, -l * A.y)
P2.v2pt_theory(P1, N, A)

P1a = me.Particle('P1a', P1, m1)
P2a = me.Particle('P2a', P2, m2)

BODY = [P1a, P2a]

FL = [(P1, F * N.x - m1*g*N.y), (P2, -m2*g*N.y)]
kd = sm.Matrix([uxc - xc.diff(t), u - q.diff(t), uxl - xl.diff(t),
    uyl - yl.diff(t)])

config_constr = sm.Matrix([xl - xc - l * sm.sin(q), yl + l * sm.cos(q)])
speed_constr = config_constr.diff(t)

q_ind = [xc, q]
q_dep = [xl, yl]
u_ind = [uxc, u]
u_dep = [uxl, uyl]

KM = me.KanesMethod(N,
    q_ind=q_ind,
    q_dependent=q_dep,
    u_ind=u_ind,
    u_dependent=u_dep,
    kd_eqs=kd,
    configuration_constraints=config_constr,
    velocity_constraints=speed_constr)

(fr, frstar) = KM.kanes_equations(BODY, FL)
EOM = kd.col_join(fr + frstar)
EOM = EOM.col_join(config_constr)
sm.pprint(EOM)
# %%
# Set up the optimization problem and solve it.
# Note: While opty could calculate the reaction forces on, say, the mover,
# (add them to the state variables), my trials showed that even with this
# small example the calculation time increases enourmously.

state_symbols = tuple((*q_ind, *q_dep, *u_ind, *u_dep))
constant_symbols = (l, m1, m2, g)
specified_symbols = (F,)
h_opty = sm.symbols('h_opty')
unknown_symbols = []
methode = "backward euler"

num_nodes = 250
duration =(num_nodes - 1) * h_opty
interval_value = h_opty

# Specify the known system parameters.
par_map = OrderedDict()
par_map[l] = 5.0
par_map[m1] = 1.0
par_map[m2] = 10.0
par_map[g] = 9.81

t0, tf = 0.0, duration

def obj(free):
    """Minimize h_opty, the time interval between nodes."""
    return free[-1]

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.
    return grad

# start location and and final location of the load.
anfang = 0.0
ende = 15.0

# %%
# Set the initial and final states to form the instance constraints.
initial_state_constraints = {xc: anfang, xl: anfang, yl: -par_map[l], q: 0.0,
    uxc: 0.0, uxl: 0.0, uyl: 0.0, u: 0.0}
final_state_constraints = {xc: ende, xl: ende, yl: -par_map[l], q: 0.0,
    uxc: 0.0, uxl: 0.0, uyl: 0.0, u: 0.0}
instance_constraints = (
) + tuple(
    xi.subs({t: t0}) - xi_val for xi, xi_val in
        initial_state_constraints.items()
) + tuple(
    xi.subs({t: tf}) - xi_val for xi, xi_val in
        final_state_constraints.items()
)
# %%
# Forcing h_opty > 0.0 sometimes avoids negative 'solutions'. Here it also
# seem to help with the convergence of the optimization: Different bounds,
# e,g, h_opty = (1.e-4, 1.) gives unreasonable results.

bounds = {
    F: (-20., 20.),
    xl: (anfang, ende),
    xc: (anfang, ende),
    h_opty: (1.e-5, 0.5),
    }

# Create the optimization problem.
prob = Problem(
    obj,
    obj_grad,
    EOM,
    state_symbols,
    num_nodes,
    interval_value,
    known_parameter_map=par_map,
    instance_constraints=instance_constraints,
    bounds=bounds,
    integration_method=methode,
    )
# %%
# Using a 'reasonable' initial guess speeds up the optimization process a lot.
# 'Bad' initial guesses may even prevent convergence.
i1 = [(ende - anfang) / num_nodes * i for i in range(num_nodes)]
i2 = [0. for _ in range(num_nodes)]
i3 = i1
i4 = [-par_map[l] for _ in range(num_nodes)]
i5 = [0. for _ in range(5*num_nodes)]
i6 = [0.01]
initial_guess = np.array(i1 + i2 + i3 + i4 + i5 + i6)

# %%
# Plot the initial guess.
prob.plot_trajectories(initial_guess)


# this allows to change the maximum number of iterations.
# Standard is 3000.
prob.add_option('max_iter', 1000)
# Find the optimal solution.
for _ in range(1):
    solution, info = prob.solve(initial_guess)
    print('message from optimizer:', info['status_msg'])
    prob.plot_objective_value()
    initial_guess = solution
print('Iterations needed', len(prob.obj_value))
print(f"Objective value {solution[-1]: .3e}")

# %%
# plot the accuracy of the solution.
prob.plot_constraint_violations(solution)

# %%
# Plot the state trajectories.
fig, ax = plt.subplots(9, 1, figsize=(7.25, 3.0*9), sharex=True)
prob.plot_trajectories(solution, ax)

# %%
# Animate the simulation.
fps = 30

tf = solution[-1] * (num_nodes - 1)
state_vals, input_vals, _ = parse_free(solution, len(state_symbols),
    len(specified_symbols), num_nodes)
t_arr = np.linspace(t0, tf, num_nodes)
state_sol = CubicSpline(t_arr, state_vals.T)
input_sol = CubicSpline(t_arr, input_vals.T)

coordinates = P1.pos_from(O).to_matrix(N)
coordinates = coordinates.row_join(P2.pos_from(O).to_matrix(N))

pL, pL_vals = zip(*par_map.items())
coords_lam = sm.lambdify((*state_symbols, F, *pL),
    coordinates, cse=True)

# Dimensions of the mover and the load.
width, height, radius = 0.5, 0.5, 0.5

def init_plot():
    xmin, xmax = anfang-1, ende+1
    ymin, ymax = -par_map[l]-1, 1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('X-axis [m]')
    ax.set_ylabel('Y-axis [m]')

    ax.axhline(0., color='black', lw=1.5)
    ax.axvline(ende+radius, color='black', lw=1.0)
    ax.axvline(anfang-radius, color='black', lw=1.0)
    ax.annotate(f'walls', xy=(ende+radius, -2),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=-.2',
        lw=0.25), xytext=(8, 0.5), fontsize=9)
    ax.annotate(f'', xy=(anfang-radius, -1),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=-.2',
        lw=0.25), xytext=(8, 0.5))

    line1, = ax.plot([], [], color='orange', lw=1)
    pfeil = ax.quiver([], [], [], [], color='red', scale=55, width=0.002,
        headwidth=8 )
    recht = patches.Rectangle((-width/2, -height/2),
        width=width, height=height, fill=True, color='red', ec='black')
    ax.add_patch(recht)
    load = patches.Circle((0, -par_map[l]), radius=radius, fill=True,
        color='blue',ec='black')
    ax.add_patch(load)
    return fig, ax, line1, recht, load, pfeil

fig, ax, line1, recht, load, pfeil = init_plot()

# Function to update the plot for each animation frame
def update(t):
    message = f'running time {t:.2f} sec \n The red arrow shows the force.'
    ax.set_title(message, fontsize=12)

    coords = coords_lam(*state_sol(t), input_sol(t), *pL_vals)

    line1.set_data([coords[0, 0], coords[0, 1]], [coords[1, 0], coords[1, 1]])
    recht.set_xy((coords[0, 0] - width/2., coords[1, 0] - height/2.))
    load.set_center((coords[0, 1], coords[1, 1]))
    pfeil.set_offsets([coords[0, 0], coords[1, 0]+0.25])
    pfeil.set_UVC(input_sol(t) , 0.25)
    return line1, recht, load, pfeil


# Create the animation
anim = animation.FuncAnimation(fig, update, frames=np.arange(t0, tf, 1 / fps),
    interval=fps)

# %%
# A frame from the animation.
fig, ax, line1, recht, load, pfeil = init_plot()

# sphinx_gallery_thumbnail_number = 6
update(2.0)

plt.show()