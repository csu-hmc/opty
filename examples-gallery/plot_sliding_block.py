# %%
"""
Block Sliding on a Road
=======================

A block, modeled as a particle is sliding on a road to cross a
hill. The block is subject to gravity and speed dependent
friction.
Gravity points in the negative Y direction.
A force tangential to the road is applied to the block.
Two objective functions to be minimized will be considered:

- selektion = 0: time to reach the end point is minimized
- selektion = 1: energy consumed is minimized.

**Constants**

- m: mass of the block [kg]
- g: acceleration due to gravity [m/s**2]
- reibung: coefficient of friction [N/(m*s)]
- a, b: paramenters determining the shape of the road.

**States**

- x: position of the block [m]
- ux: velocity of the block [m/s]

**Specifieds**

- F: force applied to the block [N]

"""
import sympy.physics.mechanics as me
import numpy as np
import sympy as sm
from opty.direct_collocation import Problem
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from copy import deepcopy
# %%
# The function below defines the shape of the road the block is
# sliding on.
def strasse(x, a, b):
    return a * x**2 * sm.exp((b - x))

# %%
# Set up Kane's EOMs.
N = me.ReferenceFrame('N')
O = me.Point('O')
O.set_vel(N, 0)
t = me.dynamicsymbols._t

P0 = me.Point('P0')
x = me.dynamicsymbols('x')
ux = me.dynamicsymbols('u_x')
F = me.dynamicsymbols('F')

m, g, reibung = sm.symbols('m, g, reibung')
a, b = sm.symbols('a b')

P0.set_pos(O, x * N.x + strasse(x, a, b) * N.y)
P0.set_vel(N, ux * N.x + strasse(x, a, b).diff(x)*ux * N.y)
bodies = [me.Particle('P0', P0, m)]
# %%
# The control force and the friction are acting in the direction of
# the tangent at the street at the point whre the particle is.
alpha = sm.atan(strasse(x, a, b).diff(x))
forces = [(P0, -m*g*N.y + F*(sm.cos(alpha)*N.x + sm.sin(alpha)*N.y) -
       reibung*ux*(sm.cos(alpha)*N.x + sm.sin(alpha)*N.y))]

kd = sm.Matrix([ux - x.diff(t)])

q_ind = [x]
u_ind = [ux]

KM = me.KanesMethod(N, q_ind=q_ind, u_ind=u_ind, kd_eqs=kd)
(fr, frstar) = KM.kanes_equations(bodies, forces)
eom = kd.col_join(fr + frstar)
eom.simplify()
sm.pprint(eom)

# %%
# opty seems to overwrite the symbols. Since two optimizations are run, the
# original symbols are saved.
speicher = deepcopy((x, ux, F))

# %%
# Store the results of the two optimizations for later plotting
solution_list = [0., 0.]
prob_list = [0., 0.]
info_list = [0., 0.]

# %%
# Define the known parameters.
par_map = {}
par_map[m] = 1.0
par_map[g] = 9.81
par_map[reibung] = 0.0
par_map[a] = 1.5
par_map[b] = 2.5

num_nodes = 150
fixed_duration = 6.0
# %%
# Set up the optinization problems and solve them.
for selektion in (0, 1):
    state_symbols = tuple((speicher[0], speicher[1]))
    laenge = len(state_symbols)
    constant_symbols = (m, g, reibung, a, b)
    specified_symbols = (speicher[2], )

    if selektion == 1:
        duration = fixed_duration
        interval_value = duration / (num_nodes - 1)

        def obj(free):
            Fx = free[laenge * num_nodes: (laenge + 1) * num_nodes]
            return interval_value * np.sum(Fx**2)

        def obj_grad(free):
            grad = np.zeros_like(free)
            l1 = laenge * num_nodes
            l2 = (laenge + 1) * num_nodes
            grad[l1: l2] = 2.0 * free[l1: l2] * interval_value
            return grad


    else:
        h = sm.symbols('h')
        duration = (num_nodes - 1) * h
        interval_value = h

        def obj(free):
            return free[-1]

        def obj_grad(free):
            grad = np.zeros_like(free)
            grad[-1] = 1.
            return grad

    t0, tf = 0.0, duration

    methode = 'backward euler'

    if selektion == 0:
        initial_guess = np.array(list(np.ones((len(state_symbols)
            + len(specified_symbols)) * num_nodes) * 0.01) + [0.02])
    else:
        initial_guess = np.ones((len(state_symbols) +
            len(specified_symbols)) * num_nodes) * 0.01

    initial_state_constraints = {x: 0., ux: 0.}

    final_state_constraints = {x: 10., ux: 0.}

    instance_constraints = (
        tuple(xi.subs({t: t0}) - xi_val for xi, xi_val
            in initial_state_constraints.items()) +
        tuple(xi.subs({t: tf}) - xi_val for xi, xi_val
            in final_state_constraints.items())
    )

    if selektion == 0:
        bounds = {F: (-15., 15.), x: (initial_state_constraints[x],
        final_state_constraints[x]), ux: (0., 1000.), h:(1.e-5, 1.)}
    else:
        bounds = {F: (-15., 15.), x: (initial_state_constraints[x],
        final_state_constraints[x]), ux: (0., 100)}

    prob = Problem(obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        known_parameter_map=par_map,
        instance_constraints=instance_constraints,
        bounds=bounds,
        integration_method=methode)

    prob.add_option('max_iter', 3000)

    solution, info = prob.solve(initial_guess)
    solution_list[selektion] = solution
    info_list[selektion] = info
    prob_list[selektion] = prob
# %%
# Animate the solutions and plot the results.
def drucken(selektion, fig, ax, video = True):
    solution = solution_list[selektion]
    info = info_list[selektion]
    prob = prob_list[selektion]

    if selektion == 0:
        duration = (num_nodes - 1) * solution[-1]
    else:
        duration = fixed_duration
    times = np.linspace(0.0, duration, num=num_nodes)
    interval_value = duration / (num_nodes - 1)

    strasse1 = strasse(x, a, b)
    strasse_lam = sm.lambdify((x, a, b), strasse1, cse=True)

    P0_x = solution[:num_nodes]
    P0_y = strasse_lam(P0_x, par_map[a], par_map[b])

# find the force vector applied to the block
    alpha = sm.atan(strasse(x, a, b).diff(x))
    Pfeil = [F*sm.cos(alpha),  F*sm.sin(alpha)]
    Pfeil_lam = sm.lambdify((x, F, a, b), Pfeil, cse=True)

    l1 = laenge * num_nodes
    l2 = (laenge + 1) * num_nodes
    Pfeil_x = Pfeil_lam(P0_x, solution[l1: l2], par_map[a], par_map[b])[0]
    Pfeil_y = Pfeil_lam(P0_x, solution[l1: l2], par_map[a], par_map[b])[1]

# needed to give the picture the right size.
    xmin = np.min(P0_x)
    xmax = np.max(P0_x)
    ymin = np.min(P0_y)
    ymax = np.max(P0_y)

    def initialize_plot():
        ax.set_xlim(xmin-1, xmax + 1.)
        ax.set_ylim(ymin-1, ymax + 1.)
        ax.set_aspect('equal')
        ax.set_xlabel('X-axis [m]')
        ax.set_ylabel('Y-axis [m]')

        if selektion == 0:
            msg = f'The speed is optimized'
        else:
            msg = f'The energy optimized'

        ax.grid()
        strasse_x = np.linspace(xmin, xmax, 100)
        ax.plot(strasse_x, strasse_lam(strasse_x, par_map[a], par_map[b]),
            color='black', linestyle='-', linewidth=1)
        ax.axvline(initial_state_constraints[x], color='r',
            linestyle='--', linewidth=1)
        ax.axvline(final_state_constraints[x], color='green',
            linestyle='--', linewidth=1)

# Initialize the block and the arrow
        line1, = ax.plot([], [], color='blue', marker='o', markersize=12)
        pfeil   = ax.quiver([], [], [], [], color='green', scale=35,
            width=0.004)
        return line1, pfeil, msg

    line1, pfeil, msg = initialize_plot()

# Function to update the plot for each animation frame
    def update(frame):
        message = (f'Running time {times[frame]:.2f} sec \n' +
            f'The red line is the initial position, the green line is ' +
            f'the final position \n' +
            f'The green arrow is the force acting on the block \n' +
            f'{msg}' )
        ax.set_title(message, fontsize=12)

        line1.set_data([P0_x[frame]], [P0_y[frame]])
        pfeil.set_offsets([P0_x[frame], P0_y[frame]])
        pfeil.set_UVC(Pfeil_x[frame], Pfeil_y[frame])
        return line1, pfeil

    if video == True:
        animation = FuncAnimation(fig, update, frames=range(len(P0_x)),
            interval=1000*interval_value, blit=True)
    else:
        animation = None
    return animation, update

# %%
# Below the results of **minimized duration** are shown.
selektion = 0
print('message from optimizer:', info_list[selektion]['status_msg'])
print(f'optimal h value is: {solution_list[selektion][-1]:.3f}')
# %%
prob_list[selektion].plot_objective_value()
# %%
# Plot errors in the solution.
prob_list[selektion].plot_constraint_violations(solution_list[selektion])
# %%
# Plot the trajectories of the block.
prob_list[selektion].plot_trajectories(solution_list[selektion])
# %%
# Animate the solution.
fig, ax = plt.subplots(figsize=(8, 8))
anim, _ = drucken(selektion, fig, ax)


# %%
# Now the results of **minimized energy** are shown.
selektion = 1
print('message from optimizer:', info_list[selektion]['status_msg'])
# %%
prob_list[selektion].plot_objective_value()
# %%
# Plot errors in the solution.
prob_list[selektion].plot_constraint_violations(solution_list[selektion])
# %%
# Plot the trajectories of the block.
prob_list[selektion].plot_trajectories(solution_list[selektion])
# %%
# Animate the solution.
fig, ax = plt.subplots(figsize=(8, 8))
anim, _ = drucken(selektion, fig, ax)

# %%
# A frame from the animation.
fig, ax = plt.subplots(figsize=(8, 8))
_, update = drucken(0, fig, ax, video=False)
update(100)
# sphinx_gallery_thumbnail_number = 9
plt.show()