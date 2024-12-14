"""
Car on a Race Course
====================

A car with rear wheel drive and fron steering must go from start of finish on
a racecourse in minimal time.
The street is modelled as two function :math:`y = f(x)` and :math:`y = g(x)` ,
with :math:`f(x) > g(x), \\forall{x}`.

The car is modelled as three rods, one for the body, one for the rear axle and
one for the front axle.

The ensure the car stays on the road, I look at *number* points
:math:`P_i(x_i | y_i)`
distributed evenly along the body of the car and force
:math:`f(x_i) > y_i > g(x_i)` by introducing additional state variables
:math:`py_{upper}` and :math:`py_{lower}` and constraints
:math:`-py_{upper_i} + f(x_i) > 0` and :math:`py_{lower_i} - g(x_i) > 0`.

Also I limit the acceleration at the front and at the rear end of the car to
avoid sliding off the road.

The is no speed allowed perpendicular to the wheels, realized by two speed
constraints.

**States**

- :math:`x, y` : coordinates of fron of the car
- :math:`u_x, u_y` : velocity of the front of the car
- :math:`q_0` : angle of the body of the car
- :math:`u_0` : angular velocity of the body of the car
- :math:`q_f` : angle of the front axis relative to the body
- :math:`u_f` : angular velocity of the front axis relative to the body
- :math:`py_{upper_i}` : distance of point i of the car to :math:`f(x_i)`
- :math:`py_{lower_i}` : distance of point i of the car to :math:`g(x_i)`
- :math:`acc_f` : acceleration of the front of the car
- :math:`acc_b` : acceleration of the back of the car

**Specifieds**

- :math:`T_f` : torque at the front axis, that is steering torque
- :math:`F_b` : force at the rear axis, that is driving force

**Known Parameters**

- :math:`l` : length of the car
- :math:`m_0` : mass of the body of the car
- :math:`m_b` : mass of the rear axis
- :math:`m_f` : mass of the front axis
- :math:`iZZ_0` : moment of inertia of the body of the car
- :math:`iZZ_b` : moment of inertia of the rear axis
- :math:`iZZ_f` : moment of inertia of the front axis
- :math:`reibung` : friction coefficient between the car and the street
- :math:`a, b, c, d` : parameters of the street

**Unknown Parameters**

- :math:`h` : time step

"""

import sympy.physics.mechanics as me
import time
import numpy as np
import sympy as sm
from scipy.interpolate import CubicSpline

from opty.direct_collocation import Problem
from opty.utils import parse_free, create_objective_function
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import HTML
matplotlib.rcParams['animation.embed_limit'] = 2**128
from matplotlib.animation import FuncAnimation

# %%
# Kane's Equations of Motion
# --------------------------

# %%
start = time.time()

N, A0, Ab, Af = sm.symbols('N A0 Ab Af', cls= me.ReferenceFrame)
t = me.dynamicsymbols._t
O, Pb, Dmc, Pf = sm.symbols('O Pb Dmc Pf', cls= me.Point)
O.set_vel(N, 0)

q0, qf = me.dynamicsymbols('q_0 q_f')
u0, uf = me.dynamicsymbols('u_0 u_f')
x, y = me.dynamicsymbols('x y')
ux, uy = me.dynamicsymbols('u_x u_y')
Tf, Fb = me.dynamicsymbols('T_f F_b')

reibung = sm.symbols('reibung')

l, m0, mb, mf, iZZ0, iZZb, iZZf = sm.symbols('l m0 mb mf iZZ0, iZZb, iZZf')

A0.orient_axis(N, q0, N.z)
A0.set_ang_vel(N, u0 * N.z)

Ab.orient_axis(A0, 0, N.z)

Af.orient_axis(A0, qf, N.z)
rot = Af.ang_vel_in(N)
Af.set_ang_vel(N, uf * N.z)
rot1 = Af.ang_vel_in(N)

Pf.set_pos(O, x * N.x + y * N.y)
Pf.set_vel(N, ux * N.x + uy * N.y)

Pb.set_pos(Pf, -l * A0.y)
Pb.v2pt_theory(Pf, N, A0)

Dmc.set_pos(Pf, -l/2 * A0.y)
Dmc.v2pt_theory(Pf, N, A0)
prevent_print = 1

# %%
# No speed perpendicular to the wheels.
vel1 = me.dot(Pb.vel(N), Ab.x) - 0
vel2 = me.dot(Pf.vel(N), Af.x) - 0

I0 = me.inertia(A0, 0, 0, iZZ0)
body0 = me.RigidBody('body0', Dmc, A0, m0, (I0, Dmc))
Ib = me.inertia(Ab, 0, 0, iZZb)
bodyb = me.RigidBody('bodyb', Pb, Ab, mb, (Ib, Pb))
If = me.inertia(Af, 0, 0, iZZf)
bodyf = me.RigidBody('bodyf', Pf, Af, mf, (If, Pf))
BODY = [body0, bodyb, bodyf]

FL = [(Pb, Fb * Ab.y), (Af, Tf * N.z), (Dmc, -reibung * Dmc.vel(N))]

kd = sm.Matrix([ux - x.diff(t), uy - y.diff(t), u0 - q0.diff(t),
                me.dot(rot1- rot, N.z)])
speed_constr = sm.Matrix([vel1, vel2])

q_ind = [x, y, q0, qf]
u_ind = [u0, uf]
u_dep = [ux, uy]

KM = me.KanesMethod(
                    N, q_ind=q_ind, u_ind=u_ind,
                    kd_eqs=kd,
                    u_dependent=u_dep,
                    velocity_constraints=speed_constr,
                    )
(fr, frstar) = KM.kanes_equations(BODY, FL)
eom = fr + frstar
eom = kd.col_join(eom)
eom = eom.col_join(speed_constr)

# %%
# Constraints to keep the car on the road.
#

# %%
XX, a, b, c, d = sm.symbols('XX a b c d')
def street(XX, a, b, c):
    return a*sm.sin(b*XX) + c

number = 4
py_upper = me.dynamicsymbols(f'py_upper:{number}')
py_lower = me.dynamicsymbols(f'py_lower:{number}')
acc_f, acc_b = me.dynamicsymbols('acc_f acc_b')

park1y = Pf.pos_from(O).dot(N.y)
park2y = Pb.pos_from(O).dot(N.y)
park1x = Pf.pos_from(O).dot(N.x)
park2x = Pb.pos_from(O).dot(N.x)

delta_x = np.linspace(park1x, park2x, number)
delta_y = np.linspace(park1y, park2y, number)

delta_p_u = [delta_y[i] - street(delta_x[i], a, b, c) for i in range(number)]
delta_p_l = [-delta_y[i] + street(delta_x[i], a, b, c+d) for i in range(number)]

eom_add = sm.Matrix([
    *[-py_upper[i] + delta_p_u[i] for i in range(number)],
    *[-py_lower[i] + delta_p_l[i] for i in range(number)],
])

eom = eom.col_join(eom_add)

# %%
# Acceleration constraints.
accel_front = Pf.acc(N).magnitude()
accel_back = Pb.acc(N).magnitude()
beschleunigung = sm.Matrix([-acc_f + accel_front, -acc_b + accel_back])

eom = eom.col_join(beschleunigung)

print(f'eom too large to print out. Its shape is {eom.shape} and it has ' +
      f'{sm.count_ops(eom)} operations')

# %%
# Set up the Optimization Problem and Solve it
# --------------------------------------------
h = sm.symbols('h')
state_symbols = ([x, y, q0, qf, ux, uy, u0, uf] + py_upper + py_lower
            + [acc_f, acc_b])
#laenge = len(state_symbols)
constant_symbols = (l, m0, mb, mf, iZZ0, iZZb, iZZf, reibung, a, b, c, d)
specified_symbols = (Fb, Tf)
unknown_symbols = ()

num_nodes = 301
t0    = 0.0
interval_value = h
tf = h * (num_nodes - 1)

# %%
# Specify the known system parameters.
par_map = {}
par_map[m0] = 1.0
par_map[mb] = 0.5
par_map[mf] = 0.5
par_map[iZZ0] = 1.
par_map[iZZb] = 0.5
par_map[iZZf] = 0.5
par_map[l] = 3.0
par_map[reibung] = 0.5
# below for the shape of the street
par_map[a] = 3.5
par_map[b] = 0.5
par_map[c] = 4.0
par_map[d] = 3.5

# %%
# Define the objective function and its gradient.
def obj(free):
    return free[-1]

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad

# %%
# Set up the constraints and the bounds.
instance_constraints = (
        x.func(t0) + 10.0,
        ux.func(t0),
        uy.func(t0),
        u0.func(t0),
        uf.func(t0),

        x.func(tf) - 10.0,
        ux.func(tf),
        uy.func(tf),
)

grenze = 20.0
grenze1 = 5.0
delta = np.pi/4.0
bounds1 = {
        Fb: (-grenze, grenze),
        Tf: (-grenze, grenze),
        # restrict the steering angle to avoid locking
        qf: (-np.pi/2. + delta, np.pi/2. - delta),
        x: (-15, 15),
        y: (0.0, 25),
        h: (0.0, 0.5),
        acc_f: (-grenze1, grenze1),
        acc_b: (-grenze1, grenze1),
}
bounds2 = {py_upper[i]: (0, 10.0) for i in range(number)}
bounds3 = {py_lower[i]: (0.0, 10.0) for i in range(number)}
bounds = {**bounds1, **bounds2, **bounds3}

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
)

# %%
# For the initial guess I use the result of some previous run, to expedite
# execution.
initial_guess = np.ones(prob.num_free)
initial_guess = np.load('car_on_racecourse_solution.npy')

prob.add_option('max_iter', 5000)
for i in range(1):
# Find the optimal solution.
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(f'{i+1} - th iteration')
    print('message from optimizer:', info['status_msg'])
    print('Iterations needed',len(prob.obj_value))
    print(f"objective value {info['obj_val']:.3e} \n")

prob.plot_objective_value()

# %%
# Plot the constraint violations.

# %%
prob.plot_constraint_violations(solution)

# %%
# Plot generalized coordinates / speeds and forces / torques
prob.plot_trajectories(solution)

# %%
# Aminate the Car
# ---------------
fps = 10
street_lam = sm.lambdify((x, a, b, c), street(x, a, b, c))

tf = solution[-1] * (num_nodes - 1)
def add_point_to_data(line, x, y):
# to trace the path of the point. Copied from Timo.
    old_x, old_y = line.get_data()
    line.set_data(np.append(old_x, x), np.append(old_y, y))


state_vals, input_vals, _ = parse_free(solution, len(state_symbols),
                len(specified_symbols), num_nodes)
t_arr = np.linspace(t0, tf, num_nodes)
state_sol = CubicSpline(t_arr, state_vals.T)
input_sol = CubicSpline(t_arr, input_vals.T)

# create additional points for the axles
Pbl, Pbr, Pfl, Pfr = sm.symbols('Pbl Pbr Pfl Pfr', cls= me.Point)

# end points of the force, length of the axles
Fbq = me.Point('Fbq')
la = sm.symbols('la')
fb, tq = sm.symbols('f_b, t_q')

Pbl.set_pos(Pb, -la/2 * Ab.x)
Pbr.set_pos(Pb, la/2 * Ab.x)
Pfl.set_pos(Pf, -la/2 * Af.x)
Pfr.set_pos(Pf, la/2 * Af.x)

Fbq.set_pos(Pb, fb * Ab.y)

coordinates = Pb.pos_from(O).to_matrix(N)
for point in (Dmc, Pf, Pbl, Pbr, Pfl, Pfr, Fbq):
    coordinates = coordinates.row_join(point.pos_from(O).to_matrix(N))

pL, pL_vals = zip(*par_map.items())
la1 = par_map[l] / 4.                      # length of an axle
la2 = la1/1.8
coords_lam = sm.lambdify((*state_symbols, fb, tq, *pL, la), coordinates,
        cse=True)

def init():
        xmin, xmax = -13, 13.
        ymin, ymax = -0.75, 12.5

        fig = plt.figure(figsize=(8, 8))
        ax  = fig.add_subplot(111)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.grid()

        XX = np.linspace(xmin, xmax,  200)
        ax.fill_between(XX, ymin, street_lam(XX, par_map[a], par_map[b],
                par_map[c]-la2), color='grey', alpha=0.25)
        ax.fill_between(XX, street_lam(XX, par_map[a], par_map[b],
                par_map[c]+par_map[d]+la2), ymax, color='grey', alpha=0.25)

        ax.plot(XX, street_lam(XX, par_map[a], par_map[b], par_map[c]-la2),
                color='black')
        ax.plot(XX, street_lam(XX, par_map[a],
                par_map[b], par_map[c]+par_map[d]+la2), color='black')
        ax.vlines(-10.0, street_lam(-10.0, par_map[a], par_map[b],
                par_map[c]-la2), street_lam(-10, par_map[a], par_map[b],
                par_map[c]+par_map[d]+la2), color='red', linestyle='--')
        ax.vlines(10.0, street_lam(10, par_map[a], par_map[b], par_map[c]-la2),
                street_lam(10, par_map[a], par_map[b], par_map[c]+par_map[d]
                +la2), color='green', linestyle='--')

        line1, = ax.plot([], [], color='orange', lw=2)
        line2, = ax.plot([], [], color='red', lw=2)
        line3, = ax.plot([], [], color='magenta', lw=2)
        line4  = ax.quiver([], [], [], [], color='green', scale=35, width=0.004,
                headwidth=8)

        return fig, ax, line1, line2, line3, line4

# Function to update the plot for each animation frame
fig, ax, line1, line2, line3, line4 = init()

def update(t):
    message = (f'running time {t:.2f} sec \n The rear axle is red, the ' +
               f'front axle is magenta \n The driving/breaking force is green')
    ax.set_title(message, fontsize=12)

    coords = coords_lam(*state_sol(t), *input_sol(t), *pL_vals, la1)

    #   Pb, Dmc, Pf, Pbl, Pbr, Pfl, Pfr, Fbq
    line1.set_data([coords[0, 0], coords[0, 2]], [coords[1, 0], coords[1, 2]])
    line2.set_data([coords[0, 3], coords[0, 4]], [coords[1, 3], coords[1, 4]])
    line3.set_data([coords[0, 5], coords[0, 6]], [coords[1, 5], coords[1, 6]])

    line4.set_offsets([coords[0, 0], coords[1, 0]])
    line4.set_UVC(coords[0, 7] - coords[0, 0] , coords[1, 7] - coords[1, 0])

frames = np.linspace(t0, tf, int(fps * (tf - t0)))
animation = FuncAnimation(fig, update, frames=frames, interval=1000/fps)

# %%

# sphinx_gallery_thumbnail_number = 5

fig, ax, line1, line2, line3, line4 = init()
update(4.7)

plt.show()

