# %%
r"""
Car on a Race Course
====================

Objective
---------
- Show various way to use inequalities for equations of motion.
- unknown input trajectories can vary discontinuously. If, for whatever
  physical reason, this is not desiriable, a way is shown how to make is
  smooth.

Introduction
------------

A car with rear wheel drive and front steering must go from start of finish on
a racecourse in minimal time.
The street is modelled as two function :math:`y = f(x)` and :math:`y = g(x)` ,
with :math:`f(x) > g(x), \forall{x}`.
The car is modelled as three rods, one for the body, one for the rear axle and
one for the front axle.
The ensure the car stays on the road, *number* points :math:`P_i(x_i | y_i)`
distributed evenly along the body of the car are introduced. The inequalities
:math:`f(x_i) > y_i > g(x_i)` ensure that the car stays on the road.

The acceleration at the front and at the rear end of the car are bound to avoid
sliding off the road. Again, enforced by using inequalities on the respective
equations of motion.

The should be aligned so its front is to the right of its back.
This is enforced by the inequality :math:`x_f - x_b \geq 0`, where
:math:`x_f` and :math:`x_b` are the x-coordinates of the front and back of the
car.

There is no speed allowed perpendicular to the wheels, realized by two speed
constraints.

The force accelerating the car, :math:`F_b`, should
change continuously. This is done by making :math:`F_b, F_{b_{dt}}`  state
variables, and adding :math:`\begin{pmatrix} \dfrac{d}{dt}F_b = F_{b_{dt}} \\
m_h \dfrac{d}{dt}F_{b_{dt}} = F_h \end{pmatrix}` to the equations of motion.
Selecting :math:`m_h` and bounding :math:`F_h` appropriately one can make
:math:`F_b` as smooth as desired.


**States**

- :math:`x, y` : coordinates of front of the car
- :math:`u_x, u_y` : velocity of the front of the car
- :math:`q_0` : angle of the body of the car
- :math:`u_0` : angular velocity of the body of the car
- :math:`q_f` : angle of the front axis relative to the body
- :math:`u_f` : angular velocity of the front axis relative to the body
- :math:`F_b` : driving force at the rear axis
- :math:`F_{b_{dt}}` : time derivative of the driving force at the rear axis


**Specifieds**

- :math:`T_f` : torque at the front axis, that is steering torque
- :math:`F_h` : driving the equation of motion for :math:`F_b`

**Known Parameters**

- :math:`l` : length of the car
- :math:`m_0` : mass of the body of the car
- :math:`m_b` : mass of the rear axis
- :math:`m_f` : mass of the front axis
- :math:`m_h` : mass of the eom of the driving force
- :math:`iZZ_0` : moment of inertia of the body of the car
- :math:`iZZ_b` : moment of inertia of the rear axis
- :math:`iZZ_f` : moment of inertia of the front axis
- :math:`\textrm{reibung}` : friction coefficient between the car and the
  street
- :math:`a, b, c, d` : parameters of the street


**Unknown Parameters**

- :math:`h` : variable time interval, to be minimized

"""

import os
import sympy.physics.mechanics as me
import numpy as np
import sympy as sm
from scipy.interpolate import CubicSpline

from opty.direct_collocation import Problem
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %%
# Kane's Equations of Motion
# --------------------------

N, A0, Ab, Af = sm.symbols('N A0 Ab Af', cls=me.ReferenceFrame)
t = me.dynamicsymbols._t
O, Pb, Dmc, Pf = sm.symbols('O Pb Dmc Pf', cls=me.Point)
O.set_vel(N, 0)

q0, qf = me.dynamicsymbols('q_0 q_f')
u0, uf = me.dynamicsymbols('u_0 u_f')
x, y = me.dynamicsymbols('x y')
ux, uy = me.dynamicsymbols('u_x u_y')
Tf, Fb, Fbdt = me.dynamicsymbols('T_f F_b F_bdt')
Fh = me.dynamicsymbols('F_h')

reibung = sm.symbols('reibung')

l, m0, mb, mf, iZZ0, iZZb, iZZf = sm.symbols('l m0 mb mf iZZ0, iZZb, iZZf')
mh = sm.symbols('mh')

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
                me.dot(rot1 - rot, N.z)])
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
fr, frstar = KM.kanes_equations(BODY, FL)
eom = fr + frstar
eom = kd.col_join(eom)
eom = eom.col_join(speed_constr)

# %%
# Define the street and the points on the body of the car.
#
XX, a, b, c, d = sm.symbols('XX a b c d')


def street(XX, a, b, c):
    return a*sm.sin(b*XX) + c


# *number* is the number of points spread evenly along the car body. These
# points must stay on the road at all times.
number = 4

park1y = Pf.pos_from(O).dot(N.y)
park2y = Pb.pos_from(O).dot(N.y)
park1x = Pf.pos_from(O).dot(N.x)
park2x = Pb.pos_from(O).dot(N.x)

delta_x = np.linspace(park1x, park2x, number)
delta_y = np.linspace(park1y, park2y, number)

# %%
# Ensure the *number* points along the car are between the lower and the upper
# limits of the road.
delta_p_u = [delta_y[i] - street(delta_x[i], a, b, c) for i in range(number)]
delta_p_l = [-delta_y[i] + street(delta_x[i], a, b, c+d)
             for i in range(number)]

eom_add = sm.Matrix([
    *[delta_p_u[i] for i in range(number)],
    *[delta_p_l[i] for i in range(number)],
])
eom = eom.col_join(eom_add)

# %%
# Acceleration constraints, to avoid sliding off the race course.
accel_front = Pf.acc(N).dot(A0.x)
accel_back = Pb.acc(N).dot(A0.x)
beschleunigung = sm.Matrix([accel_front, accel_back])

eom = eom.col_join(beschleunigung)

# %%
# Add the 'equation of motion' for the driving force of the car, so the driving
# force is smooth.
acc_delay = sm.Matrix([Fb.diff(t) - Fbdt, mh * Fbdt.diff(t) - Fh])
eom = eom.col_join(acc_delay)

# %%
# Front of the car must be to the right of its back.
front_x = Pf.pos_from(O).dot(N.x)
back_x = Pb.pos_from(O).dot(N.x)
eom = eom.col_join(
    sm.Matrix([front_x - back_x]))

print(f'eom too large to print out. Its shape is {eom.shape} and it has ' +
      f'{sm.count_ops(eom)} operations')

# %%
# Set up the Optimization Problem and Solve it
# --------------------------------------------
h = sm.symbols('h')
state_symbols = ([x, y, q0, qf, ux, uy, u0, uf] + [Fb, Fbdt])
constant_symbols = (l, m0, mb, mf, iZZ0, iZZb, iZZf, reibung, a, b, c, d)


num_nodes = 601
t0 = 0.0
interval_value = h
tf = h * (num_nodes - 1)

# %%
# Specify the known system parameters.
par_map = {}
par_map[m0] = 1.0
par_map[mb] = 0.5
par_map[mf] = 0.5
par_map[mh] = 0.20
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
# Set up constraints and bounds.
instance_constraints = (
        x.func(t0) + 10.0,
        ux.func(t0),
        uy.func(t0),
        u0.func(t0),
        uf.func(t0),
        Fb.func(t0),
        Fbdt.func(t0),
        x.func(tf) - 10.0,
        ux.func(tf),
        uy.func(tf),
)

limit = 20.0
limit1 = 15.0
limit2 = 30.0
delta = np.pi/4.0
bounds = {
        Fh: (-limit2, limit2),
        Fb: (-limit, limit),
        Tf: (-limit, limit),
        # restrict the steering angle to avoid locking
        qf: (-np.pi/2. + delta, np.pi/2. - delta),
        x: (-15, 15),
        y: (0.0, 25),
        h: (0.0, 0.5),
}

# %%
# Define the bounds on the equations of motion.
eom_bounds1 = {
    16: (-limit1, limit1),  # acc_front
    17: (-limit1, limit1),  # acc_back
    20: (0.0, np.inf)
}

eom_bounds2 = {8 + i: (0.0, np.inf) for i in range(2*number)}

eom_bounds = {**eom_bounds1, **eom_bounds2}

# %%
# Create the Problem instance. If a solution is available, use backend='numpy',
# as it is much faster setting up the Problem. If not, use backend='cython',
# (The default value) as it is faster to solve the problem.
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
        eom_bounds=eom_bounds,
        time_symbol=t,
        backend='numpy',
)

# %%
# If there is an existing solution, take it. Else calculate it, using a
# reasonable initial guess and iterate from a smoother road to a more curvy
# one..
fname = f'car_on_racecourse_smooth_{num_nodes}_nodes_solution.csv'
if os.path.exists(fname):
    # Take the existing solution.
    solution = np.loadtxt(fname)
else:
    # Calculate the solution.
    np.random.seed(123)
    initial_guess = np.random.randn(prob.num_free) * 0.001
    x_list = np.linspace(-10.0, 10.0, num_nodes)
    y_list = [6.0 for _ in range(num_nodes)]
    initial_guess[0:num_nodes] = x_list
    initial_guess[num_nodes:2*num_nodes] = y_list

    prob.add_option('max_iter', 5000)
    for i in range(7):
        # It seems necessary to iterate for a simpler problem, a less
        # curvy street to a more curvy one. If only the values of
        # par_map are changed, no need to repeat setting up Problem again.
        par_map[a] = 0.2915555555557 * i
        par_map[b] = 0.041666666666667 * i
        if i == 6:
            par_map[a] = 3.5
            par_map[b] = 0.5
        solution, info = prob.solve(initial_guess)
        initial_guess = solution
        print(f'{i+1} - th iteration')
        print('message from optimizer:', info['status_msg'])
        print('Iterations needed', len(prob.obj_value))
        print(f"objective value {info['obj_val']:.3e} \n")

    _ = prob.plot_objective_value()

# %%
_ = prob.plot_constraint_violations(solution, subplots=True, show_bounds=True)
# %%
# Plot trajectories.
_ = prob.plot_trajectories(solution, show_bounds=True)

# %%
# Aminate the Car
# ---------------
fps = 5
street_lam = sm.lambdify((x, a, b, c), street(x, a, b, c))

state_vals, input_vals, _, h_vals = prob.parse_free(solution)
tf = solution[-1] * (num_nodes - 1)
t_arr = np.linspace(t0, tf, num_nodes)
state_sol = CubicSpline(t_arr, state_vals.T)
input_sol = CubicSpline(t_arr, input_vals.T)

# create additional points for the axles
Pbl, Pbr, Pfl, Pfr = sm.symbols('Pbl Pbr Pfl Pfr', cls=me.Point)

# end points of the force, length of the axles
Fbq = me.Point('Fbq')
la = sm.symbols('la')
fb, tq = sm.symbols('f_b, t_q')

Pbl.set_pos(Pb, -la/2 * Ab.x)
Pbr.set_pos(Pb, la/2 * Ab.x)
Pfl.set_pos(Pf, -la/2 * Af.x)
Pfr.set_pos(Pf, la/2 * Af.x)

Fbq.set_pos(Pb, Fb * Ab.y)

coordinates = Pb.pos_from(O).to_matrix(N)
for point in (Dmc, Pf, Pbl, Pbr, Pfl, Pfr, Fbq):
    coordinates = coordinates.row_join(point.pos_from(O).to_matrix(N))

pL, pL_vals = zip(*par_map.items())
la1 = par_map[l] / 4.                      # length of an axle
la2 = la1/1.8
coords_lam = sm.lambdify((*state_symbols, tq, *pL, la), coordinates,
                         cse=True)


def init():
    xmin, xmax = -13, 13.
    ymin, ymax = -0.75, 12.5

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
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
                           par_map[b], par_map[c]+par_map[d]+la2),
            color='black')
    ax.vlines(-10.0, street_lam(-10.0, par_map[a], par_map[b],
              par_map[c]-la2), street_lam(-10, par_map[a], par_map[b],
              par_map[c]+par_map[d]+la2), color='red', linestyle='--')
    ax.vlines(10.0, street_lam(10, par_map[a], par_map[b], par_map[c]-la2),
              street_lam(10, par_map[a], par_map[b], par_map[c]+par_map[d]
              + la2), color='green', linestyle='--')

    line1, = ax.plot([], [], color='orange', lw=2)
    line2, = ax.plot([], [], color='red', lw=2)
    line3, = ax.plot([], [], color='magenta', lw=2)
    line4 = ax.quiver([], [], [], [], color='green', scale=35,
                      width=0.004, headwidth=8)

    return fig, ax, line1, line2, line3, line4


# Function to update the plot for each animation frame
fig, ax, line1, line2, line3, line4 = init()


def update(t):
    message = ((f'Running time {t:.2f} sec \n The rear axle is red, the '
               'front axle is magenta \n The driving/breaking force is green'))
    ax.set_title(message, fontsize=12)

    coords = coords_lam(*state_sol(t), input_sol(t), *pL_vals, la1)

    #   Pb, Dmc, Pf, Pbl, Pbr, Pfl, Pfr, Fbq
    line1.set_data([coords[0, 0], coords[0, 2]], [coords[1, 0], coords[1, 2]])
    line2.set_data([coords[0, 3], coords[0, 4]], [coords[1, 3], coords[1, 4]])
    line3.set_data([coords[0, 5], coords[0, 6]], [coords[1, 5], coords[1, 6]])

    line4.set_offsets([coords[0, 0], coords[1, 0]])
    line4.set_UVC(coords[0, 7] - coords[0, 0], coords[1, 7] - coords[1, 0])


frames = np.linspace(t0, tf, int(fps * (tf - t0)))
animation = FuncAnimation(fig, update, frames=frames, interval=1000/fps)
# %%

# sphinx_gallery_thumbnail_number = 4

fig, ax, line1, line2, line3, line4 = init()
update(4.7)

plt.show()
