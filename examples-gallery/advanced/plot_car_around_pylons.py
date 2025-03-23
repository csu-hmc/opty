# %%
r"""
Car around Pylons
=================

Objective
---------
- Show a way to circumvent the current inability of *opty* to handle  instance
  constraints at times that are not specified in advance but selected optimally
  by *opty* itself.

Introduction
------------

A car with rear wheel drive and front steering must move close to two points
and return to its starting point. It can only move forward. The car is modeled
as a rigid body with three bodies. The car is driven by a force at the rear
axis and steered by a torque at the front axis. The acceleration is limited at
the front and at the rear end of the car to avoid sliding off the road.

There is no speed allowed perpendicular to the wheels, realized by two velocity
constraints. (The trajectories show, that the car is at the limit often, as one
would expect if, as it is the case here, duration has to be minimized)

Method to Achieve the Objective
-------------------------------

Make the car to come 'close' to two points, but without specifying the time
when it should be there. *opty* should find the best time for the car to be
there. Presently intermediate points must be specified as
:math:`t_\textrm{intermediate} = \textrm{integer} \cdot
\textrm{interval}_\textrm{value}`, with :math:`0 < \textrm{integer} <
\textrm{num}_\textrm{nodes}` fixed, so:

- Set the two points as :math:`(x_{b_1}, y_{b_1})` and :math:`(x_{b_2},
  y_{b_2})` and an allowable 'radius' called *epsilon* around these points.
- Create a differentiable function :math:`\textrm{hump}(x, a, b, g_r)` such
  that it is one for :math:`a \leq x \leq b` and zero otherwise. :math:`g_r >
  0` is a parameter that determines how 'sharp' the transition is, the larger
  the sharper.
- In order to know at the end of the run whether the car came close to the
  points during its course, integrate the hump function over time. These are
  the variables :math:`\textrm{punkt}_1, \textrm{punkt}_2` with
  :math:`\textrm{punkt}_1 = \int_{t0}^{tf} \textrm{hump}(...) \, dt > 0` if the car
  came close to the point, = 0 otherwise. Same for :math:`\textrm{punkt}_2`.
- The exact values of :math:`\textrm{punkt}_1, \textrm{punkt}_2` are not known
  and should simple be positive 'enough', include two additional state variables
  :math:`\textrm{dist}_1, \textrm{dist}_2` and specified variables :math:`h_1, h_2`.
- By setting :math:`\textrm{dist}_1 = \textrm{punkt}_1 \cdot h_1` and
  :math:`\textrm{dist}_2 = \textrm{punkt}_2 \cdot h_2` and bounding
  :math:`h_1, h_2 \in (1, \textrm{value})`, and setting :math:`\textrm{dist}_1(t_f) = 1`,
  one can ensure that :math:`\textrm{punkt}_1 > \dfrac{1}{\textrm{value}}`
  and :math:`\textrm{punkt}_2 > \dfrac{1}{\textrm{value}}`.

**States**

- :math:`x, y` : coordinates of front of the car
- :math:`u_x, u_y` : velocity of the front of the car
- :math:`q_0` : angle of the body of the car
- :math:`u_0` : angular velocity of the body of the car
- :math:`q_f` : angle of the front axis relative to the body
- :math:`u_f` : angular velocity of the front axis relative to the body
- :math:`acc_f` : acceleration of the front of the car
- :math:`acc_b` : acceleration of the back of the car
- :math:`\textrm{forward}` : variable to ensure the car can only move forward
- :math:`\textrm{punkt}_1, \textrm{punkt}_2` : variables to ensure the car
  comes close to the points
- :math:`\textrm{punkt}_{\textrm{dt}_1}, \textrm{punkt}_{\textrm{dt}_2}` : time
  derivatives of the above
- :math:`\textrm{dist}_1, \textrm{dist}_2` : variables to ensure the car comes
  close to the points

**Specifieds**

- :math:`T_f` : torque at the front axis, that is steering torque
- :math:`F_b` : force at the rear axis, that is driving force
- :math:`h_1, h_2` : variables to ensure the car comes close to the points

**Known Parameters**

- :math:`l` : length of the car
- :math:`m_0` : mass of the body of the car
- :math:`m_b` : mass of the rear axis
- :math:`m_f` : mass of the front axis
- :math:`iZZ_0` : moment of inertia of the body of the car
- :math:`iZZ_b` : moment of inertia of the rear axis
- :math:`iZZ_f` : moment of inertia of the front axis
- :math:`\textrm{reibung}` : friction coefficient between the car and the street
- :math:`x_{b_1}` : x coordinate of pylon 1
- :math:`y_{b_1}`: y coordinate of pylon 1
- :math:`x_{b_2}` : x coordinate of pylon 2
- :math:`y_{b_2}`: y coordinate of pylon 2

**Unknown Parameters**

- :math:`h` : time step

"""
import os
import sympy.physics.mechanics as me
import numpy as np
import sympy as sm
from scipy.interpolate import CubicSpline

from opty.direct_collocation import Problem
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# %%
# Equations of Motion
# -------------------
N, A0, Ab, Af = sm.symbols('N A0 Ab Af', cls=me.ReferenceFrame)
t = me.dynamicsymbols._t
O, Pb, Dmc, Pf = sm.symbols('O Pb Dmc Pf', cls=me.Point)
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

# %%
# Dynamic equations of motion, Kane's method.
I0 = me.inertia(A0, 0, 0, iZZ0)
body0 = me.RigidBody('body0', Dmc, A0, m0, (I0, Dmc))
Ib = me.inertia(Ab, 0, 0, iZZb)
bodyb = me.RigidBody('bodyb', Pb, Ab, mb, (Ib, Pb))
If = me.inertia(Af, 0, 0, iZZf)
bodyf = me.RigidBody('bodyf', Pf, Af, mf, (If, Pf))
bodies = [body0, bodyb, bodyf]

forces = [(Pb, Fb * Ab.y), (Af, Tf * N.z), (Dmc, -reibung * Dmc.vel(N))]

kd = sm.Matrix([ux - x.diff(t), uy - y.diff(t), u0 - q0.diff(t),
                me.dot(rot1- rot, N.z)])
speed_constr = sm.Matrix([vel1, vel2])

q_ind = [x, y, q0, qf]
u_ind = [u0, uf]
u_dep = [ux, uy]

KM = me.KanesMethod(
    N,
    q_ind=q_ind,
    u_ind=u_ind,
    kd_eqs=kd,
    u_dependent=u_dep,
    velocity_constraints=speed_constr,
)
(fr, frstar) = KM.kanes_equations(bodies, forces)
eom = fr + frstar
eom = kd.col_join(eom)
eom = eom.col_join(speed_constr)


# %%
# Constraints so the car approaches the points :math:`(x_{b_1}, y_{b_1})`, and
# :math:`(x_{b_2}, y_{b_2})` at whatever times opty chooses, explanation above.
# Also here it is enforced that it can only move forward in that the state
# variable *forward* is bound to be positive. If gr is large, say, gr = 75 the
# optimazation will not work it may become too 'non-differentiable' for *opty*.
def hump(x, a, b, gr):
    # approx one for x in [a, b]
    # approx zero otherwise
    # the higher gr the closer the approximation
    return 1.0 - (1/(1 + sm.exp(gr*(x - a))) + 1/(1 + sm.exp(-gr*(x - b))))


forward = me.dynamicsymbols('forward')
punkt1, punktdt1 = me.dynamicsymbols('punkt1 punktdt1')
punkt2, punktdt2 = me.dynamicsymbols('punkt2 punktdt2')
dist1, dist2 = me.dynamicsymbols('dist1 dist2')
h1, h2 = me.dynamicsymbols('h1 h2')

xb1, yb1, xb2, yb2= sm.symbols('xb yb xb2 yb2')
epsilon = sm.symbols('epsilon')

treffer1 = (hump(x, xb1-epsilon, xb1+epsilon, 5)*hump(y, yb1-epsilon,
                                                      yb1+epsilon, 5))
treffer2 = (hump(x, xb2-epsilon, xb2+epsilon, 5)*hump(y, yb2-epsilon,
                                                      yb2+epsilon, 5))

eom_add = sm.Matrix([
    -forward + Pb.vel(N).dot(Ab.y),
    -punkt1.diff(t) + punktdt1,
    -punktdt1 + treffer1,
    -punkt2.diff(t) + punktdt2,
    -punktdt2 + treffer2,
    -dist1 + punkt1 * h1,
    -dist2 + punkt2 * h2,
])

eom = eom.col_join(eom_add)

# %%
# Acceleration constraints, so the car does not slide off the road.
acc_f, acc_b = me.dynamicsymbols('acc_f acc_b')

accel_front = Pf.acc(N).magnitude()
accel_back = Pb.acc(N).magnitude()
beschleunigung = sm.Matrix([-acc_f + accel_front, -acc_b + accel_back])

eom = eom.col_join(beschleunigung)

print(f'eom too large to print out. Its shape is {eom.shape} and it has '
      f'{sm.count_ops(eom)} operations')

# %%
# Set up the Optimization Problem and Solve It
# --------------------------------------------
h = sm.symbols('h')
state_symbols = ([x, y, q0, qf, ux, uy, u0, uf] + [acc_f, acc_b, forward] +
                 [punkt1, punktdt1, punkt2, punktdt2] + [dist1, dist2])
constant_symbols = (l, m0, mb, mf, iZZ0, iZZb, iZZf, reibung)
specified_symbols = (Fb, Tf, h1, h2)

num_nodes = 401
t0 = 0.0
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
par_map[xb1] = 10.0
par_map[yb1] = 15.0
par_map[xb2] = -5.0
par_map[yb2] = 10.0
par_map[epsilon] = 0.5


# %%
# Define the objective function and its gradient.
# The time needed is to be minimized.
def obj(free):
    return free[-1]


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad

# %%
# Set up the constraints, the bounds and the Problem.
instance_constraints = (
    x.func(t0),
    y.func(t0),
    q0.func(t0),
    ux.func(t0),
    uy.func(t0),
    u0.func(t0),
    uf.func(t0),
    punkt1.func(t0),
    punktdt1.func(t0),
    punkt2.func(t0),
    punktdt2.func(t0),
    dist1.func(t0),
    dist2.func(t0),
    x.func(tf),
    y.func(tf),
    ux.func(tf),
    uy.func(tf),
    dist1.func(tf) - 1.0,
    dist2.func(tf) - 1.0,
)

grenze = 20.0
grenze1 = 5.0
delta = np.pi/4.0
bounds = {
    Fb: (-grenze, grenze),
    Tf: (-grenze, grenze),
    # restrict the steering angle
    qf: (-np.pi/2. + delta - 1.e-5, np.pi/2. - delta + 1.e-5),
    x: (-20, 20),
    y: (-15, 30),
    h: (0.0, 0.5),
    acc_f: (-grenze1, grenze1),
    acc_b: (-grenze1, grenze1),
    forward: (-1.e-5, np.inf),
    h1: (1.0, 5.0),
    h2: (1.0, 5.0),
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
)

# %%
# For the initial guess the result of some previous run is used to expedite
# execution, if available
fname = f'car_around_pylons_{num_nodes}_nodes_solution.csv'
if os.path.exists(fname):
    solution = np.loadtxt(fname)
else:
    prob.add_option('max_iter', 3000)
    initial_guess= np.random.randn(prob.num_free)
    for i in range(5 ):
        solution, info = prob.solve(initial_guess)
        print('message from optimizer:', info['status_msg'])
        print('Iterations needed', len(prob.obj_value))
        print(f"objective value {info['obj_val']:.3e} \n")

        initial_guess = solution
        print(f'{i+1} - th iteration')
np.savetxt(f'car_around_pylons_{num_nodes}_nodes_solution.csv', solution,
            fmt='%.2f')

# %%
# Plot the constraint violations.

# %%
_ = prob.plot_constraint_violations(solution)

# %%
# Plot generalized coordinates / speeds and forces / torques
_ = prob.plot_trajectories(solution)

# %%
# Animate the Car
# ---------------
fps = 10

tf = solution[-1] * (num_nodes - 1)

state_vals, input_vals, _ = prob.parse_free(solution)
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

Fbq.set_pos(Pb, fb * Ab.y)

coordinates = Pb.pos_from(O).to_matrix(N)
for point in (Dmc, Pf, Pbl, Pbr, Pfl, Pfr, Fbq):
    coordinates = coordinates.row_join(point.pos_from(O).to_matrix(N))

pL, pL_vals = zip(*par_map.items())
la1 = par_map[l] / 4.                      # length of an axle

coords_lam = sm.lambdify((*state_symbols, fb, tq, *pL, la), coordinates,
                         cse=True)


def init():
    xmin, xmax = -15, 15.
    ymin, ymax = -5, 25.

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid()
    circle1 = Circle((par_map[xb1], par_map[yb1]), par_map[epsilon],
                     edgecolor='red', facecolor='none', linewidth=1)
    ax.add_patch(circle1)
    circle2 = Circle((par_map[xb2], par_map[yb2]), par_map[epsilon],
                     edgecolor='green', facecolor='none', linewidth=1)
    ax.add_patch(circle2)

    line1, = ax.plot([], [], color='orange', lw=2)
    line2, = ax.plot([], [], color='red', lw=2)
    line3, = ax.plot([], [], color='magenta', lw=2)
    line4 = ax.quiver([], [], [], [], color='green', scale=35, width=0.004,
                      headwidth=8)
    line5, = ax.plot([], [], color='blue', lw=1)

    return fig, ax, line1, line2, line3, line4, line5


fig, ax, line1, line2, line3, line4, line5 = init()

zeiten = np.linspace(t0, tf, int(fps * (tf - t0)))


def update(t):
    message = (f'running time {t:.2f} sec \n The rear axle is red, the '
               f'front axle is magenta \n The driving/breaking force is green')
    ax.set_title(message, fontsize=12)

    coords = coords_lam(*state_sol(t), input_sol(t)[0], input_sol(t)[1],
                        *pL_vals, la1)

    koords = []
    for zeit in zeiten:
        koords.append(coords_lam(*state_sol(zeit), input_sol(zeit)[0],
                                 input_sol(zeit)[1], *pL_vals, la))

    line1.set_data([coords[0, 0], coords[0, 2]], [coords[1, 0], coords[1, 2]])
    line2.set_data([coords[0, 3], coords[0, 4]], [coords[1, 3], coords[1, 4]])
    line3.set_data([coords[0, 5], coords[0, 6]], [coords[1, 5], coords[1, 6]])

    line4.set_offsets([coords[0, 0], coords[1, 0]])
    line4.set_UVC(coords[0, 7] - coords[0, 0], coords[1, 7] - coords[1, 0])

    zaehler1 = np.argwhere(zeiten >= t)[0][0] + 1
    line5.set_data([koords[i][0, 2] for i in range(zaehler1)],
                   [koords[i][1, 2] for i in range(zaehler1)])


frames = np.linspace(t0, tf, int(fps*(tf - t0)))
animation = FuncAnimation(fig, update, frames=frames, interval=1000/fps)

# %%
fig, ax, line1, line2, line3, line4, line5 = init()
update(7.26)

plt.show()

# sphinx_gallery_thumbnail_number = 3
