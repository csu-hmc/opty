 # %%
"""
ball rolling on spinning disc
=============================

A uniform, solid ball with radius r and mass m_b is rolling on a
horizontal spinning disc without slipping.
The disc starts at rest and speeds up like this:
u_3(t) = Omega * (1 - e**(-alpha * t)),
where alpha > 0 is a measure of the acceleration and Omega
is the final rotational speed.
A torque is applied to the ball, and the goal is to get it to the
center of the disc.

An observer, a particle of mass m_o, is attached
to the surface the ball.

**Constants**

- m_b: mass of the ball [kg]
- r : radius of the ball [m]
- m_o : mass of the observer [kg]
- Omega: final rotational speed of the disc
  around the vertical axis [rad/sec]
- alpha: measure of the acceleration of the disc [1/sec]

**States**

- q_1, q_2, q_3: generalized coordinates of the ball
    w.r.t the disc [rad]
- u_1, u_2, u_3: generalized angual velocities of the ball [rad/sec]
- x, y: position of the contact point w.r.t. the disc.
  Dependent states. [m]
- ux, uy: speeds of the contact point w.r.t the disc.
  Dependent states. [m/sec]

**Specifieds**

- t_1, t_2, t_3: Torque applied to the ball [N * m]

**Additional parameters**

- N: inertial frame
- A1: frame fixed to the ball
- A2: frame fixed to the disc

- O: point fixed in N
- CP: contact point of ball with disc
- Dmc: center of the ball
- m_Dmc: position of observer
- h_opty: intervall which opty should minimize.

A video similar to this one, which I saw in something JM published
gave me the idea.
https://www.youtube.com/watch?v=3oM7hX3UUEU
"""
import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
from collections import OrderedDict
from opty.direct_collocation import Problem
from opty.utils import parse_free
from scipy.interpolate import CubicSpline
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams['animation.embed_limit'] = 2**128
from matplotlib.animation import FuncAnimation
from matplotlib import patches

# %% [markdown]
# Initialize the variables.
t = me.dynamicsymbols._t
q1, q2, q3 = me.dynamicsymbols('q1 q2 q3')
u1, u2, u3 = me.dynamicsymbols('u1 u2 u3')
x, y, ux, uy = me.dynamicsymbols('x, y, ux, uy')
t1, t2, t3 = me.dynamicsymbols('t1 t2 t3')

# needed to mimick the time inm opty.
T = sm.symbols('T', cls=sm.Function)
Tdot, Tdotdot = sm.symbols('Tdot Tdotdot')

mb, mo, g, r = sm.symbols('mb, mo, g, r')
Omega, alpha = sm.symbols('Omega, alpha')

N, A1, A2 = sm.symbols('N, A1 A2', cls=me.ReferenceFrame)
O, CP, Dmc, m_Dmc = sm.symbols('O, CP, Dmc, m_Dmc', cls=me.Point)
O.set_vel(N, 0)

# %% [markdown]
# **Determine the holonomic constraints**
#
# If the ball rotates around the x axis by an angle $q_1$ the\
# contact point will be move by -$q_1 \cdot r$ in the A2.y direction\
# If the ball rotates around the y axis by an angle $q_2$ the\
# contact point will be move by $q_2 \cdot r$ in the A2.x direction\
# Hence the coordinate constraints are;
# - $x = r \cdot q_2$
# - $y = -r \cdot q_1$
#
# so the resulting speed constraints are:
# - $\frac{d}{dt} x = r \cdot \frac{d} {dt}q_2$
# - $\frac{d}{dt} y = -r \cdot \frac{d} {dt}q_1$

# %% [markdown]
# **Description of the system**
#
# The time t appears explicitly in the EOMs.\
# I copy the trick from JM in the example1.2.3 Parameter identification:\
# declare a function T(t), and then\
# pass the time as known trajectory. My OEMs also contain\
# $\frac{d}{dt} T(t)$ and $\frac{d^2}{dx^2} T(t)$\
# As $T(t) = const \cdot t$ I set these derivatives accordingly.
#
# **NOTE**: opty can handle Kane's equations with the reaction forces.\
# They must be declared as state variables.\
# At least with this simulation opty did not find a solution\
# Hence here I set up Kanes equations without reaction\
# forces, and I calculate the reaction forces further down.\
# This works very fast.
#

# %%
# set up the EOMs without virtual speeds and reaction forces.
udisc = Omega * (1 - sm.exp(-alpha * T(t)))
# I do not use sm.integrate(udisc, t), as this gives
# a sm.Piecewise(..) result, but us the result of this
# integration for alpha != 0.
qdisc = (Omega * T(t) + Omega * sm.exp(-alpha * T(t)) / alpha -
    Omega / alpha)
A2.orient_axis(N, qdisc, N.z)
A2.set_ang_vel(N, udisc*N.z)

A1.orient_body_fixed(A2, (q1, q2, q3), '123')
rot = A1.ang_vel_in(N)
A1.set_ang_vel(A2, u1*A1.x + u2*A1.y + u3*A1.z)
rot1 = A1.ang_vel_in(N)

CP.set_pos(O, x*A2.x + y*A2.y)
CP.set_vel(A2, ux*A2.x + uy*A2.y)

Dmc.set_pos(CP, r*N.z)
Dmc.set_vel(N, Dmc.pos_from(O).diff(t, N))

m_Dmc.set_pos(Dmc, r*A1.x)
m_Dmc.v2pt_theory(Dmc, N, A1)

iXX = 2./5. * mb * r**2
iYY = iXX
iZZ = iXX

I = me.inertia(A1, iXX, iYY, iZZ)
Ball = me.RigidBody('Ball', Dmc, A1, mb, (I, Dmc))
observer = me.Particle('observer', m_Dmc, mo)
BODY = [Ball, observer]

FL = [(Dmc, -mb*g*N.z), (m_Dmc, -mo*g*N.z),
    (A1, t1*A1.x + t2*A1.y + t3*A1.z)]

kd = sm.Matrix([ux - x.diff(t), uy - y.diff(t),
    *[(rot - rot1).dot(uv) for uv in N ]])
speed_constr = sm.Matrix([ux - r*u2, uy + r*u1])
hol_constr = sm.Matrix([x - r*q2, y + r*q1])

q_ind = [q1, q2, q3]
q_dep = [x, y]
u_ind = [u1, u2, u3]
u_dep = [ux, uy]

KM = me.KanesMethod(N,
    q_ind=q_ind,
    q_dependent=q_dep,
    u_ind=u_ind,
    u_dependent=u_dep,
    kd_eqs=kd,
    velocity_constraints=speed_constr,
    configuration_constraints=hol_constr,
    )

(fr, frstar) = KM.kanes_equations(BODY, FL)
EOM = kd.col_join(fr + frstar)
EOM = me.msubs((EOM.col_join(hol_constr)),
    {sm.Derivative(T(t), t):
    Tdot, sm.Derivative(T(t), (t,2)): Tdotdot},
    )
print(f'OEM contains {sm.count_ops(EOM):,} operations, ' +
    f'{sm.count_ops(sm.cse(EOM))} after cse')

# %% [markdown]
# Set up the **optimization problem** and solve it.\
# I force $h_{opty} > 0$ with the bounds $h_{opty} \in (0.0001, 1)$\
# to avoid negative 'solutions'.

h_opty = sm.symbols('h_opty')
state_symbols = (*q_ind, *q_dep, *u_ind, *u_dep)
laenge = len(state_symbols)
constant_symbols = (r, mb, mo, g, Omega,
    alpha, Tdot, Tdotdot)
specified_symbols = (t1, t2, t3)
unknown_symbols = []
methode = "backward euler"
num_nodes = 250
duration = (num_nodes - 1) * h_opty

disc_time = 7.5
interval_value = h_opty
interval_value_fix = disc_time/num_nodes

zeit = np.linspace(0, disc_time, num_nodes)
# Specify the known system parameters.
par_map = OrderedDict()
par_map[mb]  = 5.0
par_map[mo]  = 1.0
par_map[r]   = 1.0
par_map[Omega] = 10.0
par_map[alpha] = 0.5
par_map[g]   = 9.81
par_map[Tdot] = interval_value_fix
par_map[Tdotdot] = 0.0

# I minimize the square of the control torques,
# but I also want to minimize the duration of the motion.
# weight give the relative weight of duration to 'energy'.
weight = 2.5e5
def obj(free):
    free1 = free[0: -1]
    Tz1 = free1[laenge * num_nodes: (laenge + 1) * num_nodes]
    Tz2 = free1[(laenge + 1) * num_nodes: (laenge + 2) * num_nodes]
    Tz3 = free1[(laenge + 2) * num_nodes: (laenge + 3) * num_nodes]
    return free[-1] * (np.sum(Tz1**2) + np.sum(Tz2**2) +
        np.sum(Tz3**2)) + free[-1] * weight

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[laenge * num_nodes: (laenge + 1) * num_nodes] = (2.0 * free[-1]
         * free[laenge * num_nodes: (laenge + 1) * num_nodes])
    grad[(laenge + 1) * num_nodes: (laenge + 2) * num_nodes] = (2.0 *
        free[-1] * free[(laenge + 1) * num_nodes: (laenge + 2) * num_nodes])
    grad[(laenge + 2) * num_nodes: (laenge + 3) * num_nodes] = (2.0 * free[-1]
       * free[(laenge + 2) * num_nodes: (laenge + 3) * num_nodes])
    grad[-1] = weight
    return grad

t0, tf = 0.0, duration

# holonomic constrains must be fullfilled.
x_start = 7.0
q2_start = x_start / par_map[r]
y_start = 7.0
q1_start = -y_start / par_map[r]

initial_state_constraints = {
    q1: q1_start,
    q2: q2_start,
    q3: 0.0,
    u1: 0.0,
    u2: 0.0,
    u3: 0.0,
    x: x_start,
    y: y_start,
    ux: 0.0,
    uy: 0.0
    }
final_state_constraints = {
    x: 0.0,
    y: 0.0,
    ux: 0.0,
    uy: 0.0,
    }

instance_constraints = (
) + tuple(
    xi.subs({t: t0}) - xi_val for xi, xi_val in
    initial_state_constraints.items()
) + tuple(
    xi.subs({t: tf}) - xi_val for xi, xi_val in
    final_state_constraints.items()
)

grenze = 10.0
# forcing h_opty > 0. helps to avoid negative h_opty solutions.
bounds = {t1: (-grenze, grenze), t2: (-grenze, grenze),
    t3: (-grenze, grenze), h_opty: (0.0001, 1.0)}

# Create an optimization problem.
prob = Problem(
    obj,
    obj_grad,
    EOM,
    state_symbols,
    num_nodes,
    interval_value,
    known_parameter_map=par_map,
    known_trajectory_map = {T(t): zeit},
    instance_constraints=instance_constraints,
    time_symbol=t,
    bounds=bounds,
    )

# Use an initial guess, meeting the holonomic constraints.
i1b = np.zeros(num_nodes)
i2 = np.linspace(initial_state_constraints[x],
    final_state_constraints[x], num_nodes)
i1a = i2 / par_map[r]
i3 = np.linspace(initial_state_constraints[y],
    final_state_constraints[y], num_nodes)
i1 = -i3 / par_map[r]
i4 = np.zeros(8*num_nodes)
initial_guess = np.hstack((i1,i1a, i1b, i2, i3, i4, 0.01))

# set max number of iterations. Default is 3000.
prob.add_option('max_iter', 2000)

# Find the optimal solution.
for _ in range(1):
    solution, info = prob.solve(initial_guess)
    print('message from optimizer:', info['status_msg'])
    print('Iterations needed',len(prob.obj_value))
    initial_guess = solution

print('length of info[g]', len(info['g']))
prob.plot_objective_value()
prob.plot_constraint_violations(solution)
fig1, ax1 = plt.subplots(14, 1, figsize=(7.25, 3.25*14))
prob.plot_trajectories(solution, ax1)
# %% [markdown]
# with this simulation opty is slow finding the reaction forces.\
# So, I form Kane's EOMs again, but this time with auxiliary speeds\
# and the reaction forces.

t = me.dynamicsymbols._t
q1, q2, q3 = me.dynamicsymbols('q1 q2 q3')
u1, u2, u3 = me.dynamicsymbols('u1 u2 u3')
t1, t2, t3 = me.dynamicsymbols('t1 t2 t3')
T = sm.symbols('T', cls=sm.Function)
Tdot, Tdotdot = sm.symbols('Tdot Tdotdot')

x, y = me.dynamicsymbols('x y')
ux, uy = me.dynamicsymbols('ux uy')
aux = me.dynamicsymbols('auxx, auxy, auxz')
F_r = me.dynamicsymbols('Fx, Fy, Fz')
mb, mo, g, r = sm.symbols('mb, mo, g, r')
Omega, alpha = sm.symbols('Omega, alpha')
rhs = list(sm.symbols('rhs:5'))

N, A1, A2 = sm.symbols('N, A1 A2', cls=me.ReferenceFrame)
O, CP, Dmc, m_Dmc = sm.symbols('O, CP, Dmc, m_Dmc', cls=me.Point)
O.set_vel(N, 0)
udisc = Omega * (1 - sm.exp(-alpha * t))

udisc = Omega * (1 - sm.exp(-alpha * T(t)))
# I do not use sm.integrate(udisc, t), as this gives
# a sm.Piecewise(..) result, but us the result of this
# integration for alpha != 0.
qdisc = Omega * T(t) + Omega * sm.exp(-alpha * T(t)) / alpha - Omega / alpha
A2.orient_axis(N, qdisc, N.z)
A2.set_ang_vel(N, udisc*N.z)

A1.orient_body_fixed(A2, (q1, q2, q3), '123')
rot = A1.ang_vel_in(N)
A1.set_ang_vel(A2, u1*A1.x + u2*A1.y + u3*A1.z)
rot1 = A1.ang_vel_in(N)

CP.set_pos(O, x*A2.x + y*A2.y)
CP.set_vel(A2, ux*A2.x + uy*A2.y)

Dmc.set_pos(CP, r*N.z)
Dmc.set_vel(N, Dmc.pos_from(O).diff(t, N)
    + aux[0] *N.x + aux[1]*N.y + aux[2]*N.z)

m_Dmc.set_pos(Dmc, r*A1.x)
m_Dmc.v2pt_theory(Dmc, N, A1)

iXX = 2./5. * mb * r**2
iYY = iXX
iZZ = iXX

I = me.inertia(A1, iXX, iYY, iZZ)
Ball = me.RigidBody('Ball', Dmc, A1, mb, (I, Dmc))
observer = me.Particle('observer', m_Dmc, mo)
BODY = [Ball, observer]

FL = [(Dmc, -mb*g*N.z + F_r[0]*N.x + F_r[1]*N.y + F_r[2]*N.z),
(m_Dmc, -mo*g*N.z), (A1, t1*A1.x + t2*A1.y + t3*A1.z)]

kd = sm.Matrix([ux - x.diff(t), uy - y.diff(t),
    *[(rot - rot1).dot(uv) for uv in N ]])
speed_constr = sm.Matrix([ux - r*u2, uy + r*u1])
hol_constr = sm.Matrix([x - r*q2, y + r*q1])

q_ind = [q1, q2, q3]
q_dep = [x, y]
u_ind = [u1, u2, u3]
u_dep = [ux, uy]

KM = me.KanesMethod(N,
    q_ind=q_ind,
    q_dependent=q_dep,
    u_ind=u_ind,
    u_dependent=u_dep,
    u_auxiliary=aux,
    kd_eqs=kd,
    velocity_constraints=speed_constr,
    configuration_constraints=hol_constr,
    )

(fr, frstar) = KM.kanes_equations(BODY, FL)
MM = KM.mass_matrix_full
force = me.msubs(KM.forcing_full, {sm.Derivative(T(t), t):
    Tdot, sm.Derivative(T(t), (t,2)): 0},
    {i: 0 for i in F_r},
)
eingepraegt = me.msubs(KM.auxiliary_eqs,
    {sm.Derivative(T(t), t):
    Tdot, sm.Derivative(T(t), (t,2)): 0},
    {i.diff(t): rhs[j] for j, i in enumerate(u_ind + u_dep )},
    )
print(f'eingepraegt contains {sm.count_ops(eingepraegt):,} operations, ' +
    f'{sm.count_ops(sm.cse(eingepraegt))} after cse')

# %% [markdown]
# Calculate and plot the **reaction forces** on the center of the ball.\
# The reaction forces need\
# $\dfrac{d^2}{dt^2}\text{(generalized coordinates)}$\
# It is faster to calculate $\dfrac{d}{dt}\text{(gen. coords)} =\
# MM^{-1} \cdot \text{force}$ numerically,\
# compared to doing it symbolically.\
# (which sympy.physics.mehanics provides for: rhs = KM.rhs() ).\
# Of course, the solution found by opty is used.

qL = q_ind + q_dep + u_ind + u_dep + [T(t)] + [t1, t2, t3]
rhs = list(sm.symbols('rhs:5'))
pL = [i for i in par_map.keys()]
pL_vals = [par_map[i] for i in pL]
MM_lam = sm.lambdify(qL + pL, MM, cse=True)
force_lam = sm.lambdify(qL + pL, force, cse=True)
eingepraegt_lam = sm.lambdify(F_r + qL + pL + rhs, eingepraegt, cse=True)

state_vals, input_vals, _ = parse_free(solution, len(state_symbols),
    len(specified_symbols), num_nodes)

resultat2 = state_vals.T
schritte2 = resultat2.shape[0]
# times must be scaled, as opty used the time num_nodes * h_opty
# when integrating the EOMs
times2 = zeit * solution[-1] * (num_nodes - 1) / disc_time

RHS1 = np.empty((schritte2, resultat2.shape[1]))
for i in range(schritte2):
    zeit1 = times2[i]
    t11, t21, t31 = input_vals.T[i, :]
    RHS1[i, :] = np.linalg.solve(MM_lam(*[resultat2[i, j]for j in
        range(resultat2.shape[1])], zeit1, t11, t21, t31,  *pL_vals),
        force_lam(*[resultat2[i, j] for j in range(resultat2.shape[1])],
        zeit1, t11, t21, t31, *pL_vals)).reshape(10)

#calculate implied forces numerically
def func (x, *args):
# just serves to 'modify' the arguments for fsolve.
    return eingepraegt_lam(*x, *args).reshape(3)

kraftx  = np.empty(schritte2)
krafty  = np.empty(schritte2)
kraftz  = np.empty(schritte2)

x0 = tuple((1., 1., 1.))   # initial guess

for i in range(schritte2):
    for _ in range(1):
        y0 = [resultat2[i, j] for j in range(resultat2.shape[1])]
        rhs = [RHS1[i, j] for j in range(5, 10)]
        t11, t21, t31 = input_vals.T[i, :]
        zeit1 = times2[i]
        args = tuple(y0 + [zeit1, t11, t21, t31] + pL_vals + rhs)
        AAA = root(func, x0, args=args)
# improved guess. Should speed up convergence of fsolve
        x0 = AAA.x
    kraftx[i] = AAA.x[0]
    krafty[i] = AAA.x[1]
    kraftz[i] = AAA.x[2]

fig, ax = plt.subplots(figsize=(10, 5))
for i, j in zip((kraftx, krafty, kraftz),
    ('reaction force on Dmc in X direction',
    'reaction force on Dmc in Y direction',
    'reaction force on Dmc in Z direction')):
# time has to be scaled properly.
    plt.plot(times2, i, label=j)
ax.set_title('Reaction Forces on center of the ball')
ax.set_xlabel('time [sec]')
ax.set_ylabel('force [N]')
ax.grid(True)
plt.legend();

# %% [markdown]
# **Animate** the system
fps = 40

def add_point_to_data(line, x, y):
# to trace the path of the point. Copied from Timo.
    old_x, old_y = line.get_data()
    line.set_data(np.append(old_x, x), np.append(old_y, y))

state_vals, input_vals, _ = parse_free(solution, len(state_symbols),
    len(specified_symbols), num_nodes)
t_arr = np.linspace(t0, num_nodes*solution[-1], num_nodes)
state_sol = CubicSpline(t_arr, state_vals.T)
input_sol = CubicSpline(t_arr, input_vals.T)

# set the radius of the disc, so the ball will
# stay on it.
x_max = np.max(np.max([np.abs(state_vals[3, i])for i in range(num_nodes)]))
y_max = np.max(np.max([np.abs(state_vals[4, i])for i in range(num_nodes)]))
r_disc = np.sqrt(x_max**2 + y_max**2) * 1.2

t1h, t2h, t3h = sm.symbols('t1h t2h t3h')
Pl, Pr, Pu, Pd, T_total, T_Z  = sm.symbols('Pl Pr Pu Pd, T_total, T_Z',
    cls=me.Point)
Pl.set_pos(O, -r_disc*A2.x)
Pr.set_pos(O, r_disc*A2.x)
Pu.set_pos(O, r_disc*A2.y)
Pd.set_pos(O, -r_disc*A2.y)
T_total.set_pos(Dmc, t1h*A1.x + t2h*A1.y + t3h*A1.z)

# show the perpendicula portion of the torque vector
# in the X/Y plane. A bit arbitraryly, I show it
#  perpendicular to t_Total.
hilfs = sm.Max(t1h**2 + t2h**2 + t3h**2, 1.e-15)
x_coord = (t1h*A1.x + t2h*A1.y + t3h*A1.z).dot(A2.x)
y_coord = (t1h*A1.x + t2h*A1.y + t3h*A1.z).dot(A2.y)
T_Z.set_pos(Dmc, t3h/sm.sqrt(hilfs) * (y_coord*A2.x - x_coord*A2.y))

coordinates = Dmc.pos_from(O).to_matrix(N)
for point in (m_Dmc, Pl, Pr, Pu, Pd, T_total, T_Z):
    coordinates = coordinates.row_join(point.pos_from(O).to_matrix(N))

pL, pL_vals = zip(*par_map.items())
coords_lam = sm.lambdify(list(state_symbols) + [t1h, t2h, t3h] + list(pL)
    + [T(t)], coordinates, cse=True)

def init_plot():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-r_disc-1, r_disc+1)
    ax.set_ylim(-r_disc-1, r_disc+1)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)

# draw the spokes
    line1, = ax.plot([], [], lw=2, marker='o', markersize=0, color='black')
    line2, = ax.plot([], [], lw=2, marker='o', markersize=0, color='black')
    line3, = ax.plot([], [], lw=2, marker='o', markersize=0, color='black')
    line4, = ax.plot([], [], lw=2, marker='o', markersize=0, color='black')

    line5, = ax.plot([],[], color='magenta', lw=0.5)
# draw the torque vektor
    pfeil1 = ax.quiver([], [], [], [], color='green', scale=30, width=0.004)
    pfeil2 = ax.quiver([], [], [], [], color='blue', scale=30, width=0.004)
# draw the ball
    ball = patches.Circle((initial_state_constraints[x],
        initial_state_constraints[y]),
        radius=par_map[r], fill=True, color='magenta', alpha=0.75)
    ax.add_patch(ball)
# draw the observer
    observer, = ax.plot([], [], marker='o', markersize=5, color='blue')
    return (fig, ax, line1, line2, line3, line4, line5, ball,
        observer, pfeil1, pfeil2)

(fig, ax, line1, line2, line3, line4, line5, ball, observer,
    pfeil1, pfeil2) = init_plot()

# draw the disc
phi = np.linspace(0, 2*np.pi, 500)
x_phi = r_disc * np.cos(phi)
y_phi = r_disc * np.sin(phi)
ax.plot(x_phi, y_phi, color='black', lw=2)

# Function to update the plot for each animation frame
def update(t):
    message = (f'running time {t:.2f} sec \n The green arrow is the ' +
        f'projection of the torque vector on the X/Y plane \n' +
        f'The blue arrow is the component of the torque perpendicular ' +
        f'to the disc \n' +
        f'The blue dot is the observer')
    ax.set_title(message, fontsize=12)

    coords = coords_lam(*state_sol(t), *input_sol(t), *pL_vals, t)
    line1.set_data([0, coords[0, 2]], [0, coords[1, 2]])
    line2.set_data([0, coords[0, 3]], [0, coords[1, 3]])
    line3.set_data([0, coords[0, 4]], [0, coords[1, 4]])
    line4.set_data([0, coords[0, 5]], [0, coords[1, 5]])

    add_point_to_data(line5, coords[0, 0], coords[1, 0])
    observer.set_data([coords[0, 1]], [coords[1, 1]])
    ball.set_center((coords[0, 0], coords[1, 0]))
    pfeil1.set_offsets([coords[0, 0], coords[1, 0]])
    pfeil1.set_UVC(coords[0, -2] - coords[0, 0] , coords[1, -2] - coords[1, 0])

    pfeil2.set_offsets([coords[0, 0], coords[1, 0]])
    pfeil2.set_UVC(coords[0, -1] - coords[0, 0] , coords[1, -1] - coords[1, 0])

    return line1, line2, line3, line4, line5, ball, observer, pfeil1, pfeil2


# Create the animation
animation = FuncAnimation(fig, update, frames=np.arange(t0,
    num_nodes*solution[-1], 1 / fps), interval=1.5*fps, blit=False)

# below the animation
#display(HTML(animation.to_jshtml()))

# %%
# A frame from the animation.
(fig, ax, line1, line2, line3, line4, line5, ball, observer,
    pfeil1, pfeil2) = init_plot()

# sphinx_gallery_thumbnail_number = 6
# draw the disc
phi = np.linspace(0, 2*np.pi, 500)
x_phi = r_disc * np.cos(phi)
y_phi = r_disc * np.sin(phi)
ax.plot(x_phi, y_phi, color='black', lw=2)
update(2)

plt.show()