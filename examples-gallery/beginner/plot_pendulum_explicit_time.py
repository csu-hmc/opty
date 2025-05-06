r"""
Pendulum with Explicit Times
============================

Objectives
----------

- Show how to handle an explicit occurance of time in the equations of motion,
  which opty presently cannot handle directly.
- Show how to use additional state variables to have instance constraints on
  functions of state variables and imputs.


Introduction
------------

The task is to get a pendulum which is hanging straight down to swing up to a
vertical position and then to rest there. A torque of the form
:math:`\textrm{driving}_{\textrm{torque}} = t \cdot F \cdot \sin(\omega
\cdot t)`
is applied to the pendulum. As opty presently does not support explicit time
in the equations of motion, one has to introduce an additional state
variable, :math:`T` and set :math:`\dfrac{dT}{dt} = 1` in the equations of
motion. Setting an instance constraint :math:`T(t_0) = 0` will ensure that
:math:`T` is equal to the time at any point in time.

To ensure that the pendulum is at rest, set :math:`\dfrac{d^2q}{dt^2} =
acc_{\textrm{help}}` in the equations of motion. The instance constraint
:math:`acc_{\textrm{help}}(t_f) = 0` will ensure that the pendulum is at rest
at the end of the simulation.

Notes
-----

- The driving torque seems a bit 'artificial'. It was selected to show how to
  handle explicit time in the equations of motion.
- This seems to be a difficult problem for opty, it takes almost 30,000
  iterations to solve it.
- Note that opty sets the ``unkonwn input trajectory`` :math:`\omega` such that
  the actual torque on the pendulum corresponds to the 'bang bang' solution
  one would intuitively expect.

**States**

- :math:`q` - angle of the pendulum
- :math:`u` - angular velocity of the pendulum
- :math:`T` - time
- :math:`acc_{\textrm{help}}` - angular acceleration of the pendulum

**Parameters**

- :math:`m_p` - mass of the pendulum [kg]
- :math:`l_e` - length of the pendulum [kg]
- :math:`i_{ZZ}` - moment of inertia of the pendulum [kg m^2]
- :math:`g` - gravitational acceleration [m/s^2]
- :math:`\nu` - damping coefficient [kg m^2/s]

**Inputs**

- :math:`F` - driving force [N]
- :math:`\omega` - frequency of the driving force [1/s]


"""
import os
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from opty import Problem
from opty.utils import MathJaxRepr

from scipy.interpolate import CubicSpline
from matplotlib.animation import FuncAnimation

# %%
N, A = sm.symbols('N A', cls=me.ReferenceFrame)
O, P = sm.symbols('O P', cls=me.Point)
O.set_vel(N, 0)
t = me.dynamicsymbols._t

q, u = me.dynamicsymbols('q u')
acc_help, omega, T = me.dynamicsymbols('acc_help, omega, T')

mp, g, le, iZZ, F, nu = sm.symbols('m_p, g, le, i_ZZ, F, nu')

A.orient_axis(N, q, N.z)
A.set_ang_vel(N, u * N.z)

P.set_pos(O, -le * A.y)
P.v2pt_theory(O, N, A)

inert = me.inertia(A, 0, 0, iZZ)
pendulum = me.RigidBody('pendulum', P, A, mp, (inert, P))
bodies = [pendulum]

driving_torque = T * F * sm.sin(omega * T)
forces = [(A, driving_torque * N.z - u * nu * A.z), (P, -mp * g * N.y)]

kd = sm.Matrix([u - q.diff(t)])

KM = me.KanesMethod(N,
                    q_ind=[q],
                    u_ind=[u],
                    kd_eqs=kd)

fr, frstar = KM.kanes_equations(bodies, forces)
eom = kd.col_join(fr + frstar)
eom = eom.col_join(sm.Matrix([T.diff(t) - 1,
                              acc_help - u.diff(t),
                              ]))

MathJaxRepr(eom)

# %%
# Set Up the Problem and Solve it
# -------------------------------
h = sm.symbols('h')
state_symbols = [q, u, T, acc_help]

num_nodes = 301
t0, tf = 0.0, h * (num_nodes - 1)
interval_value = h

par_map = {}
par_map[mp] = 1.0
par_map[g] = 9.81
par_map[le] = 1.0
par_map[iZZ] = 1.0
par_map[nu] = 0.1

bounds = {
    h: (0.0, 0.5),
    omega: (0.0, 2.0*np.pi),
    F: (-1.0, 1.0),
}

instance_constraints = (
    q.func(t0) - 0.0,
    u.func(t0) - 0.0,
    T.func(t0) - 0.0,
    q.func(tf) - np.pi,
    u.func(tf) - 0.0,
    acc_help.func(tf) - 0.0,
)


def obj(free):
    return free[-1]


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad


# %%
# Here ``backend_'numpy'`` is used to speed up setting up the Problem if a
# solution is available.
prob = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    num_nodes,
    interval_value,
    known_parameter_map=par_map,
    instance_constraints=instance_constraints,
    time_symbol=t,
    bounds=bounds,
    backend='numpy'
    )

# %%
fname = f'pendulum_explicit_time_{num_nodes}_nodes_solution.csv'

# Use the existing solution if avaliable, else solve the problem.
if os.path.exists(fname):
    # Use existing solution.
    solution = np.loadtxt(fname)
else:
    # Solve the problem.
    # Here the default value backend = 'cython' is used to expedite the
    # solution process; there are almost 30,000 iterations.
    prob = Problem(
        obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        known_parameter_map=par_map,
        instance_constraints=instance_constraints,
        time_symbol=t,
        bounds=bounds,
    )

    prob.add_option('max_iter', 35000)
    initial_guess = np.ones(prob.num_free) * 0.1
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    _ = prob.plot_objective_value()

# %%
# Plot the violations of the constraints.
_ = prob.plot_constraint_violations(solution, subplots=True)
# %%
# Plot the trajectories.
fig, axes = plt.subplots(6, 1, figsize=(6.5, 6.5), layout='constrained',
                         sharex=True)
prob.plot_trajectories(solution, show_bounds=True, axes=axes)
sol, input, constant_values, _ = prob.parse_free(solution)
driving_torque_lam = sm.lambdify((*state_symbols, omega, F), driving_torque,
                                 cse=True)

driving_torque_values = []
for i in range(num_nodes):
    driving_torque_values.append(driving_torque_lam(*sol[:, i],
                                                    input[i],
                                                    constant_values[0]))
times = prob.time_vector(solution)
axes[-1].plot(times, driving_torque_values)
axes[-1].set_ylabel('Driving Torque')
axes[-1].set_xlabel('Time [s]')
_ = axes[-1].set_title('Actual Driving Torque')


# %%
# Animation
# ---------
fps = 20

state_vals, input_vals, constant_values, h_vals = prob.parse_free(solution)
t_arr = prob.time_vector(solution)
state_sol = CubicSpline(t_arr, state_vals.T)
input_sol = CubicSpline(t_arr, input_vals.T)

act_torque = [state_vals[2, i] * constant_values[0] * np.sin(
    state_vals[2, i] * input_vals[i]) for i in range(num_nodes)]
torque_sol = CubicSpline(t_arr, act_torque)

pL, pL_vals = zip(*par_map.items())
coordinates = P.pos_from(O).to_matrix(N)
coords_lam = sm.lambdify((*state_symbols, omega, F, *pL), coordinates,
                         cse=True)


# sphinx_gallery_thumbnail_number = 3
def init():
    xmin, xmax = -par_map[le] - 0.5, par_map[le] + 0.5
    ymin, ymax = -par_map[le] - 0.5, par_map[le] + 0.5

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid()
    ax.scatter(0.0, 0.0, color='black', marker='o', s=100)  # origin O

    line1 = ax.scatter([], [], color='blue', marker='o', s=100)   # point P
    line2, = ax.plot([], [], color='magenta', lw=1)  # connecting line
    arrow = ax.quiver([], [], [], [], color='green', scale=15,
                      width=0.004, headwidth=8)  # torque arrow
    return fig, ax, line1, line2, arrow


# Function to update the plot for each animation frame
fig, ax, line1, line2, arrow = init()


def update(t):
    message = ((f'Running time {t:.2f} sec \n The green arrow corresponds to '
                f'the driving torque'))
    ax.set_title(message, fontsize=12)

    coords = coords_lam(*state_sol(t), input_sol(t), constant_values[0],
                        *pL_vals)

    line1.set_offsets([coords[0, 0], coords[1, 0]])
    line2.set_data([0.0, coords[0, 0]], [0.0, coords[1, 0]])
    arrow.set_offsets([0.0, 0.0])
    arrow.set_UVC(torque_sol(t), 0.0)
    return line1, line2, arrow


tf = h_vals * (num_nodes - 1)
frames = np.linspace(t0, tf, int(fps * (tf - t0)))
animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)

plt.show()
