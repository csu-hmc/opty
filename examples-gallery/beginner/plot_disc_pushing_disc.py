# %%
r"""
Disc Pushing Disc
=================

Description
-----------

A homogenious disc (:math:`\textrm{disc}_2`) of mass :math:`m_2` and radius
:math:`r_2` is at rest at point A. Another disc (:math:`\textrm{disc}_1`)
of mass :math:`m_1` and radius :math:`r_1` is at rest at point B. A control
force
:math:`\vec{F} = \begin{pmatrix} f_x \\ f_y \end{pmatrix}` can be applied to
:math:`\textrm{disc}_1`.
The goal is to push :math:`\textrm{disc}_2` to a target point C, using
:math:`\textrm{disc}_1` with minimal
:math:`\int_0^{t_f} \lvert \vec{F} \rvert \,dt`.
The motion is in the horizontal Y/Y plane. A Coulomb friction force with
coefficient :math:`\mu_c` acts on the discs.
The collision between the discs is modeled as a spring force proportional to
the penetration, with proportionality constant :math:`k_{\textrm{spring}}`.
A friction force proportional to the difference of the tangential speeds at
the contact points, with coefficient :math:`\mu_s` also acts on the discs.
A particle of mass :math:`m_p` is fixed on the rim of each disc.

Notes
-----

- If point B is too far from the line AC one needs to iterate from a point b
  close to the line to the desired point B. This iteration may take a long
  time. In the simulation, a solution from an earlier iteration is used to
  minimize running time.
- In order to bound the control force :math:`\vec{F}`,
  :math:`\lvert \vec{F} \rvert = \sqrt{f_x^2 + f_y^2}` is added as an
  additional equation to the equations ofmotion, and bounded with
  ``eom_bounds``.
- Bounding the distance between the discs also seems to help convergence.

**States**

- :math:`x_1, y_1`: coordinates of the center of mass of
  :math:`\textrm{disc}_1`
- :math:`x_2, y_2`: coordinates of the center of mass of
  :math:`\textrm{disc}_2`
- :math:`q_1, q_2`: angles of the discs
- :math:`u_{x1}, u_{y1}`: speeds of the center of mass of
  :math:`\textrm{disc}_1`
- :math:`u_{x2}, u_{y2}`: speeds of the center of mass of
  :math:`\textrm{disc}_2`
- :math:`u_1, u_2`: angular speeds of the discs

**Control**

- :math:`f_x, f_y`: components of the control force applied to
  :math:`\textrm{disc}_1`

**Parameters**

- :math:`m_1, m_2`: masses of the discs
- :math:`m_p`: mass of the particles fixed on the rim of the discs
- :math:`r_1, r_2`: radii of the discs
- :math:`k_{\textrm{spring}}`: spring constant for the collision between the
  discs
- :math:`g`: gravitational acceleration
- :math:`\mu_c, \mu_s`: Coulomb and speed dependent friction coefficients

"""
import os
import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
import matplotlib.pyplot as plt

from opty import Problem
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch

# %%
# Kane's Equations of Motion
# --------------------------

N, A1, A2 = sm.symbols('N A1 A2', cls=me.ReferenceFrame)
O, Dmc1, Dmc2 = sm.symbols('O Dmc1 Dmc2', cls=me.Point)
c1, c2, P1, P2 = sm.symbols('c1 c2 P1 P2', cls=me.Point)
t = me.dynamicsymbols._t
O.set_vel(N, 0)

# %%
# Angular coordinates and speeds of the discs.
q1, q2, u1, u2 = me.dynamicsymbols('q1 q2 u1 u2')

# %%
# Locations and speeds of the centers of mass of the discs.
x1, y1, x2, y2, ux1, uy1, ux2, uy2 = me.dynamicsymbols(
    'x1 y1 x2 y2 ux1 uy1 ux2 uy2')

# %%
# Control forces acting on Dmc1.
fx, fy = me.dynamicsymbols('fx fy')

# %%
# Masses and radii of the discs, spring constant.

m1, m2, mp, r1, r2, k_spring, g = sm.symbols('m1 m2 mp r1 r2 k_spring g')

# %%
# Coulomb and speed dependent friction coefficients.
mu_c, mu_s = sm.symbols('mu_c mu_s')

# %%
# Define the reference frames for the two discs.
A1.orient_axis(N, q1, N.z)
A1.set_ang_vel(N, u1 * N.z)
A2.orient_axis(N, q2, N.z)
A2.set_ang_vel(N, u2 * N.z)

# %%
# Define the positions of the centers of mass of the discs.
Dmc1.set_pos(O, x1 * N.x + y1 * N.y)
Dmc2.set_pos(O, x2 * N.x + y2 * N.y)
Dmc1.set_vel(N, ux1 * N.x + uy1 * N.y)
Dmc2.set_vel(N, ux2 * N.x + uy2 * N.y)

P1.set_pos(Dmc1, r1 * A1.y)
P2.set_pos(Dmc2, r2 * A2.y)
P1.v2pt_theory(Dmc1, N, A1)
_ = P2.v2pt_theory(Dmc2, N, A2)

# %%
# Define the contact points.
vec_Dmc1_to_Dmc2 = Dmc2.pos_from(Dmc1).normalize()
c1.set_pos(Dmc1, r1 * vec_Dmc1_to_Dmc2)
c2.set_pos(Dmc2, -r2 * vec_Dmc1_to_Dmc2)

# %%
# Note: c1.pos_from(O).diff(t, N) does not give the correct speed: It cannot
# know, that c1 is a point fixed on the disc.
# So, ``v2pt_theory`` needs to be used, which assumes that c1 is fixed in A1.
c1.v2pt_theory(Dmc1, N, A1)
_ = c2.v2pt_theory(Dmc2, N, A2)


# %%
# Define some smooth functions needed.

steep = 50


def coulomb_direction(xx, steep=steep):
    """returns 1 if xx > 0, -1 if xx < 0"""
    return sm.tanh(steep * xx)


def smooth_step(xx, steep=steep):
    """returns 0 if xx < 0, 1 if xx > 0"""
    return 0.5 * (1 + sm.tanh(steep * xx))


def distance_c1_c2(Dmc1, Dmc2):
    """returns the distance between the contact points
    negative if the discs are overlapping"""
    return (Dmc2.pos_from(Dmc1)).magnitude() - r1 - r2


# %%
# Coulomb friction on the discs, acting at the centers of mass.
FL1 = [
    (Dmc1, -mu_c * m1 * g * coulomb_direction(ux1) * N.x +
     -mu_c * m1 * g * coulomb_direction(uy1) * N.y),
    (Dmc2, -mu_c * m2 * g * coulomb_direction(ux2) * N.x +
     -mu_c * m2 * g * coulomb_direction(uy2) * N.y)
]

# %%
# Forces at the contact points.
vecDmc1Dmc2 = Dmc2.pos_from(Dmc1).normalize()
vectangent = vecDmc1Dmc2.cross(N.z)

abstand = distance_c1_c2(Dmc1, Dmc2)
FL2 = [
    (c1, k_spring * abstand * (1 - smooth_step(abstand)) * vecDmc1Dmc2),
    (c2, -k_spring * abstand * (1 - smooth_step(abstand)) * vecDmc1Dmc2)
]

speed_delta = c1.vel(N) - c2.vel(N)
FL3 = [
    (c1, -mu_s * speed_delta.dot(vectangent) * vectangent),
    (c2, mu_s * speed_delta.dot(vectangent) * vectangent)
]

# %%
# Control forces on Dmc1.
FL4 = [(Dmc1, fx * N.x + fy * N.y)]

FL = FL1 + FL2 + FL3 + FL4

# %%
# Define the bodies.
iZZ1 = 0.5 * m1 * r1**2
iZZ2 = 0.5 * m2 * r2**2

inert1 = me.inertia(A1, 0, 0, iZZ1)
inert2 = me.inertia(A2, 0, 0, iZZ2)

disc1 = me.RigidBody('Disc1', Dmc1, A1, m1, (inert1, Dmc1))
disc2 = me.RigidBody('Disc2', Dmc2, A2, m2, (inert2, Dmc2))
P1a = me.Particle('P1a', P1, mp)
P2a = me.Particle('P2a', P2, mp)

bodies = [disc1, disc2, P1a, P2a]

# %%
# Kinematic differential equations.
kd = sm.Matrix([
    ux1 - x1.diff(t),
    uy1 - y1.diff(t),
    ux2 - x2.diff(t),
    uy2 - y2.diff(t),
    u1 - q1.diff(t),
    u2 - q2.diff(t)
])

# %%
# Form the equations of motion using Kane's method.
q_ind = [x1, y1, x2, y2, q1, q2]
u_ind = [ux1, uy1, ux2, uy2, u1, u2]

kane = me.KanesMethod(N, q_ind, u_ind, kd_eqs=kd)
fr, frstar = kane.kanes_equations(bodies, FL)

eom = kd.col_join(fr + frstar)

# %%
# The magnitude of the force is limited.
# Bounding abstand seems to help convergence.
eom = eom.col_join(sm.Matrix([sm.sqrt(fx**2 + fy**2), abstand]))

print(f"eom contains {sm.count_ops(eom)} operations, "
      f"and has shape {eom.shape}")

# %%
# Set Up the Optimization
# -----------------------

state_symbols = [x1, y1, x2, y2, q1, q2, ux1, uy1, ux2, uy2, u1, u2]
num_nodes = 501
t0, tf = 0.0, 3.0
interval_value = (tf - t0) / (num_nodes - 1)

# %%
# Set the known parameters.
par_map = {
    m1: 1.56,
    m2: 1.0,
    mp: 1.0,
    r1: 1.25,
    r2: 1.0,
    k_spring: 1000.0,
    g: 9.81,
    mu_c: 0.25,
    mu_s: 0.25,
}

# %%
# Define the objective function and its gradient.


def obj(free):
    """minimize the force needed."""
    summe = (np.sum([free[12*num_nodes + i]**2 + free[13*num_nodes + i]**2
                    for i in range(num_nodes)]) * interval_value)
    return summe


def obj_grad(free):
    """gradient of the objective function."""
    grad = np.zeros_like(free)
    for i in range(num_nodes):
        grad[12*num_nodes + i] = 2 * free[12*num_nodes + i] * interval_value
        grad[13*num_nodes + i] = 2 * free[13*num_nodes + i] * interval_value
    return grad


# %%
# Define the instance constraints.
instance_constraints = [
    x1.func(t0) - 8.75,
    y1.func(t0) - 0.0,
    x2.func(t0) - 5.0,
    y2.func(t0) - 5.0,
    ux1.func(t0) - 0.0,
    uy1.func(t0) - 0.0,
    ux2.func(t0) - 0.0,
    uy2.func(t0) - 0.0,
    q1.func(t0) - 0.0,
    q2.func(t0) - 0.0,
    u1.func(t0) - 0.0,
    u2.func(t0) - 0.0,
    x2.func(tf) - 10.0,
    y2.func(tf) - 10.0,
]

# %%
# Bound the control forces.
eom_bounds = {
    # This limits |force| that can be applied.
    12: (0.0, 25.0),
    # Limits the distance between the discs. May help convergence.
    13: (-15.0, 15.0)
}

# %%
# Find a solution.
#
# Use a given initial guess if available, else iterate to a solution  with
# an initial guess which gives a plausible first guess for the movement of
# disc1 and disc2.
# This iteration may take a long time.

fname = f"disc_pushing_disc_{num_nodes}_nodes_solution.csv"
if os.path.exists(fname):

    prob = Problem(
        obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        known_parameter_map=par_map,
        instance_constraints=instance_constraints,
        eom_bounds=eom_bounds,
        time_symbol=t,
        backend='numpy'
    )

    initial_guess = np.loadtxt(fname)
    solution, info = prob.solve(initial_guess)
    print(info['status_msg'])

else:

    for i in range(12):
        instance_constraints[0] = x1.func(t0) - 6.0 - 0.25 * i

        prob = Problem(
            obj,
            obj_grad,
            eom,
            state_symbols,
            num_nodes,
            interval_value,
            known_parameter_map=par_map,
            instance_constraints=instance_constraints,
            eom_bounds=eom_bounds,
            time_symbol=t,
        )
        if i == 0:
            initial_guess = np.ones(prob.num_free) * 0.5
            half = num_nodes // 2
            rest = num_nodes - half
            # x1
            initial_guess[0: half] = np.linspace(6.0, 5.1, half)
            initial_guess[half: num_nodes] = np.linspace(5.0, 10.1, rest)
            # y1
            initial_guess[num_nodes: num_nodes + half] = \
                np.linspace(0.0, 5.1, half)
            initial_guess[num_nodes + half: 2*num_nodes] = \
                np.linspace(5.1, 10.1, rest)
            # x2
            initial_guess[2*num_nodes: 3*num_nodes] = \
                np.linspace(5.0, 10.0, num_nodes)
            # y2
            initial_guess[3*num_nodes: 4*num_nodes] = \
                np.linspace(5.0, 10.0, num_nodes)
        else:
            initial_guess = solution

        prob.add_option('max_iter', 60000)
        solution, info = prob.solve(initial_guess)
        initial_guess = solution
        print(info['status_msg'])

    #np.savetxt(fname, solution, fmt='%.12f')

# %%
# Plot the trajectories

_ = prob.plot_trajectories(solution)

# %%
# Plot errors.

_ = prob.plot_constraint_violations(solution, subplots=True, show_bounds=True)

# %%
# Plot the objective value.

_ = prob.plot_objective_value()

# %%
# Animation
# ---------

# %%


def animateur(resultat, inputs, t0, tf, schritte):
    """
    returns the animation.
    """

    fps = 15

    t_arr = np.linspace(t0, tf, schritte)
    state_sol = interp1d(t_arr, resultat, kind='cubic', axis=0)
    input_sol = interp1d(t_arr, inputs, kind='cubic', axis=0)

    # Define point of the arrow
    arrow_head = me.Point('arrow_head')
    arrow_head.set_pos(Dmc1, fx/5 * N.x + fy/5 * N.y)

    coords = Dmc1.pos_from(O).to_matrix(N)
    for point in (Dmc2, c1, c2,  P1, P2, arrow_head):
        coords = coords.row_join(point.pos_from(O).to_matrix(N))

    pL = [key for key in par_map.keys()]
    pL_vals = [par_map[key] for key in par_map.keys()]

    qL = [x1, y1, x2, y2, q1, q2, ux1, uy1, ux2, uy2, u1, u2]
    coords_lam = sm.lambdify(qL + [fx, fy] + pL, coords, cse=True)
    coords_vals = coords_lam(*resultat[0, 0: 12], *inputs[0, 0:2], *pL_vals)

    fig, ax = plt.subplots(figsize=(7, 7))

    arrow = FancyArrowPatch([0.0, 0.0], [0.0, 0.0],
                            arrowstyle='-|>',     # nicer arrow head
                            mutation_scale=20,    # makes head bigger
                            linewidth=1,
                            color='green')
    ax.add_patch(arrow)

    disc1 = Circle(
        (coords_vals[0, 1], coords_vals[1, 1]),          # center
        radius=par_map[r1],      # full width = 2a
        facecolor='red',
        edgecolor='red',
        alpha=0.25,
    )

    disc2 = Circle(
        (coords_vals[0, 2], coords_vals[1, 2]),          # center
        radius=par_map[r2],      # full width = 2a
        facecolor='blue',
        edgecolor='blue',
        alpha=0.25,
    )

    # centers of discs
    line1 = ax.scatter(coords_vals[0, 0], coords_vals[1, 0], color='red', s=25,
                       edgecolor='black')
    line2 = ax.scatter(coords_vals[0, 1], coords_vals[1, 1], color='green',
                       s=25, edgecolor='black')
    # contact points
    line3 = ax.scatter(coords_vals[0, 2], coords_vals[1, 2], color='red', s=15,
                       edgecolor='black')
    line4 = ax.scatter(coords_vals[0, 3], coords_vals[1, 3], color='green',
                       s=15, edgecolor='black')

    # Particles
    line5 = ax.scatter(coords_vals[0, 4], coords_vals[1, 4], color='black',
                       s=25)
    line6 = ax.scatter(coords_vals[0, 5], coords_vals[1, 5], color='black',
                       s=25)

    # the line between contact points
    line7, = ax.plot([coords_vals[0, 2], coords_vals[0, 3]],
                     [coords_vals[1, 2], coords_vals[1, 3]], color='black',
                     linestyle='-', lw=0.5)

    ax.add_patch(disc1)
    ax.add_patch(disc2)

    # Trace of the center of disc 1
    trace, = ax.plot([], [], color='red', linestyle='-', lw=0.5)

    # Set limits
    x_max = (np.max((np.concatenate((resultat[:, 0], resultat[:, 2])))) +
             np.max((par_map[r1], par_map[r2])) + 1.0)
    x_min = (np.min(np.concatenate((resultat[:, 0], resultat[:, 2]))) +
             min(-1.0, -np.max((par_map[r1], par_map[r2])) - 1.0))
    y_max = (np.max(np.concatenate((resultat[:, 1], resultat[:, 3]))) +
             np.max((par_map[r1], par_map[r2])) + 1.0)
    y_min = (np.min(np.concatenate((resultat[:, 1], resultat[:, 3]))) +
             min(-1.0, -np.max((par_map[r1], par_map[r2])) - 1.0))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]', fontsize=15)
    ax.set_ylabel('y [m]', fontsize=15)
    ax.scatter(10.0, 10.0, color='green', s=50, edgecolor='black')

    # Animation update function

    def update(frame):

        t = frame

        ax.set_title(f"Running time: {t:.2f} s (shown slightly slow motion)\n "
                     "Control force is the green arrow with magnitude: "
                     f"{np.linalg.norm(input_sol(t)):.2f} N \n "
                     "The black dots are the particles.")
        coords_vals = coords_lam(*state_sol(t)[0: 12], *input_sol(t)[0:2],
                                 *pL_vals)
        # Update disc position
        disc1.set_center((coords_vals[0, 0], coords_vals[1, 0]))
        disc2.set_center((coords_vals[0, 1], coords_vals[1, 1]))

        arrow.set_positions(np.array([coords_vals[0, 0], coords_vals[1, 0]]),
                            np.array([coords_vals[0, 6], coords_vals[1, 6]]))

        line1.set_offsets((coords_vals[0, 0], coords_vals[1, 0]))
        line2.set_offsets((coords_vals[0, 1], coords_vals[1, 1]))
        line3.set_offsets((coords_vals[0, 2], coords_vals[1, 2]))
        line4.set_offsets((coords_vals[0, 3], coords_vals[1, 3]))
        line5.set_offsets((coords_vals[0, 4], coords_vals[1, 4]))
        line6.set_offsets((coords_vals[0, 5], coords_vals[1, 5]))
        line7.set_data([coords_vals[0, 2], coords_vals[0, 3]],
                       [coords_vals[1, 2], coords_vals[1, 3]])

        # This 'complete calculation' is needed to avoid that the trace remains
        # in the plot.
        xdata, ydata = [], []
        for zeit in np.arange(t0, t, 1.0/fps):
            coords_temp = coords_lam(*state_sol(zeit)[0: 12],
                                     *input_sol(zeit)[0:2],
                                     *pL_vals)

            xdata.append(coords_temp[0, 0])
            ydata.append(coords_temp[1, 0])
        trace.set_data(xdata, ydata)

        return (disc1, disc2, line1, line2, line3, line4, line5, line6, line7,
                arrow, trace)

    # Create animation

    ani = FuncAnimation(fig, update,
                        frames=np.concatenate((np.arange(0, tf, 1.0/fps),
                                               np.array([tf]))),
                        interval=1500/fps, blit=False)

    return ani


resultat, inputs, *_ = prob.parse_free(solution)
resultat = resultat.T
inputs = inputs.T
ani = animateur(resultat, inputs, t0=0.0, tf=tf, schritte=num_nodes)
plt.show()
