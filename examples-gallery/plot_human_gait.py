r"""
Human Gait
==========

This example replicates a similar solution as shown in [Ackermann2010]_ using
joint torques as inputs instead of muscle activations [1]_.

pygait2d and symmeplot and their dependencies must be installed first to run
this example. Note that pygait2d has not been released to PyPi or Conda Forge::

    conda install cython pip pydy pyyaml setuptools symmeplot sympy
    python -m pip install --no-deps --no-build-isolation git+https://github.com/csu-hmc/gait2d

gait2d provides a joint torque driven 2D bipedal human dynamical model with
seven body segments (trunk, thighs, shanks, feet) and foot-ground contact
forces based on the description in [Ackermann2010]_.

The optimal control goal is to find the joint torques (hip, knee, ankle) that
generate a minimal mean-torque periodic motion to ambulate at a specified
average speed over half a period.

This example highlights two points of interest that the other examples may not
have:

- Instance constraints are used to make the start state the same as the end
  state, for example :math:`q_b(t_0) = q_e(t_f)`, except for forward
  translation.
- The average speed is constrained in this variable time step solution by
  introducing an additional differential equation that, when integrated, gives
  the duration at :math:`\Delta_t(t)`, which can be used to calculate distance
  traveled with :math:`q_{ax}(t_f) = v_\textrm{avg} (t_f - t_0)` and used as a
  constraint.
- The parallel option is enabled because the equations of motion are on the
  large side. This speeds up the evaluation of the constraints and its Jacobian
  about 1.3X.

Import all necessary modules, functions, and classes:
"""
import os
import pprint
from opty import Problem
from opty.utils import f_minus_ma
from pygait2d import derive, simulate
from pygait2d.segment import time_symbol, contact_force
from symmeplot.matplotlib import Scene3D
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm

# %%
# Pick an average ambulation speed and the number of discretization nodes for
# the half period and define the time step as a variable :math:`h`.
speed = 1.3  # m/s
num_nodes = 40
h = sm.symbols('h', real=True, positive=True)
duration = (num_nodes - 1)*h

# %%
# Derive the equations of motion using gait2d.
symbolics = derive.derive_equations_of_motion()

mass_matrix = symbolics[0]
forcing_vector = symbolics[1]
constants = symbolics[3]
coordinates = symbolics[4]
speeds = symbolics[5]
specified = symbolics[6]

eom = f_minus_ma(mass_matrix, forcing_vector, coordinates + speeds)
eom.shape

# %%
# :math:`t_f - t_0` needs to be available to compute the average speed in the
# instance constraint, so add an extra differential equation that is the time
# derivative of the difference in time.
#
# .. math::
#
#    \Delta_t(t) = \int_{t_0}^{t} d\tau
#
delt = sm.Function('delt', real=True)(time_symbol)
eom = eom.col_join(sm.Matrix([delt.diff(time_symbol) - 1]))

states = coordinates + speeds + [delt]
num_states = len(states)

# %%
# The generalized coordinates are the hip lateral position :math:`q_{ax}` and
# veritcal position :math:`q_{ay}`, the trunk angle with respect to vertical
# :math:`q_a` and the relative joint angles:
#
# - right: hip (b), knee (c), ankle (d)
# - left: hip (e), knee (f), ankle (g)
#
# Each joint has a joint torque acting between the adjacent bodies.
qax, qay, qa, qb, qc, qd, qe, qf, qg = coordinates
uax, uay, ua, ub, uc, ud, ue, uf, ug = speeds
Fax, Fay, Ta, Tb, Tc, Td, Te, Tf, Tg = specified

# %%
# The constants are loaded from a file of realistic geometry, mass, inertia,
# and foot deformation properties of an adult human.
par_map = simulate.load_constants(constants, 'human-gait-constants.yml')
pprint.pprint(par_map)

# %%
# gait2d provides "hand of god" inputs to manipulate the trunk for some
# modeling purposes. Set these to zero.
traj_map = {
    Fax: np.zeros(num_nodes),
    Fay: np.zeros(num_nodes),
    Ta: np.zeros(num_nodes),
}

# %%
# Bound all the states to human realizable ranges.
#
# - The trunk should stay generally upright and be at a possible walking
#   height.
# - Only let the hip, knee, and ankle flex and extend to realistic limits.
# - Put a maximum on the peak torque values.
bounds = {
    h: (0.001, 0.1),
    delt: (0.0, 10.0),
    qax: (0.0, 10.0),
    qay: (0.5, 1.5),
    qa: np.deg2rad((-60.0, 60.0)),
    uax: (0.0, 10.0),
    uay: (-10.0, 10.0),
}
# hip
bounds.update({k: (-np.deg2rad(40.0), np.deg2rad(40.0))
               for k in [qb, qe]})
# knee
bounds.update({k: (-np.deg2rad(60.0), 0.0)
               for k in [qc, qf]})
# foot
bounds.update({k: (-np.deg2rad(30.0), np.deg2rad(30.0))
               for k in [qd, qg]})
# all rotational speeds
bounds.update({k: (-np.deg2rad(400.0), np.deg2rad(400.0))
               for k in [ua, ub, uc, ud, ue, uf, ug]})
# all joint torques
bounds.update({k: (-100.0, 100.0)
               for k in [Tb, Tc, Td, Te, Tf, Tg]})

# %%
# The average speed can be fixed by constraining the total distance traveled.
# To enforce a half period, set the right leg's angles at the initial time to
# be equal to the left leg's angles at the final time and vice versa. The same
# goes for the joint angular rates.
#
instance_constraints = (
    delt.func(0*h) - 0.0,
    qax.func(0*h) - 0.0,
    qax.func(duration) - speed*delt.func(duration),
    qay.func(0*h) - qay.func(duration),
    qa.func(0*h) - qa.func(duration),
    qb.func(0*h) - qe.func(duration),
    qc.func(0*h) - qf.func(duration),
    qd.func(0*h) - qg.func(duration),
    qe.func(0*h) - qb.func(duration),
    qf.func(0*h) - qc.func(duration),
    qg.func(0*h) - qd.func(duration),
    uax.func(0*h) - uax.func(duration),
    uay.func(0*h) - uay.func(duration),
    ua.func(0*h) - ua.func(duration),
    ub.func(0*h) - ue.func(duration),
    uc.func(0*h) - uf.func(duration),
    ud.func(0*h) - ug.func(duration),
    ue.func(0*h) - ub.func(duration),
    uf.func(0*h) - uc.func(duration),
    ug.func(0*h) - ud.func(duration),
)


# %%
# The objective is to minimize the mean of all joint torques.
def obj(free):
    """Minimize the sum of the squares of the control torques."""
    T, h = free[num_states*num_nodes:-1], free[-1]
    return h*np.sum(T**2)


def obj_grad(free):
    T, h = free[num_states*num_nodes:-1], free[-1]
    grad = np.zeros_like(free)
    grad[num_states*num_nodes:-1] = 2.0*h*T
    grad[-1] = np.sum(T**2)
    return grad


# %%
# Create an optimization problem and solve it.
prob = Problem(
    obj,
    obj_grad,
    eom,
    states,
    num_nodes,
    h,
    known_parameter_map=par_map,
    known_trajectory_map=traj_map,
    instance_constraints=instance_constraints,
    bounds=bounds,
    time_symbol=time_symbol,
    parallel=True,
)

# %%
# This loads a precomputed solution to save computation time. Delete the file
# to try one of the suggested initial guesses.
fname = f'human_gait_{num_nodes}_nodes_solution.csv'
if os.path.exists(fname):
    initial_guess = np.loadtxt(fname)
else:
    # choose one, comment others
    initial_guess = prob.lower_bound + (prob.upper_bound -
        prob.lower_bound)*np.random.random_sample(prob.num_free)
    initial_guess = 0.01*np.ones(prob.num_free)
    initial_guess = np.zeros(prob.num_free)

# %%
# Find the optimal solution and save it if it converges.
solution, info = prob.solve(initial_guess)

xs, rs, _, h_val = prob.parse_free(solution)
times = np.arange(0.0, num_nodes*h_val, h_val)
if info['status'] in (0, 1):
    np.savetxt(f'human_gait_{num_nodes}_nodes_solution.csv', solution,
               fmt='%.3f')


# %%
# Use symmeplot to make an animation of the motion.
def animate(fname='animation.gif'):

    ground, origin, segments = symbolics[8], symbolics[9], symbolics[10]
    trunk, rthigh, rshank, rfoot, lthigh, lshank, lfoot = segments

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    hip_proj = origin.locatenew('m', qax*ground.x)
    scene = Scene3D(ground, hip_proj, ax=ax)

    # creates the stick person
    scene.add_line([
        rshank.joint,
        rfoot.toe,
        rfoot.heel,
        rshank.joint,
        rthigh.joint,
        trunk.joint,
        trunk.mass_center,
        trunk.joint,
        lthigh.joint,
        lshank.joint,
        lfoot.heel,
        lfoot.toe,
        lshank.joint,
    ], color="k")

    # creates a moving ground (many points to deal with matplotlib limitation)
    scene.add_line([origin.locatenew('gl', s*ground.x)
                    for s in np.linspace(-2.0, 2.0)],
                   linestyle='--', color='tab:green', axlim_clip=True)

    # adds CoM and unit vectors for each body segment
    for seg in segments:
        scene.add_body(seg.rigid_body)

    # show ground reaction force vectors at the heels and toes, scaled to
    # visually reasonable length
    scene.add_vector(contact_force(rfoot.toe, ground, origin)/600.0,
                     rfoot.toe, color="tab:blue")
    scene.add_vector(contact_force(rfoot.heel, ground, origin)/600.0,
                     rfoot.heel, color="tab:blue")
    scene.add_vector(contact_force(lfoot.toe, ground, origin)/600.0,
                     lfoot.toe, color="tab:blue")
    scene.add_vector(contact_force(lfoot.heel, ground, origin)/600.0,
                     lfoot.heel, color="tab:blue")

    scene.lambdify_system(states + specified + constants)
    gait_cycle = np.vstack((
        xs,  # q, u shape(2n, N)
        np.zeros((3, len(times))),  # Fax, Fay, Ta (hand of god), shape(3, N)
        rs,  # r, shape(q, N)
        np.repeat(np.atleast_2d(np.array(list(par_map.values()))).T,
                  len(times), axis=1),  # p, shape(r, N)
    ))
    scene.evaluate_system(*gait_cycle[:, 0])

    scene.axes.set_proj_type("ortho")
    scene.axes.view_init(90, -90, 0)
    scene.plot()

    ax.set_xlim((-0.8, 0.8))
    ax.set_ylim((-0.2, 1.4))
    ax.set_aspect('equal')

    ani = scene.animate(lambda i: gait_cycle[:, i], frames=len(times),
                        interval=h_val*1000)
    ani.save(fname, fps=int(1/h_val))

    return ani


animation = animate('human-gait-earth.gif')

# %%
# Now see what the solution looks like in the Moon's gravitational field.
g = constants[0]
par_map[g] = 1.625  # m/s**2
pprint.pprint(par_map)

# %%
# Use the Earth solution as an initial guess.
initial_guess = np.loadtxt(fname)

# %%
# Create an optimization problem and solve it.
prob = Problem(
    obj,
    obj_grad,
    eom,
    states,
    num_nodes,
    h,
    known_parameter_map=par_map,
    known_trajectory_map=traj_map,
    instance_constraints=instance_constraints,
    bounds=bounds,
    time_symbol=time_symbol,
    parallel=True,
)

solution, info = prob.solve(initial_guess)

# %%
# Animate the second solution.
xs, rs, _, h_val = prob.parse_free(solution)
times = np.arange(0.0, num_nodes*h_val, h_val)

animation = animate('human-gait-moon.gif')

plt.show()

# %%
# References
# ----------
#
# .. [Ackermann2010] Ackermann, M., & van den Bogert, A. J. (2010). Optimality
#    principles for model-based prediction of human gait. Journal of
#    Biomechanics, 43(6), 1055–1060.
#    https://doi.org/10.1016/j.jbiomech.2009.12.012
#
# Footnotes
# ---------
#
# .. [1] The 2010 Ackermann and van den Bogert solution was the original target
#    problem opty was written to solve, with an aim to extend it to parameter
#    identification of closed loop control walking. For various reasons, this
#    example was not added until 2025, 10 years after the example was first
#    proposed.
