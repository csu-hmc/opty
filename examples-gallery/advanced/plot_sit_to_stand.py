r"""
Sit to Stand
============

The optimal control goal is to find the joint torques (hip, knee, ankle) that
generate a minimal mean-torque for a person to stand from a seated position.
gait2d provides a joint torque driven 2D bipedal human dynamical model with
seven body segments (trunk, thighs, shanks, feet) and foot-ground contact
forces based on the description in [Ackermann2010]_ suitable for this
simulation.

.. note::

   pygait2d and symmeplot and their dependencies must be installed first to run
   this example. Note that pygait2d has not been released to PyPi or Conda
   Forge::

      conda install cython pip pydy pyyaml setuptools symmeplot sympy
      python -m pip install --no-deps --no-build-isolation git+https://github.com/csu-hmc/gait2d

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
# Pick the number of discretization nodes and define the time step as a
# variable :math:`h`.
num_nodes = 60
h = sm.symbols('h', real=True, positive=True)
duration = (num_nodes - 1)*h

# %%
# Derive the equations of motion using gait2d, including a force acting on the
# hip joint from the seat surface above the ground.
symbolics = derive.derive_equations_of_motion(seat_force=True)

mass_matrix = symbolics[0]
forcing_vector = symbolics[1]
constants = symbolics[3]
coordinates = symbolics[4]
speeds = symbolics[5]
specified = symbolics[6]
states = coordinates + speeds
num_states = len(states)

eom = f_minus_ma(mass_matrix, forcing_vector, states)
eom.shape

# %%
# The equations of motion have this many mathematical operations:
sm.count_ops(eom)

# %%
# The generalized coordinates are the hip lateral position :math:`q_{ax}` and
# vertical position :math:`q_{ay}`, the trunk angle with respect to vertical
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
# The model constants describe the geometry, mass, and inertia of the human. We
# will need some geometry for defining the seating position:
#
# - ``ya``: trunk hip to mass center length
# - ``lb``: thigh length
# - ``lc``: shank length
# - ``fyd``: foot depth
(g, ma, ia, xa, ya, mb, ib, lb, xb, yb, mc, ic, lc, xc, yc, md, id_, xd, yd,
 hxd, txd, fyd, me, ie, le, xe, ye, mf, if_, lf, xf, yf, mg, ig, xg, yg, hxg,
 txg, fyg, kc, cc, mu, vs) = constants

# %%
# The constant values are loaded from a file of realistic geometry, mass,
# inertia, and foot deformation properties of an adult human.
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
# - The trunk should stay generally upright but can lean back and forth.
# - Only let the hip, knee, and ankle flex and extend to realistic limits.
# - Put a maximum on the peak joint torque values.
bounds = {
    h: (0.001, 0.1),
    qax: (0.0, 5.0),
    qay: (0.2, 1.5),
    qa: np.deg2rad((-60.0, 90.0)),
    uax: (-10.0, 10.0),
    uay: (-10.0, 10.0),
}
# hip
bounds.update({k: (-np.deg2rad(60.0), np.deg2rad(150.0))
               for k in [qb, qe]})
# knee
bounds.update({k: (-np.deg2rad(150.0), 0.0)
               for k in [qc, qf]})
# foot
bounds.update({k: (-np.deg2rad(30.0), np.deg2rad(30.0))
               for k in [qd, qg]})
# all rotational speeds
bounds.update({k: (-np.deg2rad(400.0), np.deg2rad(400.0))
               for k in [ua, ub, uc, ud, ue, uf, ug]})
# all joint torques
bounds.update({k: (-500.0, 500.0)
               for k in [Tb, Tc, Td, Te, Tf, Tg]})

# %%
# Set the configuration to be seated at the start and standing at the finish.
# Subtract a bit from the final height because the feet compress into the
# ground.
instance_constraints = (
    # start seated
    qax.func(0*h) - 0.0,
    qay.func(0*h) - (-fyd + lc),
    qa.func(0*h) - 0.0,
    qb.func(0*h) - np.deg2rad(90.0),
    qc.func(0*h) + np.deg2rad(90.0),
    qd.func(0*h) - 0.0,
    qe.func(0*h) - np.deg2rad(90.0),
    qf.func(0*h) + np.deg2rad(90.0),
    qg.func(0*h) - 0.0,
    # end standing
    qax.func(duration) - lb,
    qay.func(duration) - (-fyd + lb + lc - 0.02),
    qa.func(duration) - 0.0,
    qb.func(duration) - 0.0,
    qc.func(duration) - 0.0,
    qd.func(duration) - 0.0,
    qe.func(duration) - 0.0,
    qf.func(duration) - 0.0,
    qg.func(duration) - 0.0,
    # stationary at start and end
    uax.func(0*h) - 0.0,
    uay.func(0*h) - 0.0,
    ua.func(0*h) - 0.0,
    ub.func(0*h) - 0.0,
    uc.func(0*h) - 0.0,
    ud.func(0*h) - 0.0,
    ue.func(0*h) - 0.0,
    uf.func(0*h) - 0.0,
    ug.func(0*h) - 0.0,
    uax.func(duration) - 0.0,
    uay.func(duration) - 0.0,
    ua.func(duration) - 0.0,
    ub.func(duration) - 0.0,
    uc.func(duration) - 0.0,
    ud.func(duration) - 0.0,
    ue.func(duration) - 0.0,
    uf.func(duration) - 0.0,
    ug.func(duration) - 0.0,
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
# to try the suggested initial guess.
fname = f'human_sit_to_stand_{num_nodes}_nodes_solution.csv'
if os.path.exists(fname):
    initial_guess = np.loadtxt(fname)
else:
    # choose one, comment others
    initial_guess = np.zeros(prob.num_free)
    # h
    initial_guess[-1] = 2.0/num_nodes
    # qax
    initial_guess[0*num_nodes:1*num_nodes] = np.linspace(
        0.0, par_map[lb], num=num_nodes)
    # qay
    initial_guess[1*num_nodes:2*num_nodes] = np.linspace(
        (-par_map[fyd] + par_map[lc]),
        (-par_map[fyd] + par_map[lb] + par_map[lc]),
        num=num_nodes)
    # qb
    initial_guess[3*num_nodes:4*num_nodes] = np.linspace(
        np.deg2rad(90.0), 0.0, num=num_nodes)
    # qc
    initial_guess[4*num_nodes:5*num_nodes] = np.linspace(
        np.deg2rad(-90.0), 0.0, num=num_nodes)
    # qe
    initial_guess[6*num_nodes:7*num_nodes] = np.linspace(
        np.deg2rad(90.0), 0.0, num=num_nodes)
    # qf
    initial_guess[7*num_nodes:8*num_nodes] = np.linspace(
        np.deg2rad(-90.0), 0.0, num=num_nodes)

# %%
# Find the optimal solution and save it if it converges.
solution, info = prob.solve(initial_guess)
if info['status'] in (0, 1):
    np.savetxt(fname, solution, fmt='%.2f')

# %%
# Plot the solution.
_ = prob.plot_trajectories(solution)

# %%
# Plot the constraint violations of the solution.
_ = prob.plot_constraint_violations(solution)

# %%
# Use symmeplot to make an animation of the motion.
# sphinx_gallery_thumbnail_number = 3
xs, rs, _, h_val = prob.parse_free(solution)
times = np.linspace(0.0, (num_nodes - 1)*h_val, num=num_nodes)


def animate(fname='animation.gif'):

    ground, origin, segments = symbolics[8], symbolics[9], symbolics[10]
    trunk, rthigh, rshank, rfoot, lthigh, lshank, lfoot = segments

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    scene = Scene3D(ground, origin, ax=ax)

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

    # ground
    scene.add_line([
        origin.locatenew('left', -0.8*ground.x),
        origin.locatenew('right', 0.8*ground.x),
    ], color='black')

    # seat
    seat_level = origin.locatenew('seat', (segments[2].length_symbol -
                                           segments[3].foot_depth)*ground.y)

    scene.add_line([
        seat_level.locatenew('top', -0.2*ground.x + 0.5*ground.y),
        seat_level.locatenew('left', -0.2*ground.x),
        seat_level.locatenew('right', 0.2*ground.x),
    ], color='black', linewidth=4)

    # adds CoM and unit vectors for each body segment
    for seg in segments:
        scene.add_body(seg.rigid_body)

    # show ground reaction force vectors at the heels, toes, and hip, scaled to
    # visually reasonable length
    scene.add_vector(contact_force(trunk.joint, ground, seat_level)/600.0,
                     trunk.joint, color="tab:blue")
    scene.add_vector(contact_force(rfoot.toe, ground, origin)/600.0,
                     rfoot.toe, color="tab:blue")
    scene.add_vector(contact_force(rfoot.heel, ground, origin)/600.0,
                     rfoot.heel, color="tab:blue")
    scene.add_vector(contact_force(lfoot.toe, ground, origin)/600.0,
                     lfoot.toe, color="tab:blue")
    scene.add_vector(contact_force(lfoot.heel, ground, origin)/600.0,
                     lfoot.heel, color="tab:blue")

    scene.lambdify_system(states + specified + constants)
    sim_data = np.vstack((
        xs,  # q, u shape(2n, N)
        np.zeros((3, len(times))),  # Fax, Fay, Ta (hand of god), shape(3, N)
        rs,  # r, shape(q, N)
        np.repeat(np.atleast_2d(np.array(list(par_map.values()))).T,
                  len(times), axis=1),  # p, shape(r, N)
    ))
    scene.evaluate_system(*sim_data[:, 0])

    scene.axes.set_proj_type("ortho")
    scene.axes.view_init(90, -90, 0)
    scene.plot()

    ani = scene.animate(lambda i: sim_data[:, i], frames=len(times),
                        interval=h_val*1000)
    ani.save(fname, fps=int(1/h_val))

    return ani


animation = animate('human-sit-to-stand.gif')

plt.show()
