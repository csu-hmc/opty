"""
Human Gait
==========

This example replicates a similar solution as shown in [Ackermann2010]_ using
joint torques as inputs instead of muscle activations.

pygait2d and its dependencies must be installed first to run this example::

    conda install cython pip pydy pyyaml setuptools symmeplot sympy
    python -m pip install --no-deps --no-build-isolation git+https://github.com/csu-hmc/gait2d

gait2d provides a joint torque driven 2d bipedal human dynamical model with
seven body segments (trunk, thighs, shanks, feet) and foot-ground contact
forces based on the description in [Ackermann2010]_.

The optimal control goal is to find the joint torques (hip, knee, ankle) that
generate a minimal mean-torque periodic motion to ambulate at an average speed
over half a period.

.. [Ackermann2010] Ackermann, M., & van den Bogert, A. J. (2010). Optimality
   principles for model-based prediction of human gait. Journal of
   Biomechanics, 43(6), 1055–1060.
   https://doi.org/10.1016/j.jbiomech.2009.12.012

"""
import os
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
# the half period.
speed = 0.8  # m/s
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
states = coordinates + speeds
specified = symbolics[6]

num_states = len(states)

eom = f_minus_ma(mass_matrix, forcing_vector, coordinates + speeds)

# We need to have :math:`t_f - t_0` available to compute the average speed in
# the instance constraint, so add an extra differential equation that is the
# time derivative of the difference in time.
#
# ..math::
#
#   \Delta_t(t) = \int_{t_0}^{t} d\tau
#
delt = sm.Function('delt', real=True)(time_symbol)
eom = eom.col_join(sm.Matrix([delt.diff(time_symbol) - 1]))

# %%
# The generalized coordinates are the hip lateral position (qax) and veritcal
# position (qay), the trunk angle with respect to vertical (qa) and the
# relative joint angles:
#
# - right: hip (b), knee (c), ankle (d)
# - left: hip (e), knee (f), ankle (g)
#
# Each joint has a joint torque acting between the adjacent bodies.
qax, qay, qa, qb, qc, qd, qe, qf, qg = coordinates
uax, uay, ua, ub, uc, ud, ue, uf, ug = speeds
Fax, Fay, Ta, Tb, Tc, Td, Te, Tf, Tg = specified

# The constants are loaded from a file of reasonably realistic geometry, mass,
# inertia, and foot deformation properties of an adult human.
par_map = simulate.load_constants(constants, 'example_constants.yml')

# %%
# gait2d provides "hand of god" inputs to manipulate the trunk for some
# modeling purposes. Set these to zero.
traj_map = {
    Fax: np.zeros(num_nodes),
    Fay: np.zeros(num_nodes),
    Ta: np.zeros(num_nodes),
}

# %%
#
bounds = {
    h: (0.001, 0.1),
    delt: (0.0, 10.0),
    qax: (0.0, 10.0),
    qay: (0.5, 1.5),
    qa: (-np.pi/3.0, np.pi/3.0),  # +/- 60 deg
    uax: (0.0, 10.0),
    uay: (-10.0, 10.0),
}
# hip
bounds.update({k: (-np.deg2rad(40.0), np.deg2rad(40.0)) for k in [qb, qe]})
# knee
bounds.update({k: (-np.deg2rad(60.0), 0.0) for k in [qc, qf]})
# foot
bounds.update({k: (-np.deg2rad(30.0), np.deg2rad(30.0)) for k in [qd, qg]})
bounds.update({k: (-np.deg2rad(400.0), np.deg2rad(400.0))
               for k in [ua, ub, uc, ud, ue, uf, ug]})
bounds.update({k: (-1200.0, 1200.0) for k in [Tb, Tc, Td, Te, Tf, Tg]})

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
    # TODO : there are only 6 joint torques but this is pulling more than that,
    # needs correction, could use parse free.
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
    states + [delt],  # add delt as a state
    num_nodes,
    h,
    known_parameter_map=par_map,
    known_trajectory_map=traj_map,
    instance_constraints=instance_constraints,
    bounds=bounds,
    time_symbol=time_symbol,
    parallel=True,
)

# Use a random positive initial guess.
fname = f'solution-{num_nodes}-nodes.npz'
if os.path.exists(fname):
    initial_guess = np.load(fname)['solution']
else:
    #initial_guess = prob.lower_bound + (prob.upper_bound -
        #prob.lower_bound)*np.random.random_sample(prob.num_free)
    #initial_guess = -0.01*np.ones(prob.num_free)
    initial_guess = np.zeros(prob.num_free)

# %%
# Find the optimal solution.
solution, info = prob.solve(initial_guess)

state_vals, rs, _, h_val = prob.parse_free(solution)
    #solution, num_states + 1, len(specified) - len(traj_map), num_nodes,
    #variable_duration=True)
if info['status'] in (0, 1):
    np.savez(f'solution-{num_nodes}-nodes', solution=solution, x=state_vals,
             h=h_val, n=num_nodes)


def animate():

    ground, origin, segments = symbolics[8], symbolics[9], symbolics[10]
    trunk, rthigh, rshank, rfoot, lthigh, lshank, lfoot = segments

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    #scene = Scene3D(ground, origin, ax=ax, scale=1.0)
    scene = Scene3D(ground, origin.locatenew('m', qax*ground.x), ax=ax,
                    scale=1.0)

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
    #scene.add_line([
        #origin.locatenew('m', -1*ground.x),
        #origin.locatenew('m', 3*ground.x),
    #], linestyle='--')

    for seg in segments:
        scene.add_body(seg.rigid_body)

    scene.add_vector(contact_force(rfoot.toe, ground, origin)/600.0, rfoot.toe,
                     color="tab:green")
    scene.add_vector(contact_force(rfoot.heel, ground, origin)/600.0,
                     rfoot.heel, color="tab:green")
    scene.add_vector(contact_force(lfoot.toe, ground, origin)/600.0, lfoot.toe,
                     color="tab:green")
    scene.add_vector(contact_force(lfoot.heel, ground, origin)/600.0,
                     lfoot.heel, color="tab:green")

    scene.lambdify_system(coordinates + speeds + specified + constants)
    scene.evaluate_system(*np.hstack((state_vals[:9, 0],
                                      state_vals[9:18, 0],
                                      np.zeros(3),
                                      rs[:, 0],
                                      np.array(list(par_map.values())))))

    scene.axes.set_proj_type("ortho")
    scene.axes.view_init(90, -90, 0)
    scene.plot()

    #ax.set_xlim((-0.5, state_vals[0].max() + 0.5))
    ax.set_xlim((-1.0, 1.0))
    ax.set_aspect('equal')

    #times = np.linspace(0.0, h_val*num_nodes, num=num_nodes)
    times = np.arange(0.0, num_nodes*h_val, h_val)

    slow_factor = 3  # int
    right = state_vals[:9, :]
    left = right.copy()
    left[0, :] += right[0, -1]
    ani = scene.animate(lambda i: np.hstack((right[:, i] if i < num_nodes else left[:, i - num_nodes],
                                             state_vals[9:18, 0],
                                             np.zeros(3),
                                             rs[:, 0],
                                             np.array(list(par_map.values())))),
                        frames=2*len(times))  #, interval=slow_factor/FPS*1000)
    ani.save("animation.gif") #, fps=FPS//slow_factor)

    return ani
