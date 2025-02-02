"""This example replicates some of the work presented in Ackermann and van den
Bogert 2010.

pygait2d and its dependencies must be installed first to run this example::

    conda install sympy pydy pyyaml cython pip setuptools symmeplot
    python -m pip install --no-deps --no-build-isolation git+https://github.com/csu-hmc/gait2d

"""

import sympy as sm
import numpy as np
from pygait2d import derive, simulate
from pygait2d.segment import time_symbol
from opty import Problem, parse_free
from opty.utils import f_minus_ma

distance = 3.0  # m
num_nodes = 60
h = sm.symbols('h', real=True, positive=True)
duration = (num_nodes - 1)*h

symbolics = derive.derive_equations_of_motion()

mass_matrix = symbolics[0]
forcing_vector = symbolics[1]
kane = symbolics[2]
constants = symbolics[3]
coordinates = symbolics[4]
speeds = symbolics[5]
states = coordinates + speeds
specified = symbolics[6]

num_states = len(states)

eom = f_minus_ma(mass_matrix, forcing_vector, coordinates + speeds)

# right: b, c, d
# left: e, f, g
qax, qay, qa, qb, qc, qd, qe, qf, qg = coordinates
uax, uay, ua, ub, uc, ud, ue, uf, ug = speeds
Fax, Fay, Ta, Tb, Tc, Td, Te, Tf, Tg = specified

par_map = simulate.load_constants(constants, 'example_constants.yml')

# Hand of god is nothing.
traj_map = {
    Fax: np.zeros(num_nodes),
    Fay: np.zeros(num_nodes),
    Ta: np.zeros(num_nodes),
}

bounds = {
    h: (0.001, 0.1),
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
bounds.update({k: (-6.0, 6.0) for k in [ua, ub, uc, ud, ue, uf, ug]})  # ~200 deg/s
bounds.update({k: (-1200.0, 1200.0) for k in [Tb, Tc, Td, Te, Tf, Tg]})

# Specify the symbolic instance constraints, i.e. initial and end
# conditions.
uneval_states = [s.__class__ for s in states]
(qax, qay, qa, qb, qc, qd, qe, qf, qg, uax, uay, ua, ub, uc, ud, ue, uf, ug) = uneval_states

instance_constraints = (
    # start at standstill
    qax(0*h),
    qay(0*h) - 0.953,
    qa(0*h) - 0.0,
    qb(0*h) - 0.0,
    qc(0*h) - 0.0,
    qd(0*h) - 0.0,
    qe(0*h) - 0.0,
    qf(0*h) - 0.0,
    qg(0*h) - 0.0,
    uax(0*h) - 0.0,
    uay(0*h) - 0.0,
    ua(0*h) - 0.0,
    ub(0*h) - 0.0,
    uc(0*h) - 0.0,
    ud(0*h) - 0.0,
    ue(0*h) - 0.0,
    uf(0*h) - 0.0,
    ug(0*h) - 0.0,
    # after distance traveled, back at standstill
    qax(duration) - distance,
    qay(duration) - 0.953,
    qa(duration) - 0.0,
    qb(duration) - 0.0,
    qc(duration) - 0.0,
    qd(duration) - 0.0,
    qe(duration) - 0.0,
    qf(duration) - 0.0,
    qg(duration) - 0.0,
    uax(duration) - 0.0,
    uay(duration) - 0.0,
    ua(duration) - 0.0,
    ub(duration) - 0.0,
    uc(duration) - 0.0,
    ud(duration) - 0.0,
    ue(duration) - 0.0,
    uf(duration) - 0.0,
    ug(duration) - 0.0,
    # TODO : need support for including h outside of a (make opty issue)
    # Function argument.
    #qax(duration) - speed * duration,
)


# Specify the objective function and it's gradient.
def obj(free):
    """Minimize the sum of the squares of the control torque."""
    T, h = free[num_states*num_nodes:-1], free[-1]
    return h*np.sum(T**2)


def obj_grad(free):
    T, h = free[num_states*num_nodes:-1], free[-1]
    grad = np.zeros_like(free)
    grad[num_states*num_nodes:-1] = 2.0*h*T
    grad[-1] = np.sum(T**2)
    return grad


# Create an optimization problem.
prob = Problem(obj, obj_grad, eom, states, num_nodes, h,
               known_parameter_map=par_map,
               known_trajectory_map=traj_map,
               instance_constraints=instance_constraints,
               bounds=bounds,
               time_symbol=time_symbol,
               parallel=True)
               #tmp_dir='ufunc')

# Use a random positive initial guess.
#initial_guess = prob.lower_bound + (prob.upper_bound - prob.lower_bound) * np.random.randn(prob.num_free)
#initial_guess = 0.01*np.ones(prob.num_free)
initial_guess = np.zeros(prob.num_free)

# Find the optimal solution.
solution, info = prob.solve(initial_guess)

# TODO : ps is empy, not sure why
state_vals, rs, ps, h_val = parse_free(solution, num_states, len(specified) -
                                       len(traj_map), num_nodes,
                                       variable_duration=True)
np.savez('solution', solution=solution, x=state_vals, h=h_val, n=num_nodes)


def animate():

    import matplotlib.pyplot as plt
    from symmeplot.matplotlib import Scene3D

    ground, origin, segments = symbolics[8], symbolics[9], symbolics[10]
    trunk, rthigh, rshank, rfoot, lthigh, lshank, lfoot = segments

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    scene = Scene3D(ground, origin, ax=ax, scale=1.0)

    scene.add_line([
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
    ], color="k")
    #scene.add_point(nd, color='C0')

    for seg in segments:
        scene.add_body(seg.rigid_body)

    # TODO : Add ground reaction force vectors
    #scene.add_vector(A.x, nd, color="black")

    scene.lambdify_system(coordinates + speeds + specified + constants)
    scene.evaluate_system(*np.hstack((state_vals[:9, 0], state_vals[9:, 0], np.zeros(3), rs[:, 0], np.array(list(par_map.values())))))

    # this is only in dev version of symmeplot
    #scene.as_orthogonal_projection_plot()
    scene.axes.set_proj_type("ortho")
    scene.axes.view_init(90, -90, 0)
    scene.plot()

    ax.set_xlim((-0.5, distance + 0.5))
    #ax.set_ylim((-1.0, 1.0))
    #ax.set_zlim((1.0, -1.0))

    times = np.linspace(0.0, h_val*num_nodes, num=num_nodes)

    slow_factor = 3  # int
    ani = scene.animate(lambda i: np.hstack((state_vals[:9, i],
                                             state_vals[9:, i],
                                             np.zeros(3),
                                             rs[:, i],
                                             np.array(list(par_map.values())))),
                        frames=len(times)) #, interval=slow_factor/FPS*1000)
    ani.save("animation.gif") #, fps=FPS//slow_factor)

    return ani
