"""
Hilly Ride
==========

"""
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from opty import Problem

m, g, h = sm.symbols('m, g, h', real=True, nonnegative=True)
s, v, x, y, p, theta = me.dynamicsymbols('s, v, x, y, p, theta', real=True)

states = (x, y, s, v)

eom = sm.Matrix([
    x.diff() - v*sm.cos(theta),
    y.diff() - v*sm.sin(theta),
    s.diff() - v,
    m*v.diff() - p/v + m*g*sm.sin(theta) + 10*v,
])

N = 1001

xp = np.linspace(-10000.0, 10000.0, num=40001)
amp = 10.0
omega = 2*np.pi/500.0  # one period every 500 meters
yp = amp*np.sin(omega*xp)
thetap = np.atan(amp*omega*np.cos(omega*xp))

fig, axes = plt.subplots(2)
axes[0].plot(xp, yp)
axes[1].plot(xp, np.rad2deg(thetap))


def calc_theta(free):
    """
    Parameters
    ==========
    free : ndarray, shape(nN + qN + r + s, )

    Returns
    =======
    theta : ndarray, shape(N, )

    """
    x = free[0:N]
    return np.interp(x, xp, thetap)


def obj(free):
    return free[-1]


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad


t0, tf = 0*h, (N - 1)*h

instance_constraint = (
    x.func(t0),
    y.func(t0),
    s.func(t0),
    v.func(t0),
    s.func(tf) - 625.0,
    #v.func(tf),
)

bounds = {
    h: (0.0, 10.0),
    p: (-200.0, 1000.0),
}

prob = Problem(
    obj,
    obj_grad,
    eom,
    states,
    N,
    h,
    known_parameter_map={m: 100.0, g: 9.81},
    known_trajectory_map={theta: calc_theta},
    time_symbol=me.dynamicsymbols._t,
    instance_constraints=instance_constraint,
    bounds=bounds,
)

initial_guess = np.random.random(prob.num_free)
initial_guess = 0.01*np.ones(prob.num_free)
solution, info = prob.solve(initial_guess)

_ = prob.plot_objective_value()

# %%
# State and input trajectories:
_ = prob.plot_trajectories(solution)

# %%
# Constraint violations:
_ = prob.plot_constraint_violations(solution)

xs, rs, ps, dh = prob.parse_free(solution)

fig, ax = plt.subplots()
ax.plot(xs[0], xs[1])
ax.set_aspect('equal')

plt.show()
