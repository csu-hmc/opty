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
x, v, f, theta = me.dynamicsymbols('x, v, f, theta', real=True)

states = (x, v)

eom = sm.Matrix([
    x.diff() - v,
    m*v.diff() - f + m*g*sm.cos(theta),
])

N = 101

xp = np.linspace(0.0, 100.0, num=N)
yp = 10.0*np.sin(xp)
thetap = 40.0*np.cos(0.08*xp)


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
    v.func(t0),
    x.func(tf) - xp[-1],
)

bounds = {
    h: (0.0, 10.0),
    f: (-1000.0, 1000.0),
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

initial_guess = 20.0*np.ones(prob.num_free)
solution, info = prob.solve(initial_guess)

_ = prob.plot_objective_value()

# %%
# State and input trajectories:
_ = prob.plot_trajectories(solution)

# %%
# Constraint violations:
_ = prob.plot_constraint_violations(solution)

plt.show()
