"""
Hilly Ride
==========

Simulation of a particle subject to the force or gravity and air drag as it
traverses and elevation profile with a specified slope to reach the end in
minimal time.

Objectives
----------

- Demonstrate generating a known trajectory via a numerical function.
- Show how interpolation can be used to generated a specified input.

"""
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from opty import Problem

# %%
# Define the variables and equations of motion.
#
# - :math:`m`: particle mass
# - :math:`g`: acceleration due to gravity
# - :math:`h`: time step
# - :math:`s(t)`: distance traveled along elevation profile
# - :math:`v(t)`: longitudinal speed
# - :math:`x(t)`: horizontal coordinate
# - :math:`y(t)`: vertical coordinate
# - :math:`p(t)`: propulsion power
# - :math:`theta(t)`: slope angle
m, g, h = sm.symbols('m, g, h', real=True, nonnegative=True)
s, v, x, y, p, theta = me.dynamicsymbols('s, v, x, y, p, theta', real=True)

states = (x, y, s, v)

eom = sm.Matrix([
    x.diff() - v*sm.cos(theta),
    y.diff() - v*sm.sin(theta),
    s.diff() - v,
    m*v.diff() - p/v + m*g*sm.sin(theta) + v**2/3,
])

N = 101

# %%
# The elevation profile is often derived from measurements of a road surface.
# If a series of elevation values at specified linear distances are available,
# the slope is then also a function of the linear distances. The following code
# creates an elevation profile that simulates having a smooth slope.
# :math:`\theta(x(t))`.
amp = 20.0
omega = 2*np.pi/500.0  # one period every 500 meters
xp = np.linspace(-250.0, 1250.0, num=1501)
yp = amp*np.sin(omega*xp)
thetap = np.atan(amp*omega*np.cos(omega*xp))
dthetadx = -amp*omega**2*np.sin(omega*xp)/(amp**2*omega**2*np.cos(omega*xp)**2
                                           + 1)
fig, axes = plt.subplots(3, sharex=True)
axes[0].plot(xp, yp)
axes[0].set_ylabel(r'$y$ [m]')
axes[1].plot(xp, np.rad2deg(thetap))
axes[1].set_ylabel(r'$\theta$ [deg]')
axes[2].plot(xp, np.rad2deg(dthetadx))
axes[2].set_ylabel(r'$\frac{d\theta}{dx}$ [deg/m]')
axes[2].set_xlabel(r'$x$ [m]')


# %%
# The following function outputs the slope at all values of x using
# interpolation. The only input to this function should be the optimization
# free vector and the output should be an array of values for :math:`\theta`,
# one value for each node.
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


# %%
# Minimize the time to reach the final distance traveled :math:`s(t_f)` when
# starting from a standstill. The time step is the last value in free.
def obj(free):
    return free[-1]


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad


t0, tf = 0*h, (N - 1)*h
sf = 1000.0  # meters

instance_constraint = (
    x.func(t0),
    y.func(t0),
    s.func(t0),
    v.func(t0),
    s.func(tf) - sf,
)

# %%
# Limit the power and make sure the time step is positive.
bounds = {
    h: (0.0, 10.0),
    p: (0.0, 1000.0),
}

# %%
# The slope angle :math:`\theta` is set as a known trajectory and the
# ``calc_theta`` function is provided to generate the array of :math:`\theta`
# values dynamically during the optimization iterations.
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

# %%
# Provide linear initial guesses for each variable.
initial_guess = np.random.random(prob.num_free)
initial_guess[0*N:1*N] = np.linspace(0.0, sf, num=N)  # x
initial_guess[1*N:2*N] = np.zeros(N)  # y
initial_guess[2*N:3*N] = np.linspace(0.0, sf, num=N)  # s
initial_guess[3*N:4*N] = 10.0*np.ones(N)  # v
initial_guess[4*N:5*N] = 500.0*np.ones(N)  # p
initial_guess[-1] = 0.1  # h

_ = prob.plot_trajectories(initial_guess)

# %%
# Solve the probem.
solution, info = prob.solve(initial_guess)

_ = prob.plot_objective_value()

# %%
# Constraint violations:
_ = prob.plot_constraint_violations(solution)

# %%
# State and input trajectories:
_ = prob.plot_trajectories(solution)

# %%
xs, rs, ps, dh = prob.parse_free(solution)

fig, ax = plt.subplots()
ax.plot(xs[0], xs[1])
ax.set_aspect('equal')
ax.set_ylabel(r'$y$ [m]')
ax.set_xlabel(r'$x$ [m]')

plt.show()
