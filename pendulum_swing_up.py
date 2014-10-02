"""This solves the simple pendulum swing up problem presented here:

    http://hmc.csuohio.edu/resources/human-motion-seminar-jan-23-2014

A simple pendulum is controlled by a torque at its joint. The goal is to
swing the pendulum from its rest equilibrium to a target angle by minimizing
the energy used to do so.

"""

from collections import OrderedDict

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from direct_collocation import ConstraintCollocator, Problem

target_angle = 5 * np.pi
duration = 5.0
num_nodes = 100
save_animation = False

interval_value = duration / (num_nodes - 1)

I, m, g, d, t = sym.symbols('I, m, g, d, t')
theta, omega, T = [f(t) for f in sym.symbols('theta, omega, T',
                                             cls=sym.Function)]
state_symbols = (theta, omega)
constant_symbols = (I, m, g, d)
specified_symbols = (T,)

eom = sym.Matrix([theta.diff() - omega,
                  I * omega.diff() + m * g * d * sym.sin(theta) - T])

par_map = OrderedDict(zip(constant_symbols, (1.0, 1.0, 9.81, 1.0)))


def obj(free):
    T = free[2 * num_nodes:]
    return np.sum(T**2)


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[2 * num_nodes:] = 2.0 * interval_value * free[2 * num_nodes:]
    return grad

theta, omega = sym.symbols('theta, omega', cls=sym.Function)
instance_constraints = (theta(0.0),
                        theta(duration) - target_angle,
                        omega(0.0),
                        omega(duration))

con_col = ConstraintCollocator(eom, state_symbols, num_nodes,
                               interval_value, known_parameter_map=par_map,
                               instance_constraints=instance_constraints)

prob = Problem(con_col.num_free,
               con_col.num_constraints,
               obj,
               obj_grad,
               con_col.generate_constraint_function(),
               con_col.generate_jacobian_function(),
               con_col.jacobian_indices)

# Known solution as initial guess.
initial_guess = np.random.randn(con_col.num_free)

# Find the optimal solution.
solution, info = prob.solve(initial_guess)

time = np.linspace(0.0, duration, num=num_nodes)

angle = solution[:num_nodes]
rate = solution[num_nodes:2 * num_nodes]
torque = solution[2 * num_nodes:]
con_violations = prob.con(solution)

fig, axes = plt.subplots(5)

axes[0].plot(time, angle)
axes[0].set_ylabel('Angle [rad]')
axes[1].plot(time, rate)
axes[1].set_ylabel('Angular Rate [rad]')
axes[2].plot(time, torque)
axes[2].set_ylabel('Torque [Nm]')
axes[2].set_xlabel('Time [S]')
axes[3].plot(con_violations)
axes[4].plot(prob.obj_value)

# Animation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = {:0.1f}s'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    x = [0, par_map[d] * np.sin(angle[i])]
    y = [0, -par_map[d] * np.cos(angle[i])]

    line.set_data(x, y)
    time_text.set_text(time_template.format(i * interval_value))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(time)),
                              interval=25, blit=True, init_func=init)

if save_animation:
    ani.save('pendulum_swing_up.mp4', writer='avconv', fps=15)

plt.show()
