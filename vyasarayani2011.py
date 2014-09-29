"""This example is taken from the following paper:

Vyasarayani, Chandrika P., Thomas Uchida, Ashwin Carvalho, and John McPhee.
"Parameter Identification in Dynamic Systems Using the Homotopy Optimization
Approach". Multibody System Dynamics 26, no. 4 (2011): 411-24.

In Section 3.1 there is a simple example of a pendulum identification.

"""

import numpy as np
import sympy as sym
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from direct_collocation import ConstraintCollocator, Problem

# Specify the symbolic equations of motion: y' = f(y, t).
p, t = sym.symbols('p, t')
y1, y2 = [f(t) for f in sym.symbols('y1, y2', cls=sym.Function)]
y = sym.Matrix([y1, y2])
f = sym.Matrix([y2, -p * sym.sin(y1)])
eom = y.diff(t) - f

# Generate some data by simulating the equations of motion.
num_nodes = 5000
duration = 50.0
num_states = len(y)
p_val = 10.0
y0 = [np.pi / 6.0, 0.0]
time = np.linspace(0.0, duration, num=num_nodes)
interval = duration / (num_nodes - 1)

eval_f = sym.lambdify((t, y1, y2, p), f, modules='numpy')
func = lambda y, t, p: eval_f(t, y[0], y[1], p).flatten()

y_meas = odeint(func, y0, time, args=(p_val,))

y1_meas = y_meas[:, 0]
y2_meas = y_meas[:, 1]

# Setup the optimization problem


def obj(free):
    """Minimize the error in the angle, y1."""
    return interval * np.sum((y1_meas - free[:num_nodes])**2)


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[:num_nodes] = 2.0 * interval * (free[:num_nodes] - y1_meas)
    return grad

con_col = ConstraintCollocator(eom, (y1, y2), num_nodes, interval)

prob = Problem(con_col.num_free,
               con_col.num_constraints,
               obj,
               obj_grad,
               con_col.generate_constraint_function(),
               con_col.generate_jacobian_function(),
               con_col.jacobian_indices)

# Zeros for the state trajectories and a random positive value for the
# parameter.
initial_guess = np.hstack((np.zeros(num_states * num_nodes),
                           50.0 * np.random.random(1)))

# Known solution as initial guess.
initial_guess = np.hstack((y1_meas, y2_meas, 10.0))

# Find the optimal solution.
solution, info = prob.solve(initial_guess)
p_sol = solution[-1]

print("Known value of p = {}".format(p_val))
print("Identified value of p = {}".format(p_sol))

# Simulate with the identified parameter.
y_sim = odeint(func, y0, time, args=(p_sol,))
y1_sim = y_sim[:, 0]
y2_sim = y_sim[:, 1]

# Plot results
fig_y1, axes_y1 = plt.subplots(3, 1)

legend = ['measured', 'initial guess', 'direct collocation solution',
          'identified simulated']

axes_y1[0].plot(time, y1_meas, '.k',
                time, initial_guess[:num_nodes], '.b',
                time, solution[:num_nodes], '.r',
                time, y1_sim, 'g')
axes_y1[0].set_xlabel('Time [s]')
axes_y1[0].set_ylabel('y1 [rad]')
axes_y1[0].legend(legend)

axes_y1[1].set_title('Initial Guess Constraint Violations')
axes_y1[1].plot(prob.con(initial_guess)[:num_nodes - 1])
axes_y1[2].set_title('Solution Constraint Violations')
axes_y1[2].plot(prob.con(solution)[:num_nodes - 1])

plt.tight_layout()

fig_y2, axes_y2 = plt.subplots(3, 1)

axes_y2[0].plot(time, y2_meas, '.k',
                time, initial_guess[num_nodes:-1], '.b',
                time, solution[num_nodes:-1], '.r',
                time, y2_sim, 'g')
axes_y2[0].set_xlabel('Time [s]')
axes_y2[0].set_ylabel('y2 [rad]')
axes_y2[0].legend(legend)

axes_y2[1].set_title('Initial Guess Constraint Violations')
axes_y2[1].plot(prob.con(initial_guess)[num_nodes - 1:])
axes_y2[2].set_title('Solution Constraint Violations')
axes_y2[2].plot(prob.con(solution)[num_nodes - 1:])

plt.tight_layout()

plt.show()
