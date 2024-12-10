# %%
"""
Heat Diffusion Process with Inequality
======================================

This is example 10.57 from Betts' book "Practical Methods for Optimal Control
Using Nonlinear Programming", 3rd edition, chapter 10: Test Problems.
It deals with the 'discretization' of a PDE.

There are N equations of motion:

:math:`\\dot{y}_i = function(y_j, u_0, u_{pi}, parameters)`.

There are also N inequality constraints:

:math:`y_i - g(x_k, t) \\geq 0`, with :math:`x_k = k \\dfrac{\ \pi}{N}`.

So, I first do this: :math:`\\dfrac{d}{dt} y(t) \\geq \\dfrac{d}{dt} g(x_k, t)`.

Then I rewrite the equations of motion like this:

:math:`\\dfrac{d}{dt} g(x_k, t) = factor \cdot function(y_j, u_0, u_{pi}, parameters)`.

Where factor :math:`\in (1.0, \infty)` if :math:`g(x_k, t) \\geq 0`
and factor :math:`\in (-\infty, 1.0)` if :math:`g(x_k, t) < 0`.

As the equations of motion are only algebraic equations, and *opty* needs at
least one differential equation, I simply add a differential equation which
is not needed for the problem.

**States**

- :math:`y_0, .....y_{N-1}` : state variables
- :math:`u_{y0}` : not needed state variable

**Specifieds**

- :math:`u_0, u_{pi}` : control variables

"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
import matplotlib.pyplot as plt

# %%
# Equations of Motion
# -------------------
t = me.dynamicsymbols._t
T = sm.symbols('T', cls=sm.Function)

N = 20
t0, tf = 0.0, 5.0
exponent = 500
num_nodes = 101

y = list(me.dynamicsymbols(f'y:{N}'))
uy0 = me.dynamicsymbols('uy0')
u0, upi = me.dynamicsymbols('u0 upi')

faktor1, faktor2 = sm.symbols('faktor1 faktor2')

#Parameters
q1 = 1.e-3
q2 = 1.e-3
a = 0.5
b = 0.2
c = 1.0
delta = np.pi/N

# %%
# The function gdt gives the derivative of the constraints, needed for the
# equations of motion.
def gdt(k, t):
    x = k * np.pi/N
    return ((c * (sm.sin(x) * sm.sin( np.pi*t/tf) - a) - b).diff(t))

# %%
# This function determines the factor for the equations of motion.
def faktor(k, t, faktor1, faktor2):
    test = (c * (sm.sin(k * np.pi/N) * sm.sin(np.pi*t/tf) - a) - b)
    hilfs = (1/(1+sm.exp(-exponent*test))*faktor1 +
             1/(1+sm.exp(exponent*test))*faktor2)
    return hilfs

# %%
# The first equation is only needed, because *opty* needs at least one
# differential equation.
eom = sm.Matrix([
    -y[0].diff(t) + uy0,
    -gdt(1, T(t)) + faktor(1, T(t), faktor1, faktor2) * 1/delta**2 * (y[1]
                - 2*y[0] + u0),
    *[-gdt(i+1, T(t)) + faktor(i+1, T(t), faktor1,
            faktor2) * 1/delta**2 * (y[i+1] - 2*y[i] + y[i-1])
            for i in range(1, N-1)],
    -gdt(N, T(t)) + faktor(N, T(t), faktor1, faktor2) * 1/delta**2 * (upi -
                2*y[N-1] + y[N-2]),
])

sm.pprint(eom)

# %%
# Solve the Optimization Problem
# ------------------------------
interval_value = (tf - t0)/(num_nodes - 1)

state_symbols = [uy0] +  y
specified_symbols = (u0, upi)

times = np.linspace(t0, tf, num=num_nodes)

# %%
# Plot the constraints and the approximated Heavyside functions. It shows
# that even with exponent = 500 (the largest numpy seems to accept here) the
# Heavyside functions are not 'sharp' if the slope of the constraint is very
# close to zero.
x = sm.symbols('x')
t1 = sm.symbols('t1')
g = (c * (sm.sin(x) * sm.sin(sm.pi*t1/tf) - a) - b)

hilfs = 1/(1+sm.exp(-exponent*g))
hilfs1 = 1/(1+sm.exp(exponent*g))
g_lam = sm.lambdify((x, t1), g, cse=True)
hilfs_lam = sm.lambdify((x, t1), hilfs, cse=True)
hilfs1_lam = sm.lambdify((x, t1), hilfs1, cse=True)

Delta = []
DELTA_1 = []
DELTA_2 = []
for i in range(N):
    delta_h = []
    delta_h_1 = []
    delta_h_2 = []
    for j in range(num_nodes):
        delta_h.append(g_lam((i+1)*np.pi/N, times[j]))
    Delta.append(delta_h)

fig, ax = plt.subplots(N ,1, figsize=(8, 1.5*N), constrained_layout=True)
for i in range(N):
    ax[i].plot(times, Delta[i], label=str(i))
    ax[i].plot(times, hilfs_lam((i+1)*np.pi/N, times), label='hilfs')
    ax[i].plot(times, hilfs1_lam((i+1)*np.pi/N, times), label='hilfs1')
#    ax[i].axhline(0, color='black')
    ax[i].legend()
ax[0].set_title('Constraints and approx. Heavyside functions')
ax[-1].set_xlabel('time [s]')
prevent_printing = 1

# %%
# Specify the objective function and form the gradient.
def obj(free):
    value1 = interval_value * (delta/2 + q1) * sum([free[(N+1)*num_nodes+i]**2
                for i in range(num_nodes)])
    value2 = interval_value * (delta/2 + q2) * sum([free[(N+2)*num_nodes+i]**2
                for i in range(num_nodes)])
    value3 = 0
    for i in range(1, N+1):
        value3 += interval_value * delta * sum([free[i*num_nodes+j]**2
                for j in range(num_nodes)])
    return value1 + value2 + value3

def obj_grad(free):
        grad = np.zeros_like(free)
        grad[(N+1)*num_nodes:(N+2)*num_nodes] = (2 * (delta/2 + q1) *
                interval_value * free[(N+1)*num_nodes:(N+2)*num_nodes])
        grad[(N+2)*num_nodes:(N+3)*num_nodes] = (2 * (delta/2 + q2) *
                interval_value * free[(N+2)*num_nodes:(N+3)*num_nodes])
        for i in range(1, N+1):
            grad[i*num_nodes:(i+1)*num_nodes] = (2 * delta * interval_value *
                free[i*num_nodes:(i+1)*num_nodes])
        return grad

# %%
# Specify the instance constraints, as per the example, and the bounds.
instance_constraints = (
        *[y[i].func(t0) for i in range(N)],
)

bounds = {
        faktor1: (1.0, np.inf),
        faktor2: (-np.inf, 1.0),
}

# %%
# Create the optimization problem and set any options.
prob = Problem(obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        instance_constraints=instance_constraints,
        known_trajectory_map={T(t): times},
        bounds=bounds,
)

prob.add_option('max_iter', 20000)

# %%
# Give some rough estimates for the trajectories. Here I use the solution from
# a previous run to speed up the optimization process.
initial_guess = np.zeros(prob.num_free)
initial_guess = np.load('betts_10_57_solution.npy')

# %%
# Find the optimal solution.
for _ in range(1):
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
#    np.save('betts_10_57_solution', solution)
    print(info['status_msg'])
    print(f'Objective value achieved: {info['obj_val']:.4f}, as per the book ' +
        f'it is {4.68159793*1.e-1}, so the error is: '
        f'{(info['obj_val'] - 4.68159793*1.e-1)/(4.68159793*1.e-1)*100:.3f} % ')

# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# %%
# Plot the objective function as a function of optimizer iteration.
prob.plot_objective_value()


# %%
# Plot the inequality constraint violations. It shows, that the constraints
# are not always fulfilled.
x = sm.symbols('x')
t1 = sm.symbols('t1')
g = c * (sm.sin(x) * sm.sin(sm.pi*t1/tf) - a) - b
g_lam = sm.lambdify((x, t1), g, cse=True)
Delta = []
for i in range(N):
    delta_h = []
    for j in range(num_nodes):
        delta_h.append(solution[(i+1)*num_nodes + j] -
            g_lam((i+1)*np.pi/N, times[j]))
    Delta.append(delta_h)

fig, ax = plt.subplots(N ,1, figsize=(8, 1.5*N), constrained_layout=True)
for i in range(N):
    ax[i].plot(times, Delta[i], label=str(i))
    ax[i].axhline(0, color='black')
    ax[i].legend()
ax[0].set_title('Constraint violation, Must be $\\geq 0.0$')
ax[-1].set_xlabel('time [s]')
prevent_printing = 1


# %%
# sphinx_gallery_thumbnail_number = 3
