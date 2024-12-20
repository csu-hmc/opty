"""
Tumor Antigionesis
==================

This is example 10.141 from Betts' book "Practical Methods for Optimal Control
Using NonlinearProgramming", 3rd edition, Chapter 10: Test Problems.
It is described in more detail in section 8.17 of the book.

Note: Only if I set :math:`t_f \\approx 1.25....2.0` do I get results that are
close to the book. That is :math:`h_{fast} = \dfrac{1.25...2.0}{ \\text{num_nodes-1}}`.
Otherwise it does not converge, or converges to VERY different results.

**States**

- :math:`p, q, y` : state variables

**Controls**

- :math:`u` : control variable

"""
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem

# %%
# Equations of Motion.
# --------------------
t = me.dynamicsymbols._t

p, q, y = me.dynamicsymbols('p, q, y')
u = me.dynamicsymbols('u')

h_fast = sm.symbols('h_fast')

# %%
# Parameters
chi = 0.084
G = 0.15
b = 5.85
nu = 0.02
d = 0.00873
a = 75
A = 15

pbar = ((b - nu)/d)**(3/2)
qbar = pbar
p0 = pbar/2
q0 = qbar/4

# %%
# equations of motion.
eom = sm.Matrix([
    -p.diff(t) - chi*p*sm.log(p/q),
    -q.diff(t) +  q*(b - (nu + d*p**(2/3) + G*u)),
    -y.diff(t) + u,
])
sm.pprint(eom)

# %%
# Define and Solve the Optimization Problem.
# ------------------------------------------
num_nodes = 2001
iterations = 2000

interval_value = h_fast
t0, tf = 0*h_fast, (num_nodes-1)*h_fast

state_symbols = (p, q, y)
unkonwn_input_trajectories = (u, )

# Specify the objective function and form the gradient.
def obj(free):
    return free[num_nodes-1]

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[num_nodes-1] = 1.0
    return grad

instance_constraints = (
        p.func(t0) - p0,
        q.func(t0) - q0,
        y.func(t0),
)

bounds = {
        h_fast: (0.0, 1.3/(num_nodes-1)),
        p: (0.01, pbar),
        q: (0.01, qbar),
        y: (0.0, A),
        u: (0, a),
}

prob = Problem(
        obj,
        obj_grad,
        eom,
        state_symbols,
        num_nodes,
        interval_value,
        instance_constraints= instance_constraints,
        bounds=bounds,
)

# %%
# Solve the optimization problem.
# Give some rough estimates for the trajectories.
# Here is used the solution of a previous run, stored in
# *betts_10_141_solution.npy* to speed up the running.
initial_guess = np.ones(prob.num_free)
initial_guess =np.load('betts_10_141_solution.npy')

# Find the optimal solution.
for i in range(1  ):
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    print(f'Duration is: {solution[-1]*(num_nodes-1):.4f}, ' +
          f'as per the book it is {1.1961336}, so the deviation is: ' +
        f'{(solution[-1]*(num_nodes-1) - 1.1961336)/1.1961336*100 :.3e} %')
    print(f'p(tf) = {solution[num_nodes-1]:.4f}' +
          f'as per the book it is {7571.6712}, so the deviation is: ' +
        f'{(solution[num_nodes-1] - 7571.6712)/7571.6712*100 :.3e} %')

# %%
# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# %%
# Plot the objective function as a function of optimizer iteration.
prob.plot_objective_value()

# %%
# sphinx_gallery_thumbnail_number = 2

