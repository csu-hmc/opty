 # %%
"""
Singular Arc Problem
====================

This is example 10.47 from Betts' book "Practical Methods for Optimal Control
Using NonlinearProgramming", 3rd edition, Chapter 10: Test Problems.
John T. Betts sent me his detailed results of this example, and when I say
'error' it is the relative error in the value compared to the values sent.

**Phase 1 Maximum Thrust**

**States**

- :math:`h, v, m`: state variables

Note: I simply copied the equations of motion, the bounds and the constants
from the book. I do not know their exact meaning.

"""
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
import time

# %%
# Equations of motion.
t = me.dynamicsymbols._t
h, v, m = me.dynamicsymbols('h v m')
h_fast = sm.symbols('h_fast')

# Parameters, the same for all three phases
Tm = 193.044
g = 32.174
sigma = 5.49153 * 1.e-5
c = 1580.942579
h0 = 23800


eom = sm.Matrix([
    -h.diff(t) + v,
    -v.diff(t) + 1/m * (Tm - sigma*v**2*sm.exp(-h/h0)) - g,
    -m.diff(t) - Tm/c,
])
sm.pprint(eom)

# %%
num_nodes = 101
t0, tf = 0*h_fast, num_nodes * h_fast
interval_value = h_fast

state_symbols = (h, v, m)

# %%
# Specify the objective function and form the gradient.
# h(tf) is to be maximized.
# Note that h_fast is the last entry of free.

def obj(free):
    return -free[num_nodes-1] * free[-1]

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[num_nodes-1] = -free[-1]
    grad[-1] = -free[num_nodes-1]
    return grad

# Specify the symbolic instance constraints, as per the example
instance_constraints = (
    h.func(0*h_fast),
    v.func(0*h_fast),
    m.func(0*h_fast) - 3.0,
    )

bounds = {
    h_fast: (0.0, 0.5),
    m: (1.0, 3.0)
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
            bounds=bounds,
    )

#prob.add_option('nlp_scaling_method', 'gradient-based')
prob.add_option('max_iter', 1000)

# Give some rough estimates for the trajectories.
initial_guess = np.ones(prob.num_free) * 0.1

# %%
# Find the optimal solution.
solution, info = prob.solve(initial_guess)
print(info['status_msg'])
t_phase1 = solution[-1]

# %%
# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# %%
# Plot the objective function as a function of optimizer iteration.
prob.plot_objective_value()
error_t = (t_phase1*(num_nodes-1) - 13.726485)/13.726485  * 100
error_v = (solution[2*num_nodes-1] - 791.2744)/791.2744 * 100
error_h = (solution[num_nodes-1] - 4560.8912)/4560.8912 * 100
error_m = (solution[3*num_nodes-1] - 1.323901)/1.323901 * 100
print(f'duration of phase 1 = {(num_nodes-1) * solution[-1]:.3f}, ' +
      f'error is {error_t:.3f} %')
print(f'Height achieved is {- info['obj_val']/solution[-1]:.3f}, ' +
      f'error is  {error_h:.3f} %')
print(f'Velocity achieved is {solution[2*num_nodes-1]:.3f}, ' +
      f'error is {error_v:.3f} %')
print(f'Final mass is {solution[3*num_nodes-1]:.3f}, ' +
      f'        error is {error_m:.3f} %')
# %%
# **Phase 2  Singular Arc**
#
# Now the thrust T(t) becomes a state variable.
#
# **States**
#
# - :math:`h, v, m, T`: state variables
#

# Set up the eoms
T = me.dynamicsymbols('T')


eom = sm.Matrix([
    -h.diff(t) + v,
    -v.diff(t) + 1/m * (T - sigma*v**2*sm.exp(-h/h0)) - g,
    -m.diff(t) - T/c,
#    testdt,
    T - sigma*v**2*sm.exp(-h/h0) - m*g
     -m*g/(1 + 4*c/v + 2*c**2/v**2) * (c**2/(h0*g)*(1+v/c) - 1 - 2*c/v),
])
sm.pprint(eom)

state_symbols = (h, v, m, T)

# %%
# The instance constraints (of course) are the values achieved at the end of
# the first phase.
instance_constraints = (
    h.func(0*h_fast) - solution[num_nodes-1],
    v.func(0*h_fast) - solution[2*num_nodes-1],
    m.func(0*h_fast) -  solution[3*num_nodes-1],
    T.func(0*h_fast) - Tm ,
    m.func((num_nodes-1)*h_fast) - 1.0,
    )

# As per the example, the mass m >= 1.0
bounds = {
    h_fast: (0.0, 0.5),
    T: (0.0, Tm),
    m: (1.0, solution[3*num_nodes-1])
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
            bounds=bounds,
    )

prob.add_option('max_iter', 1000)

# Give some rough estimates for the trajectories.
initial_guess = np.array((*[solution[num_nodes-1] for _ in range(num_nodes)],
                          *[solution[2*num_nodes-1] for _ in range(num_nodes)],
                          *[solution[3*num_nodes-1] for _ in range(num_nodes)],
                          *[Tm/c for _ in range(num_nodes)], solution[-1]))

# %%
# Find the optimal solution.
solution, info = prob.solve(initial_guess)
print(info['status_msg'])
t_phase2 = solution[-1]

# %%
# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# %%
# Plot the objective function as a function of optimizer iteration.
prob.plot_objective_value()

error_t = ((t_phase2*(num_nodes-1) - (22.025604-13.726485))/
           (22.025604-13.726485)  * 100)
error_v = (solution[2*num_nodes-1] - 789.64098)/789.64098 * 100
error_h = (solution[num_nodes-1] - 11121.110)/11121.110 * 100
error_m = (solution[3*num_nodes-1] - 1.0)/1.0 * 100
print(f'duration of phase 2 = {(num_nodes-1) * solution[-1]:.3f}, ' +
      f'error is   {error_t:.3f} %')
print(f'Height achieved is {- info['obj_val']/solution[-1]:.3f}, ' +
      f'error is  {error_h:.3f} %')
print(f'Velocity achieved is {solution[2*num_nodes-1]:.3f}, ' +
      f'error is   {error_v:.3f} %')
print(f'Final mass is {solution[3*num_nodes-1]:.3f}, ' +
      f'           error is {error_m:.3f} %')

# %%
# **Phase 3  No Thrust**
#
# Now it is free gliding without thrust.
#
# **States**
#
# - :math:`h, v, m`: state variables.

# Set up the eoms

eom = sm.Matrix([
    -h.diff(t) + v,
    -v.diff(t) - sigma*v**2*sm.exp(-h/h0)/m - g,
    -m.diff(t) - 0,
])
sm.pprint(eom)

state_symbols = (h, v, m)

# %%
# Now it is free gliding without thrust. The maximum height is achieved when
# the velocity is zero. The objective function to be minimized is the square
# of the speed.

def obj(free):
    return free[2*num_nodes-1]**2

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[2*num_nodes-1] = 2*free[2*num_nodes-1]
    grad[-1] = 0.0 #free[num_nodes-1]**2
    return grad


# Specify the symbolic instance constraints. The starting values are the final
# values of the previous phase. The mass must be 1.0 at the end.
instance_constraints = (
    h.func(0*h_fast) - solution[num_nodes-1],
    v.func(0*h_fast) - solution[2*num_nodes-1],
    m.func(0*h_fast) -  solution[3*num_nodes-1],
    m.func((num_nodes-1)*h_fast) - 1.0,
    )

bounds = {
    h_fast: (0.0, 0.5),
    m: (1.0, solution[3*num_nodes-1])
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
            bounds=bounds,
#            time_symbol=t,
    )

#prob.add_option('nlp_scaling_method', 'gradient-based')
prob.add_option('max_iter', 9000)

# Give some rough estimates for the trajectories.
initial_guess = np.array((*[solution[num_nodes-1] for _ in range(num_nodes)],
                          *[solution[2*num_nodes-1] for _ in range(num_nodes)],
                          *[solution[3*num_nodes-1] for _ in range(num_nodes)],
                          solution[-1]))


# %%
# Find the optimal solution.
solution, info = prob.solve(initial_guess)
print(info['status_msg'])
t_phase3 = solution[-1]

# %%
# Plot the optimal state and input trajectories.
prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
prob.plot_constraint_violations(solution)

# %%
# Plot the objective function as a function of optimizer iteration.
prob.plot_objective_value()

print(f'final speed is  {solution[2*num_nodes-1]:.3e}, ideally it should be 0')
print('\n')
error_h = (solution[num_nodes-1] - 18664.187)/18664.187 * 100
print(f'final height achieved is is {solution[num_nodes-1]:.3f}, error is '+
      f'           {error_h:.3f} %')

error_t1 = (t_phase1*(num_nodes-1) - 13.751270)/13.751270  * 100
error_t2 = ((t_phase1+t_phase2)*(num_nodes-1) - 21.987)/21.987  * 100
print('duration of first and second phase is ' +
      f'{(t_phase1+t_phase2)*(num_nodes-1):.3f} sec, error is {error_t2:.3f} %')
error_t3 = ((t_phase1+t_phase2+t_phase3)*(num_nodes-1) - 42.641355)/42.641355  * 100
print(f'total duration  is {(t_phase1 + t_phase2 + t_phase3) * (num_nodes-1):.3f}'+
      f' sec, error is                     {error_t3:.3f} %')

# %%
# sphinx_gallery_thumbnail_number = 8
