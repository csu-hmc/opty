"""
Path Constraints
================

"""
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from opty import Problem
from opty.utils import create_objective_function

m, r = sm.symbols('m, r', real=True)
x, y = me.dynamicsymbols('x, y', real=True)
vx, vy = me.dynamicsymbols('v_x, v_y', real=True)
Fx, Fy = me.dynamicsymbols('F_x, F_y', real=True)
t = me.dynamicsymbols._t

eom = sm.Matrix([
    m*vx.diff() + vx - Fx,
    x.diff() - vx,
    m*vy.diff() + vy - Fy,
    y.diff() - vy,
    x**2 + y**2 - r**2,
])

states = (x, y, vx, vy)
specifieds = (Fx, Fy)

num_nodes = 101
interval_value = 0.1
dur = interval_value*(num_nodes - 1)

obj_func = sm.Integral(Fx**2 + Fy**2, t)
obj, obj_grad = create_objective_function(
    obj_func, states, specifieds, tuple(), num_nodes,
    interval_value, time_symbol=t)

instance_constraints = (
    x.func(0.0),
    y.func(0.0) + r,
    vx.func(0.0),
    vy.func(0.0),
    x.func(dur),
    y.func(dur) - r,
    vx.func(dur),
    vy.func(dur),
)

par_map = {
    m: 1.0,
    r: 1.0,
}

prob = Problem(
    obj,
    obj_grad,
    eom,
    states,
    num_nodes,
    interval_value,
    known_parameter_map=par_map,
    instance_constraints=instance_constraints,
    time_symbol=t,
    backend='numpy',
)

initial_guess = np.zeros(prob.num_free)
sol, _ = prob.solve(initial_guess)

# %%
_ = prob.plot_trajectories(sol)

# %%
_ = prob.plot_constraint_violations(sol)

# %%
_ = prob.plot_objective_value()

plt.show()
