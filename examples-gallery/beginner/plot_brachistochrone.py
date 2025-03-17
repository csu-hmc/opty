
# %%
"""
Brachistochrone
===============

Objective
---------

Find the Brachistochrone by letting opty fin the fastest path to go from A(0,0)
to B(b1,b2) in minimum time.

Idea
----

Let f(x) be the curve along it should slide. let t(x(t)) be the tangent to the
curve at x(t). The angle between the tangent and the verticalo is beta, the
control parameter. Split the gravitational force into a component normal to
the tangent and a component along the tangent. The component along the tangent
is the force that accelerates the particle, the normal one does no work.
The tangential component is plit into N.x, N.y directions to get the eoms.

"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from opty import Problem

from scipy.optimize import root

# %%
# Equations of Motion
# -------------------
N = me.ReferenceFrame('N')
O, P = sm.symbols('O P', cls=me.Point)
O.set_vel(N, 0)
t = me.dynamicsymbols._t

m, g = sm.symbols('m g')
x, y, ux, uy = me.dynamicsymbols('x y ux uy')
beta = me.dynamicsymbols('beta')

P.set_pos(O, x*N.x + y*N.y)
P.set_vel(N, ux*N.x + uy*N.y)

body = [me.Particle('body', P, m)]
delta = sm.pi/2 - beta
forces = [(P, -m*g*sm.cos(beta)**2*N.y + m*g*sm.sin(beta)*sm.cos(beta)*N.x)]

kd = sm.Matrix([ux - x.diff(t), uy - y.diff(t)])

kane = me.KanesMethod(N, q_ind=[x, y], u_ind=[ux, uy], kd_eqs=kd)
fr, frstar = kane.kanes_equations(body, forces)

eom = kd.col_join(fr + frstar)
eom

# %%
# Set up the Optimization Problem and Solve It
# --------------------------------------------
h = sm.symbols('h')
num_nodes = 101
t0, tf = 0.0, h*(num_nodes - 1)
interval = h

state_symbols = (x, y, ux, uy)

# B(b1/b2) is the final point, the starting point is (0/0), always
b1 = 10.0
b2 = -10.0
instance_constraint = (
    x.func(t0),
    y.func(t0),
    ux.func(t0),
    uy.func(t0),

    x.func(tf) - b1,
    y.func(tf) - b2,
)

bounds = {
        h: (0.0, 1.0),
        beta: (0.0, np.pi/2.0),
          }

par_map = {m: 1.0, g: 9.81}

def obj(free):
    return free[-1]

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad

prob = Problem(
                obj,
                obj_grad,
                eom,
                state_symbols,
                num_nodes,
                interval,
                known_parameter_map=par_map,
                instance_constraints=instance_constraint,
                bounds=bounds,
                backend='numpy',
)

#%%
# Calculate the Brachistochrone from (0, 0) to (b1, b2). It does not work for
# all values of b1 and b2. I did not investigate
def func(X0):
    """gives the Brachistochrome equation for the starting point at (0/0) and
    the final point at (b1/b2)"""
    R = X0[0]
    theta = X0[1]
    return [R*theta - R*np.sin(theta) - b1, R*(1.0 - np.cos(theta)) - b2]

X0 = [0.0, 0.0]
resultat = root(func, X0)
times = np.linspace(0.0, resultat.x[1], num_nodes)
XX = resultat.x[0]*times - resultat.x[0]*np.sin(times)
YY = resultat.x[0] * (1.0 - np.cos(times))

# solve the problem. Give the Brachiostrone as initial guess. A better one
# should not be possible.
initial_guess = np.random.randn(prob.num_free) * 0.1
x_guess = np.linspace(0.0, 1.0, num_nodes)
y_guess = np.linspace(0.0, -1.0, num_nodes)
initial_guess[0:num_nodes] = XX
initial_guess[num_nodes:2*num_nodes] = YY

solution, info = prob.solve(initial_guess)
print(info['status_msg'])
# %%
_ = prob.plot_trajectories(solution)
# %%
_ = prob.plot_constraint_violations(solution)
# %%
_ = prob.plot_objective_value()

# %%
tff = solution[-1]  * (num_nodes - 1)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_aspect('equal')
ax.plot(solution[:num_nodes], solution[1*num_nodes:2*num_nodes], label='solution')
ax.plot(XX, YY, label='Brachistochrone')
ax.set_xlabel('x')
ax.set_ylabel('y')
_ = ax.legend()