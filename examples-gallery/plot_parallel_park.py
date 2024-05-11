"""
Parallel Park a Car
===================

Given the nonholonomic bicycle model of the car find a solution for parallel
parking.

- m : car mass
- I : car yaw moment of inertia
- a : distance from front axle to mass center
- b : distance from rear axle to mass center

States

- x : position of mass center
- y : position of mass center
- theta : yaw angle of the car
- delta : steer angle of the front wheels relative to the car
- vx : longitudinal speed of the car's mass center
- vy : lateral speed of the car's mass center
- omega : yaw angular rate of the car
- beta : steer angular rate of the front wheels relative to the car

Specifieds

- F : longitudinal propulsion force
- T : steering torque

"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
from opty.utils import create_objective_function, parse_free
import matplotlib.pyplot as plt
import matplotlib.animation as animation

m, I, a, b = sm.symbols('m, I, a, b', real=True)
x, y, vx, vy = me.dynamicsymbols('x, y, v_x, v_y', real=True)
theta, omega = me.dynamicsymbols('theta, omega', real=True)
delta, beta = me.dynamicsymbols('delta, beta', real=True)
T, F = me.dynamicsymbols('T, F', real=True)
t = me.dynamicsymbols._t

O, Ao, Pr, Pf = sm.symbols('O, A_o, P_r, P_F', cls=me.Point)
N, A, B = sm.symbols('N, A, B', cls=me.ReferenceFrame)

A.orient_axis(N, theta, N.z)
B.orient_axis(A, delta, A.z)

Ao.set_pos(O, x*N.x + y*N.y)
Pr.set_pos(Ao, -b*A.x)
Pf.set_pos(Ao, a*A.x)

A.set_ang_vel(N, omega*N.z)

kinematical = [
    vx - (x.diff()*N.x + y.diff()*N.y).dot(A.x),
    vy - (x.diff()*N.x + y.diff()*N.y).dot(A.y),
    omega - theta.diff(),
    beta - delta.diff(),
]

O.set_vel(N, 0)
Ao.set_vel(N, vx*A.x + vy*A.y)
Pr.v2pt_theory(Ao, N, A)
Pf.v2pt_theory(Ao, N, A)

nonholonomic = [
    Pr.vel(N).dot(A.y),
    Pf.vel(N).dot(B.y),
]

IA = me.inertia(A, 0, 0, I)
car = me.RigidBody('A', Ao, A, m, (IA, Ao))
IB = me.inertia(B, 0, 0, I/32)
wheel = me.RigidBody('B', Pf, B, m/6, (IB, Pf))

propulsion = (Pr, F*A.x)
steeringA = (A, -T*B.z)
steeringB = (B, T*B.z)

kane = me.KanesMethod(
    N,
    [x, y, theta, delta],
    [vx, beta],
    kd_eqs=kinematical,
    u_dependent=[vy, omega],
    velocity_constraints=nonholonomic,
)

fr, frstar = kane.kanes_equations([car, wheel],
                                  [propulsion, steeringA, steeringB])

eom = (fr + frstar).col_join(
    sm.Matrix(nonholonomic)).col_join(
        sm.Matrix(kinematical))
sm.pprint(eom)

duration = 30.0  # seconds
num_nodes = 501
interval_value = duration/(num_nodes - 1)
time = np.linspace(0.0, duration, num=num_nodes)

par_map = {
    I: 1/12*1200*(2**2 + 3**2),
    m: 1200,
    a: 1.5,
    b: 1.5,
}

state_symbols = (x, y, theta, delta, vx, vy, omega, beta)
specified_symbols = (T, F)

# %%
# Specify the objective function and it's gradient, in this case it calculates
# the area under the input torque curve over the simulation.
#obj_func = sm.Integral(F*vx + T*beta, t)
obj_func = sm.Integral(F**2 + T**2, t)
sm.pprint(obj_func)
obj, obj_grad = create_objective_function(obj_func,
                                          state_symbols,
                                          specified_symbols,
                                          tuple(),
                                          num_nodes,
                                          node_time_interval=interval_value)

# %%
# Specify the symbolic instance constraints, i.e. initial and end conditions.
instance_constraints = (
    x.func(0.0),
    y.func(0.0),
    theta.func(0.0),
    delta.func(0.0),
    vx.func(0.0),
    vy.func(0.0),
    omega.func(0.0),
    beta.func(0.0),
    x.func(duration),
    y.func(duration) - 1.0,
    theta.func(duration),
    delta.func(duration),
    vx.func(duration),
    vy.func(duration),
    omega.func(duration),
    beta.func(duration),
)

# %%
# Limit the torque to a maximum magnitude.
bounds = {
    delta: (np.deg2rad(-45.0), np.deg2rad(45.0)),
    T: (-100.0, 100.0),
    F: (-10000.0, 10000.0),
}

# %%
# Create an optimization problem.
prob = Problem(obj, obj_grad, eom, state_symbols,
               num_nodes, interval_value,
               known_parameter_map=par_map,
               instance_constraints=instance_constraints,
               integration_method='midpoint',
               bounds=bounds)

prob.add_option('nlp_scaling_method', 'gradient-based')

# %%
# Use a random positive initial guess.
x_guess = 3.0/duration*2.0*time
x_guess[num_nodes//2:] = (x_guess[num_nodes//2] -
                          3.0/duration*2.0*time[num_nodes//2:])
y_guess = 1.0/duration*time
initial_guess = np.ones(prob.num_free)
initial_guess[:num_nodes] = x_guess
initial_guess[num_nodes:2*num_nodes] = y_guess

# %%
# Find the optimal solution.
solution, info = prob.solve(initial_guess)
print(info['status_msg'])
print(info['obj_val'])

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
xs, us, ps = parse_free(solution, len(state_symbols), len(specified_symbols),
                        num_nodes)
fig, ax = plt.subplots()
ax.plot(xs[0], xs[1])

# %%
# Animate

points = [Ao, Pf, Pf.locatenew('Bf', a/4*B.x), Pf.locatenew('Br', -a/4*B.x)]
coordinates = Pr.pos_from(O).to_matrix(N)
for point in points:
    coordinates = coordinates.row_join(point.pos_from(O).to_matrix(N))
eval_point_coords = sm.lambdify((state_symbols, specified_symbols,
                                 list(par_map.keys())), coordinates)

x, y, z = eval_point_coords(xs[:, 0], us[:, 0], list(par_map.values()))

fig, ax = plt.subplots()
ax.set_aspect('equal')

lines, = ax.plot(x, y, color='black',
                 marker='o', markerfacecolor='blue', markersize=4)
# some empty lines to use for the wheel paths
Pr_path, = ax.plot([], [])
Pf_path, = ax.plot([], [])

title_template = 'Time = {:1.2f} s'
title_text = ax.set_title(title_template.format(time[0]))
ax.set_xlim((np.min(xs.T[:, 0]) - 1.0, np.max(xs.T[:, 0]) + 1.0))
ax.set_ylim((np.min(xs.T[:, 1]) - 1.0, np.max(xs.T[:, 1]) + 1.0))
ax.set_xlabel(r'$x$ [m]')
ax.set_ylabel(r'$y$ [m]')

coords = []
for xi, ui in zip(xs.T, us.T):
    coords.append(eval_point_coords(xi, ui, list(par_map.values())))
coords = np.array(coords)  # shape(600, 3, 8)


def animate(i):
    title_text.set_text(title_template.format(time[i]))
    lines.set_data(coords[i, 0, :], coords[i, 1, :])
    Pr_path.set_data(coords[:i, 0, 0], coords[:i, 1, 0])
    Pf_path.set_data(coords[:i, 0, 2], coords[:i, 1, 2])


ani = animation.FuncAnimation(fig, animate, len(time),
                              interval=int(interval_value*1000))

plt.show()
