r"""
Quarter Car Model on a Bumpy Road
=================================

Objective
---------

- Show on a simple example how to simultaneously optimize free
  parameters and unknown trajectories of a system.
- Show how one can iterate from a simpler problem (road less bumpy) to a
  harder problem (road more bumpy).


Introduction
------------

A quarter car model must move from a starting point to a final point on a bumpy
road. The body is connected to the wheel on the road by a linear spring and a
damper.

https://www.tevema.com/what-is-the-best-suspension-for-luxury-cars/

Body and wheel are modeled by particles. Movement is in the X/Z plane. Gravity
points in the negative Z direction.


**States**

- :math:`x_{car}` : x position of the car [m]
- :math:`z_{car}` : z position of the car [m]
- :math:`ux_{car}` : x velocity of the car [m/s]
- :math:`uz_{car}` : z velocity of the car [m/s]
- :math:`prevent_{jump}` : to ensure the wheel always has a negative load,
  meaning it will not jump [N]
- :math:`steady_{body}` : to ensure that the vertical motions of the body are
  not too large [m]
- :math:`accel_{body}` : holds the acceleration of the body, which is to be
  minimized [m/s^2]
- :math:`accel_{wheel}` : holds the accelerations of the wheel, only needed for
  the animation. [m/s^2]



**Fixed Parameters**

- :math:`m_{car}` : sprung mass [kg]
- :math:`m_{wheel}` : unsprung mass [kg]
- :math:`g` : gravity [m/s^2]
- :math:`l_0` : equilibrium length of the spring [m]
- :math:`r_1, r_2, r_3, r_4, r_5` : parameters of the street.


**Free Parameters**

- :math:`c` : damping constant [Ns/m]
- :math:`k` : spring constant [N/m]

**Unknown Trajectories**

- :math:`f_x` : driving force [N]

"""
import os
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty import Problem
from opty.utils import parse_free
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# %%
# Set up the Equations of Motion
#-------------------------------
N = me.ReferenceFrame('N')
O, P_car, P_wheel = sm.symbols('O, P_car, P_wheel', cls=me.Point)
O.set_vel(N, 0)
t = me.dynamicsymbols._t
x_car, z_car = me.dynamicsymbols('x_car z_car')
ux_car, uz_car = me.dynamicsymbols('ux_car uz_car')
prevent_jump, steady_body = me.dynamicsymbols('prevent_jump, steady_body')
accel_body, accel_wheel = me.dynamicsymbols('accel_body accel_wheel')

fx = me.dynamicsymbols('fx')
m_car, m_wheel, g = sm.symbols('m_car m_wheel g')
r1, r2, r3, r4, r5 = sm.symbols('r1 r2 r3 r4 r5')

l_0, k, c = sm.symbols('  l_0, k, c')
# %%
# Define the rough surface of the street.

def rough_surface(x_car):
    omega = 0.75
    return sm.S(0.135 ) *  (r1*sm.sin(omega*x_car)**2 + r2*sm.sin(2*omega*x_car)**2
        + r3*sm.sin(3*omega*x_car)**2 + r4*sm.sin(7*omega*x_car)**2
        + r5*sm.sin(9*omega*x_car)**2)

# %%
# Set up the system.
P_car.set_pos(O, x_car*N.x + z_car*N.z)
P_wheel.set_pos(O, x_car*N.x  + rough_surface(x_car)*N.z)

P_car.set_vel(N, ux_car*N.x  + uz_car*N.z)
P_wheel.set_vel(N, ux_car*N.x + rough_surface(x_car).diff(t)*N.z)

Car = me.Particle('Car', P_car, m_car)
Wheel = me.Particle('Wheel', P_wheel, m_wheel)
bodies = [Car, Wheel]

F_car =[(P_car, -m_car*g*N.z - c*(uz_car - rough_surface(x_car).diff(t))*N.z
         + k*(l_0 - (z_car - rough_surface(x_car)))*N.z
        + fx * N.x)]
F_wheel = [(P_wheel, -m_wheel*g*N.z + c*(uz_car
            - rough_surface(x_car).diff(t))*N.z
            - k*(l_0 - (z_car - rough_surface(x_car)))*N.z)]

F_contact = (-m_wheel*g + c*(uz_car - rough_surface(x_car).diff(t))
            - k*(l_0 - (z_car - rough_surface(x_car))))
forces  = F_car + F_wheel

kd = sm.Matrix([x_car.diff(t) - ux_car, uz_car - z_car.diff(t)])

KM = me.KanesMethod(N,
                    q_ind=[x_car, z_car],
                    u_ind=[ux_car, uz_car],
                    kd_eqs=kd
)
fr, frstar = KM.kanes_equations(bodies, forces)
eom = kd.col_join(fr + frstar)

# %%
# Add the constraints.
# The 'detour' with aux_1 is needed, else an unkonwn trajectory
# :math:`\dfrac{d^2}{dt^2}x_{car}` will be created by opty.
aux_1 = (rough_surface(x_car).diff(t)).subs({x_car.diff(t): ux_car})
aux_1 = aux_1.diff(t)
eom = eom.col_join(sm.Matrix([
                                prevent_jump - F_contact,
                                steady_body - (z_car - rough_surface(x_car)),
                                accel_body - (uz_car.diff(t)),
                                accel_wheel - aux_1
]))

print(f'eoms contains {sm.count_ops(eom)} equations and have shape {eom.shape}')
# %%
# Set up the Optimization Problem
#--------------------------------
state_symbols = [x_car, z_car, ux_car, uz_car, prevent_jump, steady_body,
                 accel_body, accel_wheel]

h = sm.symbols('h')
num_nodes = 301
t0, tf = 0, h*(num_nodes - 1)
interval = h

par_map = {}
par_map[m_car] = 10.0
par_map[m_wheel] = 1.0
par_map[g] = 9.81
par_map[l_0] = 0.75
par_map[r1] = 0.1
par_map[r2] = 0.1
par_map[r3] = 0.1
par_map[r4] = 0.1
par_map[r5] = 0.1

# %%
#To be minimized: :math:`\int (\dfrac{d}{dt}uz_{car})^2 dt + \text{weight} \cdot t_f`
# ``weight`` is a scalar that can be used to adjust the importance of the
# speed.
weight = 1.e5

def obj(free):
    UZdot = np.sum([free[i]**2 for i in range(6*num_nodes, 7*num_nodes)])
    return (UZdot)*free[-1] + weight*free[-1]

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[6*num_nodes:7*num_nodes] = 2*free[6*num_nodes:7*num_nodes]*free[-1]
    grad[-1] = (
            + np.sum([free[i]**2 for i in range(6*num_nodes, 7*num_nodes)])
            + weight
        )
    return grad

# %%
# Add the instance constraints and bounds.
instance_constraints = (
    x_car.func(t0) - 0.0,
    ux_car.func(t0) - 0.0,

    x_car.func(tf) - 10.0,
    ux_car.func(tf) - 0.0,
)

bounds = {
    h: (0.0, 1.0),
    x_car: (0.0, 10.0),
    ux_car: (0.0, np.inf),
    prevent_jump: (-1000.0, 0.0),
    steady_body: (0.55, 0.65),
    c: (0.0, 750),
    k: (0.0, 5000),
    fx: (-500, 500),
}

# %%
# Use an existing solution if available, else iterate to find one.
fname =f'quarter_car_on_bumpy_road_{num_nodes}_nodes_solution.csv'
if os.path.exists(fname):
    # use the existing solution
    par_map[r3] = 0.1 + 0.075*4
    par_map[r4] = 0.075*4

    prob = Problem(obj,
               obj_grad,
               eom,
               state_symbols,
               num_nodes,
               interval,
               known_parameter_map=par_map,
               instance_constraints=instance_constraints,
               bounds=bounds,
               time_symbol=t,
               backend='numpy',
    )

    solution = np.loadtxt(fname)
else:
    # Iterate to find the solution. As the convergence is not easy, one has to
    # start with a smooth road and then increase the roughness gradually.
    for i in range(5):
        par_map[r3] = 0.1 + 0.075*i
        par_map[r4] = 0.075*i

        prob = Problem(obj,
               obj_grad,
               eom,
               state_symbols,
               num_nodes,
               interval,
               known_parameter_map=par_map,
               instance_constraints=instance_constraints,
               bounds=bounds,
               time_symbol=t,
        )

        prob.add_option('max_iter', 3000)
        if i == 0:
            initial_guess = np.ones(prob.num_free)*0.5
        else:
            initial_guess = solution
        for _ in range(2):
            solution, info = prob.solve(initial_guess)
            initial_guess = solution
            print(info['status_msg'])
            print('Objective value', info['obj_val'])

print('Sequence of unknown trajectories',
               prob.collocator.unknown_input_trajectories)
# %%
fig, axes = plt.subplots(9, 1, figsize=(7, 16), layout='tight')
_ = prob.plot_trajectories(solution, axes=axes)
# %%
_ = prob.plot_constraint_violations(solution)
print(f'value of optimum spring constant: {solution[-2]:.2f}')
print(f'value of optimum damping constant: {solution[-3]:.2f}')

# %%
# Plot the road.
r11, r22, r33, r44, r55 = [par_map[key] for key in [r1, r2, r3, r4, r5]]
rough_surface_lam = sm.lambdify((x_car, r1, r2, r3, r4, r5),
                                rough_surface(x_car), cse=True)
XX = np.linspace(0, 10, 100)
r11, r22, r33, r44, r55 = [par_map[key] for key in [r1, r2, r3, r4, r5]]
fig, ax = plt.subplots(figsize=(6, 2), layout='tight')
ax.plot(XX, rough_surface_lam(XX, r11, r22, r33, r44, r55))
ax.set_xlabel('[m]')
ax.set_ylabel('[m]')
_ = ax.set_title('Road Profile')
# %%
# Animate the Solution
# --------------------
fps = 100

def add_point_to_data(line, x, y):
    # to trace the path of the point.
    old_x, old_y = line.get_data()
    line.set_data(np.append(old_x, x), np.append(old_y, y))

state_vals, input_vals, _, _ = parse_free(solution, 8, 1, num_nodes,
                        variable_duration=True)
t_arr = np.linspace(t0, num_nodes*solution[-1], num_nodes)
state_sol = CubicSpline(t_arr, state_vals.T)
input_sol = CubicSpline(t_arr, input_vals.T)

xmin = -1.0
xmax = 1.0
ymin = -0.5
ymax = 1.5

# Define the points to be plotted.
coordinates = P_car.pos_from(O).to_matrix(N)
coordinates = coordinates.row_join(P_wheel.pos_from(O).to_matrix(N))

pL, pL_vals = zip(*par_map.items())
coords_lam = sm.lambdify(list(state_symbols) + [fx, c, k] + list(pL),
    coordinates, cse=True)

def init_plot():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('z', fontsize=15)

    # draw the road
    XX = np.linspace(0, 10, 200)
    street, = ax.plot(XX, rough_surface_lam(XX, r11, r22, r33, r44, r55),
            color='black', lw=0.75)

    # draw the wheel and the body and a line connecting them.
    line1 = ax.scatter([], [], color='red', marker='o', s=100) # wheel
    line2 = ax.scatter([], [], color='red', marker='o', s=500) # body
    line3, = ax.plot([], [], lw=0.5, color='red') # line connecting them

    # draw the arrows
    # driving force
    pfeil1 = ax.quiver([], [], [], [], color='green', scale=2000, width=0.008)
    # acceleratrion of the wheel
    pfeil2 = ax.quiver([], [], [], [], color='blue', scale=600, width=0.008)
    # acceleration of the body
    pfeil3 = ax.quiver([], [], [], [], color='magenta', scale=20, width=0.008)

    return fig, ax, line1, line2, line3, pfeil1, pfeil2, pfeil3, street

# Function to update the plot for each animation frame

def update(t):
    message = (f'running time {t:.2f} sec' +
        f'\n The blue arrow is the ' +
        f'accelerationon the wheel due to uneven street \n' +
        f'The magenta arrow is the acceleration of the body, magnified 30' +
        f' times \n' +
        f'The green arrow is the driving force / video is in slow motion')
    ax.set_title(message, fontsize=12)

    coords = coords_lam(*state_sol(t), input_sol(t), solution[-3],
                        solution[-2], *pL_vals)
    line1.set_offsets([0, coords[2, 1]])
    line2.set_offsets([0, coords[2, 0]])
    line3.set_data([0, 0], [coords[2, 0], coords[2, 1]])

    XX= np.linspace(-coords[0, 0]-1, 11-coords[0, 0], 200)
    YY = np.linspace(-1, 11, 200)
    street.set_data(XX, rough_surface_lam(YY, r11, r22, r33, r44, r55))

    pfeil1.set_offsets([coords[0, 0]*0, coords[2, 0]])
    pfeil1.set_UVC(input_sol(t), 0)

    pfeil2.set_offsets([-0.025, coords[2, 1]])
    pfeil2.set_UVC(0.0, state_sol(t)[7])

    pfeil3.set_offsets([+0.05, coords[2, 0]])
    pfeil3.set_UVC(0.0, state_sol(t)[6])

# sphinx_gallery_thumbnail_number = 4

# Create the animation.
fig, ax, line1, line2, line3, pfeil1, pfeil2, pfeil3, street = init_plot()
animation = FuncAnimation(fig, update, frames=np.arange(t0,
    num_nodes*solution[-1], 1 / fps), interval=18000/fps)
plt.show()
