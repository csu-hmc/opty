# %%
"""
Upright a double pendulum
==========================

A double pendulum is rotably attached to a cart which can move
along the horizontal X axis. The goal is to get the double
pendulum to an upright position in the shortest time possible,
given an upper limit on the absolute value of the force that
can be applied to the cart.
Gravity points in the negative Y direction.

**Constants**

- m1:     mass of the cart [kg]
- m2:     mass of the first pendulum [kg]
- m3:     mass of the second pendulum [kg]
- lx:     length of the first pendulum rods [m]
- iZZ1:   moment of inertia of the first pendulum about its end point [kg*m^2]
- iZZ2:   moment of inertia of the second pendulum about its end point [kg*m^2]
- g:      acceleration due to gravity [m/s^2]

**States**

- q1:     position of the cart along the X axis [m]
- q2:     angle of the first pendulum with respect to the vertical [rad]
- q3:     angle of the second pendulum with respect to the first [rad]
- u1:     speed of the cart along the X axis [m/s]
- u2:     angular speed of the first pendulum [rad/s]
- u3:     angular speed of the second pendulum [rad/s]

**Specifieds**

- f:      force applied to the cart [N]

"""
import sympy.physics.mechanics as me
from collections import OrderedDict
import numpy as np
import sympy as sm
from opty.direct_collocation import Problem
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches

# %%
# Generate the nonholonomic equations of motion of the system.
N, A1, A2 = sm.symbols('N A1, A2', cls = me.ReferenceFrame)
t = me.dynamicsymbols._t
O, P1, P2, P3  = sm.symbols('O P1 P2 P3', cls = me.Point)                               
O.set_vel(N, 0)                                                                        
q1, q2, q3, u1, u2, u3, F = me.dynamicsymbols('q1 q2 q3 u1 u2 u3 F')                   
lx, m1, m2, m3, g, iZZ1, iZZ2 = sm.symbols('lx, m1, m2, m3 g, iZZ1, iZZ2')             

A1.orient_axis(N, q2, N.z)
A1.set_ang_vel(N, u2 * N.z)
A2.orient_axis(N, q3, N.z)
A2.set_ang_vel(N, u3 * N.z)

P1.set_pos(O, q1 * N.x)
P2.set_pos(P1, lx * A1.x)
P2.v2pt_theory(P1, N, A1)

P3.set_pos(P2, lx * A2.x)
P3.v2pt_theory(P2, N, A2)

P1a = me.Particle('P1a', P1, m1)                                                

I = me.inertia(A1, 0, 0, iZZ1)
P2a = me.RigidBody('P2a', P2, A1, m2, (I, P2))                                 

I = me.inertia(A2, 0, 0, iZZ2)
P3a = me.RigidBody('P3a', P3, A2, m3, (I, P3))                                  
BODY = [P1a, P2a, P3a]

FL = [(P1, F * N.x - m1*g*N.y), (P2, -m2*g*N.y), (P3, -m3*g*N.y)]                   
kd = sm.Matrix([q1.diff(t) - u1, q2.diff(t) - u2, q3.diff(t) - u3])                   

q_ind = [q1, q2, q3]
u_ind = [u1, u2, u3]

KM = me.KanesMethod(N, q_ind=q_ind, u_ind=u_ind, kd_eqs=kd)
(fr, frstar) = KM.kanes_equations(BODY, FL)
EOM = kd.col_join(fr + frstar)
sm.pprint(sm.trigsimp(EOM))                

# %%
# Define various objects to be use in the optimization problem.
h = sm.symbols('h')         

state_symbols = tuple((*q_ind, *u_ind))
constant_symbols = (lx, m1, m2, m3, g, iZZ1, iZZ2)
specified_symbols = (F,)

target_angle = np.pi /2.0                     
num_nodes = 400     

duration = (num_nodes - 1) * h
interval_value = h

# %%
# Specify the known system parameters.
par_map = OrderedDict()
par_map[lx]  = 2.0
par_map[m1]  = 1.0
par_map[m2]  = 1.0
par_map[m3]  = 1.0
par_map[g]   = 9.81
par_map[iZZ1] = 2.0
par_map[iZZ2] = 2.0

def obj(free):
    """Minimize h, the time interval between nodes."""
    return free[-1]

def obj_grad(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.
    return grad

# %%
# set the initial and final states to form the instance constraints.
initial_state_constraints = {
    q1: 0.,  
    q2: -np.pi/2., 
    q3: -np.pi/2., 
    u1: 0.,
    u2: 0., 
    u3: 0.
    }

final_state_constraints  = {
    q2: target_angle, 
    q3: target_angle, 
    u1: 0., 
    u2: 0., 
    u3: 0.
    }

instance_constraints = (
) + tuple(
    xi.subs({t: 0}) - xi_val for xi, xi_val in initial_state_constraints.items()
) + tuple(
    xi.subs({t: duration}) - xi_val for xi, xi_val in final_state_constraints.items()
)
# %%
# bounding h > 0 helps avoid 'solutions' with h < 0.
bounds = {F: (-150., 150.), q1: (-5, 5), h: (1.e-5, 1.0)} 

# %%
# Create an optimization problem and solve it.
prob = Problem(obj, 
    obj_grad, 
    EOM, 
    state_symbols, 
    num_nodes, 
    interval_value,
    known_parameter_map=par_map,
    instance_constraints=instance_constraints,
    time_symbol=t,
    bounds=bounds)

# Initial guess.
WERT = (0., 0., 0., 0.0, 0.0, 0.0, 0.0)
initial_guess =  np.array([wert  for wert in WERT for _ in range(num_nodes)] + [0.01])
# allows to change the number of iterations.
# standard is 3000
prob.add_option('max_iter', 3000)

# Find the optimal solution.
solution, info = prob.solve(initial_guess)

prob.plot_objective_value()
print('message from optimizer:', info['status_msg'])
print('Iterations needed',len(prob.obj_value))
print(f"objective value {info['obj_val']:.3e}")
print('h value:', solution[-1])
print('\n')

# %%
# Plot the accuracy of the solution, and the state trajectories.
prob.plot_trajectories(solution)
# %%
prob.plot_constraint_violations(solution)

# %%
# animate the solution.
P1_x = np.empty(num_nodes)
P1_y = np.empty(num_nodes)
P2_x = np.empty(num_nodes)
P2_y = np.empty(num_nodes)
P3_x = np.empty(num_nodes)
P3_y = np.empty(num_nodes)

P1_loc = [me.dot(P1.pos_from(O), uv) for uv in [N.x, N.y]]
P2_loc = [me.dot(P2.pos_from(O), uv) for uv in [N.x, N.y]]
P3_loc = [me.dot(P3.pos_from(O), uv) for uv in [N.x, N.y]]

qL = q_ind + u_ind
pL_vals = list(constant_symbols)
P1_loc_lam = sm.lambdify(qL + pL_vals, P1_loc, cse=True)
P2_loc_lam = sm.lambdify(qL + pL_vals, P2_loc, cse=True)
P3_loc_lam = sm.lambdify(qL + pL_vals, P3_loc, cse=True)

for i in range(num_nodes):
    q_1 = solution[i]
    q_2 = solution[i + num_nodes]
    q_3 = solution[i + 2 * num_nodes]
    u_1 = solution[i + 3 * num_nodes]
    u_2 = solution[i + 4 * num_nodes]
    u_3 = solution[i + 5 * num_nodes]
    P1_x[i], P1_y[i] = P1_loc_lam(q_1, q_2, q_3, u_1, u_2, u_3, *list(par_map.values()))
    P2_x[i], P2_y[i] = P2_loc_lam(q_1, q_2, q_3, u_1, u_2, u_3, *list(par_map.values()))
    P3_x[i], P3_y[i] = P3_loc_lam(q_1, q_2, q_3, u_1, u_2, u_3, *list(par_map.values()))
    

# needed to give the picture the right size.
xmin = min(np.min(P1_x),np.min(P2_x), np.min(P3_x))
xmax = max(np.max(P1_x), np.max(P2_x), np.max(P3_x))
ymin = min(np.min(P1_y), np.min(P2_y), np.min(P3_y))
ymax = max(np.max(P1_y), np.max(P2_y), np.max(P3_y))

width, height = par_map[lx]/3., par_map[lx]/3.

def animate_pendulum(time, P1_x, P1_y, P2_x, P2_y):
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'aspect': 'equal'})
    
    ax.axis('on')
    ax.set_xlim(xmin - 1., xmax + 1.)
    ax.set_ylim(ymin - 1., ymax + 1.)
    
    ax.set_xlabel('X direction', fontsize=15)
    ax.set_ylabel('Y direction', fontsize=15)
    ax.axhline(0, color='black', lw=2)

    line1, = ax.plot([], [], 'o-', lw=0.5, color='blue')    
    line2, = ax.plot([], [], 'o-', lw=0.5, color='green')                    
    
    recht = patches.Rectangle((P1_x[0] - width/2, P1_y[0] - height/2), 
            width=width, height=height, fill=True, color='red', ec='black')
    ax.add_patch(recht)
    return fig, ax, line1, line2, recht

duration = (num_nodes - 1) * solution[-1]
times = np.linspace(0.0, duration, num=num_nodes)
fig, ax, line1, line2, recht = animate_pendulum(times, P1_x, P1_y, P2_x, P2_y)

def animate(i):
    message = (f'running time {times[i]:.2f} sec')
    ax.set_title(message, fontsize=15)
    recht.set_xy((P1_x[i] - width/2., P1_y[i] - height/2.))

    wert_x = [P1_x[i], P2_x[i]]
    wert_y = [P1_y[i], P2_y[i]]               
    line1.set_data(wert_x, wert_y)    

    wert_x = [P2_x[i], P3_x[i]]
    wert_y = [P2_y[i], P3_y[i]]
    line2.set_data(wert_x, wert_y)              
    return line1, line2,

anim = animation.FuncAnimation(fig, animate, frames=num_nodes,
                                   interval=solution[-1]*1000.0,
                                   blit=False)

## %%
# A frame from the animation.
fig, ax, line1, line2, recht = animate_pendulum(times, P1_x, P1_y, P2_x, P2_y)

# sphinx_gallery_thumbnail_number = 5
animate(100)

plt.show()