
# %%
""""
sliding a block on a road
=========================
A block, modeled as a particle is sliding on a road to cross a
hill. The block is subject to gravity and speed dependent
friction. 
Gravity points in the negative Y direction.
A force tangential to the road is applied to the block.
Three objective functions to me minimized may be selected:

- selektion = 0: time to reach the end point is minimized
- selektion = 1: time to reach the end point is minimized and 
  the energy consumed is minimized.
- selektion = 2: energy consumed is minimized.

**Constants**

- m: mass of the block [kg]
- g: acceleration due to gravity [m/s**2]
- reibung: coefficient of friction [N/(m*s)]
 -a, b: paramenters determining the shape of the road.

**States**

- x: position of the block [m]
- ux: velocity of the block [m/s]

**Specifieds**

- F: force applied to the block [N]

"""
import sympy.physics.mechanics as me
from collections import OrderedDict
import numpy as np
import sympy as sm
from opty.direct_collocation import Problem
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %%
# select the objective function to be minimized
selektion = 2

# %%
# define the road
def strasse(x, a, b):
    return a * x**2 * sm.exp((b - x))

# %%
# set up Kane's EOMs
N = me.ReferenceFrame('N')                                                                                           
O = me.Point('O')                                                                                                         
O.set_vel(N, 0)                                                                                                        
t = me.dynamicsymbols._t

P0 = me.Point('P0')                                                             
x = me.dynamicsymbols('x')    
ux = me.dynamicsymbols('u_x')                                                                                                                                                    
F = me.dynamicsymbols('F')                                                               

m, g, reibung = sm.symbols('m, g, reibung')     
a, b = sm.symbols('a b')                                                                         

P0.set_pos(O, x * N.x + strasse(x, a, b) * N.y)
P0.set_vel(N, ux * N.x + strasse(x, a, b).diff(x)*ux * N.y)
BODY = [me.Particle('P0', P0, m)]

# The control force and the friction are acting in the direction of 
# the tangent at the street at the point whre the particle is.
alpha = sm.atan(strasse(x, a, b).diff(x))
FL = [(P0, -m*g*N.y + F*(sm.cos(alpha)*N.x + sm.sin(alpha)*N.y) - 
       reibung*ux*(sm.cos(alpha)*N.x + sm.sin(alpha)*N.y))]     

kd = sm.Matrix([ux - x.diff(t)])      

q_ind = [x]
u_ind = [ux]
 
KM = me.KanesMethod(N, q_ind=q_ind, u_ind=u_ind, kd_eqs=kd)
(fr, frstar) = KM.kanes_equations(BODY, FL)
EOM = kd.col_join(fr + frstar) 
EOM.simplify()
sm.pprint(EOM)

# %%
# set up the objects needed for the optimization
state_symbols = tuple((x, ux))
laenge = len(state_symbols)
constant_symbols = (m, g, reibung, a, b)
specified_symbols = (F,)
                          
num_nodes = 300       

if selektion == 2:
    duration = 6.
    interval_value = duration / (num_nodes - 1)
else:
    h = sm.symbols('h')
    duration = (num_nodes - 1) * h
    interval_value = h

# %%
# Specify the known system parameters.
par_map = OrderedDict()
par_map[m] = 1.0                            
par_map[g] = 9.81                               
par_map[reibung] = 0.                                  
par_map[a] = 1.5                                  
par_map[b] = 2.5                                  

# %%
# pick the objective function selected above

# verstaerkung is a factor to set the weight of the speed in the
# objective function relative to the weight of the energy (which is 1)
verstaerkung = 2.e5

if selektion == 1:
    def obj(free): 
# verstaerkung is a factor to set the weight of the speed in the
# objective function relative to the weight of the energy (which is 1)
        verstaerkung = 2.e5
        Fx = free[laenge * num_nodes: (laenge + 1) * num_nodes] 
        return free[-1] * np.sum(Fx**2) + free[-1] * verstaerkung

    def obj_grad(free):  
        grad = np.zeros_like(free)
        l1 = laenge * num_nodes
        l2 = (laenge + 1) * num_nodes
        grad[l1: l2] = 2.0 * free[l1: l2] * free[-1]
        grad[-1] = 1 * verstaerkung
        return grad

elif selektion == 0:
    def obj(free):
        return free[-1]

    def obj_grad(free):
        grad = np.zeros_like(free)
        grad[-1] = 1.
        return grad
    
elif selektion == 2:
    def obj(free): 
        Fx = free[laenge * num_nodes: (laenge + 1) * num_nodes] 
        return interval_value * np.sum(Fx**2)

    def obj_grad(free):
        grad = np.zeros_like(free)
        l1 = laenge * num_nodes
        l2 = (laenge + 1) * num_nodes
        grad[l1: l2] = 2.0 * free[l1: l2] * interval_value
        return grad
else:
    raise Exception('selektion must be 0, 1, 2')

# %%
# create the optimization problem and solve it
t0, tf = 0.0, duration              

# pick the integration method. backward euler and midpoint are the choices.
# backward euler is the default.
methode = 'backward euler' 

if selektion in (0, 1): 
    initial_guess = np.array(list(np.ones((len(state_symbols) 
        + len(specified_symbols)) * num_nodes) * 0.01) + [0.02])
else:
    initial_guess = np.ones((len(state_symbols) + 
        len(specified_symbols)) * num_nodes) * 0.01   

initial_state_constraints = {x: 0., ux: 0.}

final_state_constraints = {x: 10., ux: 0.}    
    
instance_constraints = ( 
    tuple(xi.subs({t: t0}) - xi_val for xi, xi_val in initial_state_constraints.items()) + 
    tuple(xi.subs({t: tf}) - xi_val for xi, xi_val in final_state_constraints.items())
    )

# %%
# forcing h > 0 avoids negative h as 'solutions'.
if selektion in (0, 1):
    bounds = {F: (-15., 15.), x: (initial_state_constraints[x], 
    final_state_constraints[x]), ux: (0., 1000.), h:(1.e-5, 1.)}
else:
    bounds = {F: (-15., 15.), x: (initial_state_constraints[x], 
    final_state_constraints[x]), ux: (0., 1000.)}
               
prob = Problem(obj, 
    obj_grad, 
    EOM, 
    state_symbols, 
    num_nodes, 
    interval_value,
    known_parameter_map=par_map,
    instance_constraints=instance_constraints,
    bounds=bounds,
    integration_method=methode)

# set max number of iterations. Default is 3000.
prob.add_option('max_iter', 3000)          

solution, info = prob.solve(initial_guess)
print('message from optimizer:', info['status_msg'])
print(f'objective value {obj(solution):,.1f}')
if selektion in (0, 1):
    print(f'optimal h value is: {solution[-1]:.3f}')
prob.plot_objective_value()

# %%
# plot the accuracy of the solution and the trajectories
prob.plot_constraint_violations(solution)

# %%
prob.plot_trajectories(solution)

# %% 
# animate the solution
if selektion in (0, 1):
    duration = (num_nodes - 1) * solution[-1]
times = np.linspace(0.0, duration, num=num_nodes)
interval_value = duration / (num_nodes - 1)

strasse1 = strasse(x, a, b)
strasse_lam = sm.lambdify((x, a, b), strasse1, cse=True)    

P0_x = solution[:num_nodes]
P0_y = strasse_lam(P0_x, par_map[a], par_map[b])

# find the force vector applied to the block
alpha = sm.atan(strasse(x, a, b).diff(x))
Pfeil = [F*sm.cos(alpha),  F*sm.sin(alpha)]
Pfeil_lam = sm.lambdify((x, F, a, b), Pfeil, cse=True)     

l1 = laenge * num_nodes
l2 = (laenge + 1) * num_nodes
Pfeil_x = Pfeil_lam(P0_x, solution[l1: l2], par_map[a], par_map[b])[0]
Pfeil_y = Pfeil_lam(P0_x, solution[l1: l2], par_map[a], par_map[b])[1]

# needed to give the picture the right size.
xmin = np.min(P0_x)
xmax = np.max(P0_x)
ymin = np.min(P0_y)
ymax = np.max(P0_y)


def initialize_plot():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin-1, xmax + 1.)
    ax.set_ylim(ymin-1, ymax + 1.)
    ax.set_aspect('equal')
    ax.set_xlabel('X-axis [m]')
    ax.set_ylabel('Y-axis [m]')

    if selektion == 0:
        msg = f'speed was optimized'
    elif selektion == 1:
        msg = f'speed and energy optimized, weight of speed = {verstaerkung:.1e}'
    else:
        msg = f'energy optimized'

    ax.grid()
    strasse_x = np.linspace(xmin, xmax, 100)
    ax.plot(strasse_x, strasse_lam(strasse_x, par_map[a], par_map[b]), 
        color='black', linestyle='-', linewidth=1)  
    ax.axvline(initial_state_constraints[x], color='r', linestyle='--', linewidth=1)                              
    ax.axvline(final_state_constraints[x], color='green', linestyle='--', linewidth=1)                           

# Initialize the block and the arrow
    line1, = ax.plot([], [], color='blue', marker='o', markersize=12)                                              
    pfeil   = ax.quiver([], [], [], [], color='green', scale=35, width=0.004)
    return fig, ax, line1, pfeil, msg

fig, ax, line1, pfeil, msg = initialize_plot()

# Function to update the plot for each animation frame
def update(frame):
    message = (f'running time {times[frame]:.2f} sec \n' +  
        f'the red line is the initial position, the green line is the final position \n' + 
        f'the green arrow is the force acting on the block \n' +
        f'{msg}' )
    ax.set_title(message, fontsize=12)
    
    line1.set_data([P0_x[frame]], [P0_y[frame]])
    pfeil.set_offsets([P0_x[frame], P0_y[frame]])                                                  
    pfeil.set_UVC(Pfeil_x[frame], Pfeil_y[frame])
    return line1, pfeil

animation = FuncAnimation(fig, update, frames=range(len(P0_x)), 
    interval=1000*interval_value, blit=True)

## %%
# A frame from the animation.
fig, ax, line1, pfeil, msg = initialize_plot()

# sphinx_gallery_thumbnail_number = 5
update(100 )

plt.show()


