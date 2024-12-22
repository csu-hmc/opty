# %%
"""
Drone Flight in Tube
====================

Given a cuboid shaped drone of dimensions l x w x d with propellers at each
corner in a uniform gravitational field, find the propeller thrust trajectories
that will take it from a starting point to an ending point with minimum power.
The drone must not leave a tube defined by a curve (centerline) in space and a
radius.

The only interesting is maybe this:
( In what follows all components are w.r.t. the inertial frame N.)
The curve is given as X(r) = (f(r, params), g(r, params), h(r, params)),
where r is the parameter of the curve. Let :math:`r_1` be the parameter of the
curve where the distance of thedrone from the curve is closest. Then
:math:`\\dfrac{df}{dr}, \\dfrac{dg}{dr}, \\dfrac{dh}{dr}`
at :math:`R = r_1` is the tangential vecor of the curve at the point of closest.
So, I form the equation of the plane, which is perpendicular to the curve at the
point of closest, and contains the point of the drone. The intersection of the
curve and the plane gives the point of closest distance of the curve from the
drone.
This leads to a nonlinear equation for :math:`r_1`, which I add to the equations
of motion by declaring a new state variable :math:`cut param`, just another
name for :math:`r_1`.

In addition, I introduce a new state variable :math:`dist`, which is the distance
of the drone from the curve. The reason I do this is so I can bound it to be less
than a certain value, which is the radius of the tube.

**Constants**

- :math:`m` : drone mass, [kg]
- :math:`l` : length (along body x) [m]
- :math:`w` : width (along body y) [m]
- :math:`d` : depth (along body z) [m]
- :math:`c` : viscous friction coefficient of air [Nms]
- :math:`a1, a2, a3` : parameters of the curve (centerline) [m]
- :math:`radius` : radius of the tube [m]
- :math:`max_z` : maximum height of the drone [m]


**States**

- :math:`x, y, z` : position of mass center [m]
- :math:`v_1, v_2, v_3` : body fixed speed of mass center [m]
- :math:`q_0, q_1, q_2, q_3` : quaternion measure numbers [rad]
- :math:`w_x, w_y, w_z` : body fixed angular rates [rad/s]
- :math:`dist` : distance of the drone from the curve [m]
- :math:`cut param` : parameter of the curve where the distance is closest.
- :math:`cutdt = \\dfrac{d}{dt} cut param` : I set :math:`cutdt \\geq 0`, so it never flies backwards.


**Specifieds**

- :math:`F_1, F_2, F_3, F_4` : propeller propulsion forces [N]

"""

import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
from opty import Problem, parse_free, create_objective_function
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from IPython.display import HTML
matplotlib.rcParams['animation.embed_limit'] = 2**128

import time

# %%
# Generate the equations of motion of the system.
m, l, w, d, g, c = sm.symbols('m, l, w, d, g, c', real=True)
x, y, z, vx, vy, vz = me.dynamicsymbols('x, y, z, v_x, v_y v_z', real=True)
q0, q1, q2, q3 = me.dynamicsymbols('q0, q1, q2, q3', real=True)
u0, wx, wy, wz = me.dynamicsymbols('u0, omega_x, omega_y, omega_z', real=True)
F1, F2, F3, F4 = me.dynamicsymbols('F1, F2, F3, F4', real=True)
t = me.dynamicsymbols._t

O, Ao, P1, P2, P3, P4 = sm.symbols('O, A_o, P1, P2, P3, P4', cls=me.Point)
N, A = sm.symbols('N, A', cls=me.ReferenceFrame)

A.orient_quaternion(N, (q0, q1, q2, q3))

Ao.set_pos(O, x*N.x + y*N.y + z*N.z)
P1.set_pos(Ao, l/2*A.x + w/2*A.y)
P2.set_pos(Ao, -l/2*A.x + w/2*A.y)
P3.set_pos(Ao, l/2*A.x - w/2*A.y)
P4.set_pos(Ao, -l/2*A.x - w/2*A.y)

N_w_A = A.ang_vel_in(N)
N_v_P = Ao.pos_from(O).dt(N)

kinematical = sm.Matrix([
    vx - N_v_P.dot(A.x),
    vy - N_v_P.dot(A.y),
    vz - N_v_P.dot(A.z),
    u0 - q0.diff(t),
    wx - N_w_A.dot(A.x),
    wy - N_w_A.dot(A.y),
    wz - N_w_A.dot(A.z),
])

A.set_ang_vel(N, wx*A.x + wy*A.y + wz*A.z)

O.set_vel(N, 0)
Ao.set_vel(N, vx*A.x + vy*A.y + vz*A.z)
P1.v2pt_theory(Ao, N, A)
P2.v2pt_theory(Ao, N, A)
P3.v2pt_theory(Ao, N, A)
P4.v2pt_theory(Ao, N, A)

# x: l, y: w, z: d
IA = me.inertia(A, m*(w**2 + d**2)/12, m*(l**2 + d**2)/12, m*(l**2 + w**2)/12)
drone = me.RigidBody('A', Ao, A, m, (IA, Ao))

prop1 = (P1, F1*A.z)
prop2 = (P2, F2*A.z)
prop3 = (P3, F3*A.z)
prop4 = (P4, F4*A.z)
# use a linear simplification of air drag for continuous derivatives
grav = (Ao, -m*g*N.z - c*Ao.vel(N))

# enforce the unit quaternion
holonomic = sm.Matrix([q0**2 + q1**2 + q2**2 + q3**2 - 1])

kane = me.KanesMethod(
    N,
    (x, y, z, q1, q2, q3),
    (vx, vy, vz, wx, wy, wz),
    kd_eqs=kinematical,
    q_dependent=(q0,),
    u_dependent=(u0,),
    configuration_constraints=holonomic,
    velocity_constraints=holonomic.diff(t),
)

fr, frstar = kane.kanes_equations([drone], [prop1, prop2, prop3, prop4, grav])

eom = kinematical.col_join(fr + frstar).col_join(holonomic)

# %%
def plane(vector, point, x1, x2, x3):
    """
    It returns the plane equations, whose normal vector is vector, and the
    point is in the plane.
    - point: given as tuple (p1, p2, p3) in the N frame
    - vector: given as tuple (n1, n2, n3) in the N frame
    - x1, x2, x3: symbols of the coordinates in the plane
    The plane is returned in coordinate form:
        n1*x1 + n2*x2 + n3*x3 - (n1*p1 + n2*p2 + n3*p3) = 0
    """
    p1, p2, p3 = point[0], point[1], point[2]
    n1, n2, n3 = vector[0], vector[1], vector[2]
    return n1*x1 + n2*x2 + n3*x3 - (n1*p1 + n2*p2 + n3*p3)

# %%
def intersect(r1, curve, point, x1, x2, x3):
    """
    returns a non-linear equation for r1, which, if inserted into curve, gives
    the point of intersection.
    curve: given as tuple(f(r, params), g(r, params), g(r, params)),
        where r is the parameter of the curve, and params is the parameters
        of the curve
    point: given as tuple (p1, p2, p3), not on the curve, in N frame.
    """
    f, g, h = curve[0], curve[1], curve[2]
    fdr, gdr, hdr = f.diff(r), g.diff(r), h.diff(r)

    vector = (fdr.subs(r, r1), gdr.subs(r, r1), hdr.subs(r, r1))
    intersect_eqn = me.msubs(plane(vector, point, x1, x2, x3),
            {x1: f.subs(r, r1), x2: g.subs(r, r1), x3: h.subs(r, r1)})
    return intersect_eqn

# %%
def distance(N, r1, curve, point):
    """
    returns the distance of the curve to the point
    curve: given as tuple(f(r, params), g(r, params), g(r, params)),
        where r is the parameter of the curve, and params is the parameters
        of the curve, in the N frame.
    point: given as tuple(p1, p2, p3), not on the curve. in the N frame.
    """
    f, g, h = curve[0].subs(r, r1), curve[1].subs(r, r1), curve[2].subs(r, r1)

    P11 = f*N.x + g*N.y + h*N.z
    P21 = point[0]*N.x + point[1]*N.y + point[2]*N.z

    dist = (P11 - P21).magnitude()
    return dist

# %%
dist, cut_param, cutdt = me.dynamicsymbols('dist, cut_param, cutdt', real=True)
a1, a2, a3 = sm.symbols('a1, a2, a3', real=True)
x1, x2, x3 = sm.symbols('x1, x2, x3', real=True)

r = sm.symbols('r', real=True)

# %%
# Define the curve.
curve = (a1*sm.sin(2*np.pi*r), a2*sm.cos(2*np.pi*r), a3*r)

h1 = intersect(cut_param, curve, (x, y, z), x1, x2, x3)
h2 = distance(N, cut_param, curve, (x, y, z))
eom = eom.col_join(sm.Matrix([h1, dist-h2, -cutdt + cut_param.diff(t)]))

print(f'Shape of eoms is {eom.shape}, they contain {sm.count_ops(eom)} operations')

# %%
t0, duration = 0, 5.0
num_nodes = 101
interval_value = duration/(num_nodes - 1)

# %%
# Provide some values for the constants.
radius = 1.0
max_z = 12.0
start_param = 0.1
par_map = {
    c: 0.5*0.1*1.2,
    d: 0.1,
    g: 9.81,
    l: 1.0,
    m: 2.0,
    w: 0.5,
    a1: 5.0,
    a2: 5.0,
    a3: 5.0,
}

state_symbols = (x, y, z, q0, q1, q2, q3, vx, vy, vz, u0, wx, wy, wz,
            cut_param, dist, cutdt)
specified_symbols = (F1, F2, F3, F4)

# %%
# Specify the objective function and form the gradient.
obj_func = sm.Integral(F1**2 + F2**2 + F3**2 + F4**2, t)

sm.pprint(obj_func)
obj, obj_grad = create_objective_function(
    obj_func,
    state_symbols,
    specified_symbols,
    tuple(),
    num_nodes,
    interval_value,
)


# %%
# Specify the symbolic instance constraints.
eval_curve = sm.lambdify((r, a1, a2, a3), curve, cse=True)

instance_constraints = (
    # start level
    x.func(0.0) - eval_curve(start_param, par_map[a1], par_map[a2],
                par_map[a3])[0],
    y.func(0.0) - eval_curve(start_param, par_map[a1], par_map[a2],
                par_map[a3])[1],
    z.func(0.0) - eval_curve(start_param, par_map[a1], par_map[a2],
                par_map[a3])[2] + 0.5,
    cut_param.func(0.0) - start_param,
    dist.func(0.0),
    q0.func(0.0) - 1.0,
    q1.func(0.0),
    q2.func(0.0),
    q3.func(0.0),

    # end level
    x.func(duration) - eval_curve(max_z/par_map[a3], par_map[a1], par_map[a2],
                par_map[a3])[0],
    y.func(duration) - eval_curve(max_z/par_map[a3], par_map[a1], par_map[a2],
                par_map[a3])[1],
    z.func(duration) - max_z,
#    q0.func(duration) - 1.0,
#    q1.func(duration),
#    q2.func(duration),
#    q3.func(duration),

    # stationary at start and finish
    vx.func(0.0),
    vy.func(0.0),
    vz.func(0.0),
    u0.func(0.0),
    wx.func(0.0),
    wy.func(0.0),
    wz.func(0.0),
    vx.func(duration),
    vy.func(duration),
    vz.func(duration),
    u0.func(duration),
    wx.func(duration),
    wy.func(duration),
    wz.func(duration),
)
print('len(instance_constraints) =', len(instance_constraints))
# %%
# Add some physical limits to the propeller thrust.
grenze = 100.0
bounds = {
    F1: (-grenze, grenze),
    F2: (-grenze, grenze),
    F3: (-grenze, grenze),
    F4: (-grenze, grenze),
    dist: (0.0, radius),
    cut_param: (0.0, np.inf),
    z: (0.0, max_z),
    cutdt: (0.0, 5.0),
}

# %%
# Create the optimization problem and set any options.
prob = Problem(obj, obj_grad, eom, state_symbols,
               num_nodes, interval_value,
               known_parameter_map=par_map,
               instance_constraints=instance_constraints,
               bounds=bounds)

prob.add_option('nlp_scaling_method', 'gradient-based')
prob.add_option('max_iter', 12000)

# %%
# Give a guess of a direct route with constant thrust.

initial_guess = np.zeros(prob.num_free)
x_guess, y_guess, z_guess = eval_curve(np.linspace(0.0, 10/par_map[a3],
            num=num_nodes), par_map[a1], par_map[a2], par_map[a3])
initial_guess[0*num_nodes:1*num_nodes] = x_guess
initial_guess[1*num_nodes:2*num_nodes] = y_guess
initial_guess[2*num_nodes:3*num_nodes] = z_guess
initial_guess[-4*num_nodes:] = 10.0  # constant thrust

initial_guess = np.load('drone_in_tube_solution.npy')
# %%
# Find an optimal solution.
for _ in range(1):
    zeit = time.time()
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    print(info['obj_val'])
    print('Time taken:', time.time()-zeit, '\n')

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
# Animate the motion of the drone.
time = np.linspace(0.0, duration, num=num_nodes)
coordinates = Ao.pos_from(O).to_matrix(N)
for point in [P1, Ao, P2, Ao, P3, Ao, P4]:
    coordinates = coordinates.row_join(point.pos_from(O).to_matrix(N))
eval_point_coords = sm.lambdify((state_symbols, specified_symbols,
                                 list(par_map.keys())), coordinates, cse=True)

xs, us, ps = parse_free(solution, len(state_symbols), len(specified_symbols),
                        num_nodes)
coords = []
for xi, ui in zip(xs.T, us.T):
    coords.append(eval_point_coords(xi, ui, list(par_map.values())))
coords = np.array(coords)  # shape(n, 3, 8)

# This function is to draw the 'tube', which the drome must not leave
def frenet_frame(f, g, h, r, num_points=100, tube_radius=bounds[dist][1]+0.25):
    # Given to me by chatGPT
    # Parameterize the curve
    r_vals = r
    f_vals = f(r_vals)
    g_vals = g(r_vals)
    h_vals = h(r_vals)

    # Compute derivatives for tangent vectors
    f_prime = np.gradient(f_vals, r_vals)
    g_prime = np.gradient(g_vals, r_vals)
    h_prime = np.gradient(h_vals, r_vals)
    tangent_vectors = np.stack([f_prime, g_prime, h_prime], axis=-1)
    tangent_vectors /= np.linalg.norm(tangent_vectors, axis=1)[:, np.newaxis]

    # Approximate normal and binormal
    normal_vectors = np.cross(tangent_vectors, np.roll(tangent_vectors, 1,
        axis=0))
    normal_vectors /= np.linalg.norm(normal_vectors, axis=1)[:, np.newaxis]
    binormal_vectors = np.cross(tangent_vectors, normal_vectors)

    # Create the circular cross-section
    theta = np.linspace(0, 2 * np.pi, 30)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=-1)

    # Generate tube surface
    X, Y, Z = [], [], []
    for i in range(num_points):
        frame = np.stack([normal_vectors[i], binormal_vectors[i]], axis=-1)
        offset = circle @ frame.T * tube_radius
        X.append(f_vals[i] + offset[:, 0])
        Y.append(g_vals[i] + offset[:, 1])
        Z.append(h_vals[i] + offset[:, 2])

    return np.array(X), np.array(Y), np.array(Z)


def frame(i):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x1, y1, z1 = eval_point_coords(xs[:, i], us[:, i], list(par_map.values()))

    drone_lines, = ax.plot(x1, y1, z1,
                color='black',
                marker='o', markerfacecolor='blue', markersize=4)
    Ao_path, = ax.plot(coords[:i, 0, 0], coords[:i, 1, 0], coords[:i, 2, 0],
                color='black', lw=0.75)

    title_template = 'Time = {:1.2f} s'
    title_text = ax.set_title(title_template.format(time[i]))

    ax.set_xlim(-par_map[a1]-1, par_map[a1]+1)
    ax.set_ylim(-par_map[a2]-1, par_map[a2]+1)
    ax.set_zlim(0.0, bounds[z][1]+1)
#    ax.set_aspect('equal')

    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_zlabel(r'$z$ [m]')

    return fig, ax, title_text, drone_lines, Ao_path


fig, ax, title_text, drone_lines, Ao_path  = frame(0)

# Define the curve functions
f = lambda r: eval_curve(r, par_map[a1], par_map[a2], par_map[a3])[0]
g = lambda r: eval_curve(r, par_map[a1], par_map[a2], par_map[a3])[1]
h = lambda r: eval_curve(r, par_map[a1], par_map[a2], par_map[a3])[2]

# Generate the tube
curve_param = np.linspace(0, max_z/par_map[a3], 100)
X, Y, Z = frenet_frame(f, g, h, r=curve_param)

# Plot the tube
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='grey', alpha=0.1,
        edgecolor='red')

#ax.plot(eval_curve(curve_param, par_map[a1], par_map[a2], par_map[a3])[0],
#        eval_curve(curve_param, par_map[a1], par_map[a2], par_map[a3])[1],
#        eval_curve(curve_param, par_map[a1], par_map[a2], par_map[a3])[2],
#        color='green')

def animate(i):
    title_text.set_text('Time = {:1.2f} s'.format(time[i]))
    drone_lines.set_data_3d(coords[i, 0, :], coords[i, 1, :], coords[i, 2, :])
    Ao_path.set_data_3d(coords[:i, 0, 0], coords[:i, 1, 0], coords[:i, 2, 0])

ani = animation.FuncAnimation(fig, animate, range(0, len(time), 2),
                              interval=int(interval_value*2000))


display(HTML(ani.to_jshtml()))

# %%
