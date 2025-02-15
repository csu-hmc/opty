r"""
Particle Flight in Tube
=======================

A particle of mass :math:`m` is moving from a starting point to an ending
point, subject to a viscous friction force and to a uniform gravitational
field. The particle must not leave a tube defined by a curve (centerline) in
space and a radius. At one point during the motion, it must pass through a
narrow gate, modelled as a circle.

Interesting maybe this:

(In what follows all components are with respect to the inertial frame N.) The
curve is given as :math:`X(r) = (f(r, \textrm{params}), g(r, \textrm{params}),
h(r, \textrm{params}))`, where :math:`r` is the parameter of the curve.  Let
:math:`\textrm{cut}_{\textrm{param}}` be the parameter of the curve where the
distance of the particle from the curve is closest. Then :math:`(
\dfrac{df}{dr}, \dfrac{dg}{dr}, \dfrac{dh}{dr} ) |_{r =
\textrm{cut}_{\textrm{param}}}` is the tangential vector on the curve at the
point of closest distance from the particle.

The equation of the plane, which is perpendicular to the curve at the point of
closest distance and contains the point of the particle is formed.  The
intersection of the curve and the plane gives the point of closest distance of
the curve from the particle.  This leads to a nonlinear equation for
:math:`\textrm{cut}_{\textrm{param}}`, which is added to the equations of
motion by declaring a new state variable :math:`\textrm{cut}_{\textrm{param}}`.

In addition, a new state variable ``dist`` is introduced, which is the distance
from the particle from the curve. This way the distance particle to the
centerline of the tube may be bounded to be less than the radius of the tube.

The particle must pass through a narrow gate at an intermediate time. The gate
is modeled as a circle with its center on the curve. As opty presently does not
allow inequalities in instance constraints, a new state variable ``gate``, and
a new specified variable, :math:`\textrm{gate}_h` are introduced.

**Constants**

- :math:`m` : particle mass, [kg]
- :math:`c` : viscous friction coefficient of air [Nms]
- :math:`a_1, a_2, a_3` : parameters of the curve (centerline) [m]
- :math:`\textrm{radius}` : radius of the tube [m]
- :math:`\textrm{max}_z` : maximum height of the particle [m]

**States**

- :math:`x, y, z` : position of the particle [m]
- :math:`v_1, v_2, v_3` : speed of particle [m]
- :math:`\textrm{dist}` : distance of the particle from the curve [m]
- :math:`\textrm{cut}_{\textrm{param}}` : parameter of the curve where the
  distance is closest.
- :math:`\textrm{cut}_{\textrm{dt}} = \dfrac{d}{dt} \textrm{cut}_{param}`:
  :math:`\textrm{cut}_{\textrm{dt}} \geq 0`, so the particle never flies
  backwards.
- :math:`\textrm{gate}` : explained further down.

**Specifieds**

- :math:`F_x, F_y, F_z` : forces acting on the particle [N]
- :math:`\textrm{gate}_h` : explained further down.

"""
import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
from scipy.interpolate import CubicSpline
from opty import Problem, parse_free, create_objective_function
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# %%
# Equations of Motion
# -------------------
m, g, c = sm.symbols('m, g, c', real=True)
x, y, z, vx, vy, vz = me.dynamicsymbols('x, y, z, v_x, v_y v_z', real=True)
Fx, Fy, Fz = me.dynamicsymbols('F_x, F_y, F_z', real=True)
t = me.dynamicsymbols._t

O, Dmc = sm.symbols('O, Dmc', cls=me.Point)
N = sm.symbols('N', cls=me.ReferenceFrame)

Dmc.set_pos(O, x*N.x + y*N.y + z*N.z)
Dmc.set_vel(N, vx*N.x + vy*N.y + vz*N.z)

kinematical = sm.Matrix([
    vx - x.diff(t),
    vy - y.diff(t),
    vz - z.diff(t),
])

O.set_vel(N, 0)

point = me.Particle('point', Dmc, m)

grav = (Dmc, -m*g*N.z - c*Dmc.vel(N))
force = (Dmc, Fx*N.x + Fy*N.y + Fz*N.z)

q_ind = [x, y, z]
u_ind = [vx, vy, vz]

kane = me.KanesMethod(
    N,
    q_ind=q_ind,
    u_ind=u_ind,
    kd_eqs=kinematical,
)

fr, frstar = kane.kanes_equations([point], [force, grav])

eom = kinematical.col_join(fr + frstar)


# %%
# Define some functions to get the distance of the particle from the center
# curve of the tube.
def plane(vector, point, x1, x2, x3):
    """
    Returns the plane equations, whose normal vector is vector, and the point
    is in the plane.

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
# Enlarge the equations of motion.
#
# A.
# :math:`h_1` is a nonlinear equation for :math:`\textrm{cut}_\textrm{param}`,
# the parameter for the point on the curve closest to the particle. :math:`h_2`
# is the distance of the particle from the curve, so :math:`\textrm{dist}` can
# be bound to be less than the radius of the tube.
dist, cut_param, cutdt = me.dynamicsymbols('dist, cut_param, cutdt', real=True)
a1, a2, a3 = sm.symbols('a1, a2, a3', real=True)
x1, x2, x3 = sm.symbols('x1, x2, x3', real=True)

r = sm.symbols('r', real=True)

curve = [a1*sm.sin(2*np.pi*r), a2*sm.cos(2*np.pi*r), a3*r]

h1 = intersect(cut_param, curve, (x, y, z), x1, x2, x3)
h2 = distance(N, cut_param, curve, (x, y, z))
eom = eom.col_join(sm.Matrix([h1, dist-h2, -cutdt + cut_param.diff(t)]))

# %%
# B.
# At an intermediate time :math:`t_m`, the particle must go through a narrow
# gate. Set :math:`\textrm{gate} = \textrm{gate}_h \cdot \textrm{dist}`, set
# :math:`\textrm{gate}_h \geq 1`, and :math:`\textrm{gate}(t_m)` = radius of
# the gate.
gate, gate_h = me.dynamicsymbols('gate, gate_h', real=True)
eom = eom.col_join(sm.Matrix([-gate + gate_h*dist]))
print(f'the shape of the eoms is {eom.shape}, and they contain '
      f'{sm.count_ops(eom)} operations.')

# %%
# Set up the Optimization and Solve It
# ------------------------------------
state_symbols = (x, y, z, vx, vy, vz, cut_param, dist, cutdt, gate)
specified_symbols = (Fx, Fy, Fz, gate_h)

t0, duration = 0, 5.0
num_nodes = 101
interval_value = duration/(num_nodes - 1)
tm = interval_value*50

# %%
# Provide some values for the constants.
radius = 1.0
max_z = 12.0
start_param = 0.001
par_map = {
    c: 0.5*0.1*1.2,
    g: 9.81,
    m: 2.0,
    a1: 5.0,
    a2: 5.0,
    a3: 5.0,
}

# %%
# Specify the objective function and form the gradient.
obj_func = sm.Integral(Fx**2 + Fy**2 + Fz**2, t)

sm.pprint(obj_func)
obj, obj_grad = create_objective_function(
    obj_func,
    state_symbols,
    specified_symbols,
    tuple(),
    num_nodes,
    interval_value,
    time_symbol=t,
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
                             par_map[a3])[2],
    cut_param.func(0.0) - start_param,
    dist.func(0.0),
    # intermediate level
    gate.func(tm) - radius/4.0,
    # end level
    x.func(duration) - eval_curve(max_z/par_map[a3], par_map[a1], par_map[a2],
                                  par_map[a3])[0],
    y.func(duration) - eval_curve(max_z/par_map[a3], par_map[a1], par_map[a2],
                                  par_map[a3])[1],
    z.func(duration) - max_z,
    # stationary at start and finish
    vx.func(0.0),
    vy.func(0.0),
    vz.func(0.0),
    vx.func(duration),
    vy.func(duration),
    vz.func(duration),
)
print('len(instance_constraints) =', len(instance_constraints))

# %%
# Add some physical limits to the force, other bounds as needed to realize
# the inequalities.
grenze = 100.0
epsilon = 1.e-5
bounds = {
    Fx: (-grenze-epsilon, grenze+epsilon),
    Fy: (-grenze-epsilon, grenze+epsilon),
    Fz: (-grenze-epsilon, grenze+epsilon),
    dist: (0.0-epsilon, radius+epsilon),
    cut_param: (0.0-epsilon, np.inf),
    z: (0.0-epsilon, max_z+epsilon),
    cutdt: (0.0-epsilon, 5.0+epsilon),
    gate_h: (1.0-epsilon, 10.0+epsilon),
}

# %%
# Create the optimization problem and set any options.
prob = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    num_nodes, interval_value,
    known_parameter_map=par_map,
    instance_constraints=instance_constraints,
    bounds=bounds,
    time_symbol=t,
)

prob.add_option('max_iter', 3000)

# %%
# Give a guess of a plausible route with constant thrust. Here the
# solution of a previous run is used as initial guess.
initial_guess = np.zeros(prob.num_free)
x_guess, y_guess, z_guess = eval_curve(np.linspace(0.0, max_z/par_map[a3],
                                                   num=num_nodes), par_map[a1],
                                       par_map[a2], par_map[a3])
initial_guess[0*num_nodes:1*num_nodes] = x_guess
initial_guess[1*num_nodes:2*num_nodes] = y_guess
initial_guess[2*num_nodes:3*num_nodes] = z_guess
initial_guess[-4*num_nodes:] = 10.0  # constant thrust

initial_guess = np.load('particle_in_tube_solution.npy')

# %%
# Find an optimal solution.
for _ in range(1):
    solution, info = prob.solve(initial_guess)
    initial_guess = solution
    print(info['status_msg'])
    print(info['obj_val'])

# %%
# Plot the optimal state and input trajectories.
_ = prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
_ = prob.plot_constraint_violations(solution)

# %%
# Plot the objective function as a function of optimizer iteration.
_ = prob.plot_objective_value()

# %%
# Animate the Motion of the Particle
# ----------------------------------
state_vals, input_vals, _ = parse_free(solution, len(state_symbols),
                                       len(specified_symbols), num_nodes)
t_arr = np.linspace(t0, duration, num_nodes)
state_sol = CubicSpline(t_arr, state_vals.T)
input_sol = CubicSpline(t_arr, input_vals.T)

fx, fy, fz = sm.symbols('fx, fy, fz', real=True)
Pf = me.Point('Pf')
Pf.set_pos(Dmc, fx*N.x + fy*N.y + fz*N.z)
time = np.linspace(0.0, duration, num=num_nodes)

coordinates = Dmc.pos_from(O).to_matrix(N)
coordinates = coordinates.row_join(Pf.pos_from(O).to_matrix(N))
eval_coords = sm.lambdify((state_symbols, fx, fy, fz, list(par_map.keys())),
                          coordinates, cse=True)


# %%
# This function is to draw the tube which the particle must not leave.
def frenet_frame(f, g, h, r, num_points=100,
                 tube_radius=bounds[dist][1] + 0.25):
    # From the internet
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


# %%
# this function draws a circle.
def plot_3d_circle(ax, center, radius, normal, num_points=100):
    # Essentially from the internet.
    """
    Plots a 3D circle based on the given center, radius, and normal vector.

    Parameters:
    - center: The center of the circle (numpy array of shape (3,))
    - radius: The radius of the circle (scalar)
    - normal: The normal vector perpendicular to the circle's plane
      (numpy array of shape (3,))
    - num_points: The number of points used to plot the circle (default is 100)
    """
    # Generate points on a circle in the xy-plane (z=0)
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(theta)

    # Create the points in the plane of the circle
    circle_points = np.array([x, y, z])

    # Normalize the normal vector to get the orientation of the circle plane
    normal = normal / np.linalg.norm(normal)

    # Find two perpendicular vectors to the normal
    # (to define the plane of the circle)
    # One of them can be chosen arbitrarily as long as it's not parallel
    # to the normal
    v1 = np.array([1, 2, 3]) / np.sqrt(1**2 + 2**2 + 3**2)
    v1 = np.cross(normal, v1)
    v1 = v1 / np.sqrt(np.sum([v1[i]**2 for i in range(3)]))

    # Find the second perpendicular vector by crossing the normal and v1
    v2 = np.cross(normal, v1)

    # Parametrize the circle using v1 and v2 and translate to the center
    circle_3d = (center[:, np.newaxis] +
                 v1[:, np.newaxis]*circle_points[0, :] +
                 v2[:, np.newaxis]*circle_points[1, :])

    # Plot the 3D circle
    ax.plot(circle_3d[0, :], circle_3d[1, :], circle_3d[2, :], color='blue')


# %%
# Animate the motion of the particle.
def init():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    line1, = ax.plot([], [], [], marker='o', color='black', markersize=7)
    line2, = ax.plot([], [], [], color='black', lw=1)
    pfeil = ax.quiver([], [], [], [], [], [], color='green')

    ax.set_xlim(-par_map[a1]-1, par_map[a1]+1)
    ax.set_ylim(-par_map[a2]-1, par_map[a2]+1)
    ax.set_zlim(0.0, bounds[z][1]+1)

    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_zlabel(r'$z$ [m]')

    return fig, ax, line1, line2, pfeil


fig, ax, line1, line2, pfeil = init()

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

# Plot the gate, a circle
center = np.array(eval_curve(1.2, par_map[a1], par_map[a2], par_map[a3]))
curvedt = [curve[i].diff(r) for i in range(3)]
eval_curvedt = sm.lambdify((r, a1, a2, a3), curvedt, cse=True)
normal = np.array(eval_curvedt(1.2, par_map[a1], par_map[a2], par_map[a3]))
normal = normal / np.sqrt([np.sum(normal[i]**2) for i in range(3)])
plot_3d_circle(ax, center=center, radius=radius/3.0, normal=normal)


def animate(i):
    global line1, pfeil, line2, koords
    fx1 = input_sol(time[i])[0]
    fy1 = input_sol(time[i])[1]
    fz1 = input_sol(time[i])[2]
    skale = 13.0
    coords = eval_coords(state_sol(time[i]), fx1, fy1, fz1,
                         list(par_map.values()))
    line1.set_data_3d([coords[0, 0]], [coords[1, 0]], [coords[2, 0]])
    pfeil.remove()
    pfeil = ax.quiver([coords[0, 0]],
                      [coords[1, 0]], [coords[2, 0]],
                      [(coords[0, 1]-coords[0, 0])/skale],
                      [(coords[1, 1]-coords[1, 0])/skale],
                      [(coords[2, 1]-coords[2, 0])/skale], color='green')

    koords = []
    for k in range(i):
        koords.append(eval_coords(state_sol(time[k]), fx1, fy1, fz1,
                                  list(par_map.values())))
    line2.set_data_3d([koords[k][0][0] for k in range(i)],
                      [koords[k][1][0] for k in range(i)],
                      [koords[k][2][0] for k in range(i)])

    ax.set_title(
        f'Running time = {time[i]:.2f} s. \n The small blue circle '
        f'is the gate \n The green arrow is proportional to the force.'
    )


ani = animation.FuncAnimation(fig, animate, range(0, len(time), 2),
                              interval=int(interval_value*2000))

# %%
fig, ax, line1, line2, pfeil = init()
animate(35)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='grey', alpha=0.1,
                edgecolor='red')
plot_3d_circle(ax, center=center, radius=radius/3.0, normal=normal)

# sphinx_gallery_thumbnail_number = 4
plt.show()
