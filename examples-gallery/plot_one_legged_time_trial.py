"""
Single human leg with four driving lumped muscles. The crank inertia and
resistance mimic the torque felt if accelerating the whole bicycle with rider
on flat ground.

The goal will be travel a specific distance in the shortest amount of time
given that the leg muscles have to coordinate and can only be excited from 0 to
1.

Second goal will then be to solve for crank length and seat height that gives
optimal performance.

"""
from opty.direct_collocation import Problem
from opty.utils import parse_free
from scipy.optimize import fsolve
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
import sympy.physics.biomechanics as bm
import sympy.physics.mechanics as me

t = me.dynamicsymbols._t

# bike frame, crank, pedal/foot, lower leg, upper leg
N, A, B, C, D = sm.symbols('N, A, B, C, D', cls=me.ReferenceFrame)

q1, q2, q3, q4 = me.dynamicsymbols('q1, q2, q3, q4', real=True)
u1, u2, u3, u4 = me.dynamicsymbols('u1, u2, u3, u4', real=True)
q = sm.Matrix([q1, q2, q3, q4])
u = sm.Matrix([u1, u2, u3, u4])

ls, lc, lf, ll, lu = sm.symbols('ls, lc, lf, ll, lu', real=True, positive=True)
lam, g, rk = sm.symbols('lam, g, rk', real=True)
mA, mB, mC, mD = sm.symbols('mA, mB, mC, mD')
IAzz, IBzz, ICzz, IDzz = sm.symbols('IAzz, IBzz, ICzz, IDzz')
J, m, rw, G, Cr, CD, rho, Ar = sm.symbols('J, m, rw, G, Cr, CD, rho, Ar')

A.orient_axis(N, N.z, q1)  # crank angle
B.orient_axis(A, A.z, q2)  # pedal/foot angle
C.orient_axis(B, B.z, q3)  # ankle angle
D.orient_axis(C, C.z, q4)  # knee angle

A.set_ang_vel(N, u1*N.z)
B.set_ang_vel(A, u2*A.z)
C.set_ang_vel(B, u3*B.z)
D.set_ang_vel(C, u4*C.z)

# P1 : crank center
# P2 : pedal center
# P3 : ankle
# P4 : knee
# P5 : hip
# P6 : seat center
# P7 : heel
P1, P2, P3, P4, P5, P6, P7 = sm.symbols('P1, P2, P3, P4, P5, P6, P7',
                                        cls=me.Point)
Ao, Bo, Co, Do = sm.symbols('Ao, Bo, Co, Do', cls=me.Point)

Ao.set_pos(P1, 0*A.x)
P2.set_pos(P1, lc*A.x)  # pedal center
Bo.set_pos(P2, lf/2*B.x)  # foot mass center
P3.set_pos(P2, lf*B.x)  # ankle
P7.set_pos(P2, 3*lf/2*B.x)  # heel
Co.set_pos(P3, ll/2*C.x)  # lower leg mass center
P4.set_pos(P3, ll*C.x)  # knee
Do.set_pos(P4, lu/2*D.x)  # upper leg mass center
P5.set_pos(P4, lu*D.x)  # hip
P6.set_pos(P1, -ls*sm.cos(lam)*N.x + ls*sm.sin(lam)*N.y)  # seat

P1.set_vel(N, 0)
P6.set_vel(N, 0)
Ao.v2pt_theory(P1, N, A)
P2.v2pt_theory(P1, N, A)
P7.v2pt_theory(P2, N, B)
Bo.v2pt_theory(P2, N, B)
P3.v2pt_theory(P2, N, B)
Co.v2pt_theory(P3, N, C)
P4.v2pt_theory(P3, N, C)
Do.v2pt_theory(P4, N, D)
P5.v2pt_theory(P4, N, D)

kindiff = sm.Matrix([ui - qi.diff() for ui, qi in zip(u, q)])

holonomic = (P5.pos_from(P1) - P6.pos_from(P1)).to_matrix(N)[:2, :]

IA = me.Inertia.from_inertia_scalars(Ao, A, 0, 0, IAzz + (J + m*rw**2)*G)
IB = me.Inertia.from_inertia_scalars(Bo, B, 0, 0, IBzz)
IC = me.Inertia.from_inertia_scalars(Co, C, 0, 0, ICzz)
ID = me.Inertia.from_inertia_scalars(Do, D, 0, 0, IDzz)

crank = me.RigidBody('crank', masscenter=Ao, frame=A, mass=mA, inertia=IA)
foot = me.RigidBody('foot', masscenter=Bo, frame=B, mass=mB, inertia=IB)
lower_leg = me.RigidBody('lower leg', masscenter=Co, frame=C, mass=mC,
                         inertia=IC)
upper_leg = me.RigidBody('upper leg', masscenter=Do, frame=D, mass=mD,
                         inertia=ID)

gravB = me.Force(Bo, -mB*g*N.y)
gravC = me.Force(Co, -mC*g*N.y)
gravD = me.Force(Do, -mD*g*N.y)

# TODO : Check why gear ratio is not applied to rolling resistance.
resistance = me.Torque(
    crank,
    # NOTE : we enforce later that u1 < 0 (forward pedaling), thus the
    # resistance should be a posistive torque to resist the negative speed
    # NOTE : using sm.sign() will break the constraing jacobian due taking the
    # derivative of sm.sign().
    (Cr*m*g*rw + CD*rho*Ar*G**2*u1**2*rw**3/2)*N.z
)

# add inertia due to bike and wheels to crank
# add resitance torque due to air drag and rolling resitance to crank
# omega = gear_ratio*ul
# T = TA/gear_ratio
# (2*J + m*r**2)*omega' = -sign(omega*r)*1/2*rho*Cd*(omega*r)**2 - Cr*m*g + T
# w*r = v -> w'*r = v' -> w' = v'/r


class ExtensorPathway(me.PathwayBase):
    def __init__(self, origin, insertion, axis_point, axis, parent_axis,
                 child_axis, radius, coordinate):
        """A custom pathway that wraps a circular arc around a pin joint.
        This is intended to be used for extensor muscles. For example, a
        triceps wrapping around the elbow joint to extend the upper arm at
        the elbow.

        Parameters
        ==========
        origin : Point
            Muscle origin point fixed on the parent body (A).
        insertion : Point
            Muscle insertion point fixed on the child body (B).
        axis_point : Point
            Pin joint location fixed in both the parent and child.
        axis : Vector
            Pin joint rotation axis.
        parent_axis : Vector
            Axis fixed in the parent frame (A) that is directed from the pin
            joint point to the muscle origin point.
        child_axis : Vector
            Axis fixed in the child frame (B) that is directed from the pin
            joint point to the muscle insertion point.
        radius : sympyfiable
            Radius of the arc that the muscle wraps around.
        coordinate : sympfiable function of time
            Joint angle, zero when parent and child frames align. Positive
            rotation about the pin joint axis, B with respect to A.

        Notes
        =====
        Only valid for coordinate >= 0.

        """
        super().__init__(origin, insertion)
        self.origin = origin
        self.insertion = insertion
        self.axis_point = axis_point
        self.axis = axis.normalize()
        self.parent_axis = parent_axis.normalize()
        self.child_axis = child_axis.normalize()
        self.radius = radius
        self.coordinate = coordinate
        self.origin_distance = axis_point.pos_from(origin).magnitude()
        self.insertion_distance = axis_point.pos_from(insertion).magnitude()
        self.origin_angle = sm.asin(self.radius/self.origin_distance)
        self.insertion_angle = sm.asin(self.radius/self.insertion_distance)

    @property
    def length(self):
        """Length of the pathway.
        Length of two fixed length line segments and a changing arc length
        of a circle.
        """
        angle = self.origin_angle + self.coordinate + self.insertion_angle
        arc_length = self.radius*angle
        origin_segment_length = self.origin_distance*sm.cos(self.origin_angle)
        insertion_segment_length = self.insertion_distance*sm.cos(
            self.insertion_angle)
        return origin_segment_length + arc_length + insertion_segment_length

    @property
    def extension_velocity(self):
        """Extension velocity of the pathway.
        Arc length of circle is the only thing that changes when the elbow
        flexes and extends.
        """
        return self.radius*self.coordinate.diff(me.dynamicsymbols._t)

    def to_loads(self, force_magnitude):
        """Loads in the correct format to be supplied to `KanesMethod`.
        Forces applied to origin, insertion, and P from the muscle wrapped
        over circular arc of radius r.
        """
        parent_tangency_point = me.Point('Aw')  # fixed in parent
        child_tangency_point = me.Point('Bw')  # fixed in child
        parent_tangency_point.set_pos(
            self.axis_point,
            -self.radius*sm.cos(self.origin_angle)*self.parent_axis.cross(
                self.axis)
            + self.radius*sm.sin(self.origin_angle)*self.parent_axis,
        )
        child_tangency_point.set_pos(
            self.axis_point,
            self.radius*sm.cos(self.insertion_angle)*self.child_axis.cross(
                self.axis)
            + self.radius*sm.sin(self.insertion_angle)*self.child_axis),
        parent_force_direction_vector = self.origin.pos_from(
            parent_tangency_point)
        child_force_direction_vector = self.insertion.pos_from(
            child_tangency_point)
        force_on_parent = (force_magnitude*
                           parent_force_direction_vector.normalize())
        force_on_child = (force_magnitude*
                          child_force_direction_vector.normalize())
        loads = [
            me.Force(self.origin, force_on_parent),
            me.Force(self.axis_point, -(force_on_parent + force_on_child)),
            me.Force(self.insertion, force_on_child),
        ]
        return loads


# four muscles
# NOTE : pathway only valid for q4>=0
knee_top_pathway = ExtensorPathway(Co, P5, P4, C.z, -C.x, D.x, rk, q4)
knee_top_act = bm.FirstOrderActivationDeGroote2016.with_defaults('knee_top')
knee_top_mus = bm.MusculotendonDeGroote2016.with_defaults('knee_top',
                                                          knee_top_pathway,
                                                          knee_top_act)
knee_bot_pathway = me.LinearPathway(Co, P5)
knee_bot_act = bm.FirstOrderActivationDeGroote2016.with_defaults('knee_bot')
knee_bot_mus = bm.MusculotendonDeGroote2016.with_defaults('knee_bot',
                                                          knee_bot_pathway,
                                                          knee_bot_act)
ankle_top_pathway = me.LinearPathway(Co, P2)
ankle_top_act = bm.FirstOrderActivationDeGroote2016.with_defaults('ankle_top')
ankle_top_mus = bm.MusculotendonDeGroote2016.with_defaults('ankle_top',
                                                           ankle_top_pathway,
                                                           ankle_top_act)
ankle_bot_pathway = me.LinearPathway(Co, P7)
ankle_bot_act = bm.FirstOrderActivationDeGroote2016.with_defaults('ankle_bot')
ankle_bot_mus = bm.MusculotendonDeGroote2016.with_defaults('ankle_bot',
                                                           ankle_bot_pathway,
                                                           ankle_bot_act)

loads = (
    knee_top_mus.to_loads() +
    knee_bot_mus.to_loads() +
    ankle_top_mus.to_loads() +
    ankle_bot_mus.to_loads() +
    [resistance, gravB, gravC, gravD]
)

qd_repl = {q1.diff(): u1, q2.diff(): u2, q3.diff(): u3, q4.diff(): u4}

kane = me.KanesMethod(
    N,
    (q1, q2),
    (u1, u2),
    kd_eqs=kindiff[:],
    q_dependent=(q3, q4),
    configuration_constraints=holonomic,
    velocity_constraints=me.msubs(holonomic.diff(me.dynamicsymbols._t),
                                  qd_repl),
    u_dependent=(u3, u4),
    constraint_solver='CRAMER',
)
bodies = (crank, foot, lower_leg, upper_leg)
Fr, Frs = kane.kanes_equations(bodies, loads)

muscle_diff_eq = sm.Matrix([
    knee_top_mus.a.diff() - knee_top_mus.rhs()[0, 0],
    knee_bot_mus.a.diff() - knee_bot_mus.rhs()[0, 0],
    ankle_top_mus.a.diff() - ankle_top_mus.rhs()[0, 0],
    ankle_bot_mus.a.diff() - ankle_bot_mus.rhs()[0, 0],
])

eom = kindiff.col_join(Fr + Frs).col_join(muscle_diff_eq).col_join(holonomic)

state_vars = (
    q1, q2, q3, q4, u1, u2, u3, u4,
    knee_top_mus.a,
    knee_bot_mus.a,
    ankle_top_mus.a,
    ankle_bot_mus.a,
)

num_nodes = 201
h = sm.symbols('h', real=True)

# objective
# minimize h
# gradient of h wrt free variables: states, unknown_inputs, unknown_pars, h
# so then there would be all zeros and a single 1 in the gradient for the last
# entry


def obj(free):
    return free[-1]


def gradient(free):
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad


# body segment inertia from https://nbviewer.org/github/pydy/pydy-tutorial-human-standing/blob/master/notebooks/n07_simulation.ipynb
par_map = {
    Ar: 0.55,  # m^2, Tab 5.1, pg 188 Wilson 2004, Upright commuting bike
    CD: 1.15,  # unitless, Tab 5.1, pg 188 Wilson 2004, Upright commuting bike
    Cr: 0.006,  # unitless, Tab 5.1, pg 188 Wilson 2004, Upright commuting bike
    G: 2.0,
    IAzz: 0.0,
    IBzz: 0.01,  # guess, TODO
    ICzz: 0.101,  # lower_leg_inertia [kg*m^2]
    IDzz: 0.282,  # upper_leg_inertia [kg*m^2],
    J: 2*0.1524,  # from Browser Jason's thesis (rear wheel times 2)
    g: 9.81,
    lam: np.deg2rad(75.0),
    lc: 0.175,  # crank length [m]
    lf: 0.08,  # pedal to ankle [m]
    ll: 0.611,  # lower_leg_length [m]
    ls: 0.8,  # seat tube length [m]
    lu: 0.424,  # upper_leg_length [m],
    m: 175.0,  # kg
    #mA: 0.0,  # not in eom
    mB: 1.0,  # foot mass [kg] guess TODO
    mC: 6.769,  # lower_leg_mass [kg]
    mD: 17.01,  # upper_leg_mass [kg],
    rho: 1.204,  # kg/m^3
    rk: 0.04,  # m
    rw: 0.3,  # m
    ankle_bot_mus.F_M_max: 200.0,
    ankle_bot_mus.l_M_opt: np.nan,
    ankle_bot_mus.l_T_slack: np.nan,
    ankle_top_mus.F_M_max: 200.0,
    ankle_top_mus.l_M_opt: np.nan,
    ankle_top_mus.l_T_slack: np.nan,
    knee_bot_mus.F_M_max: 500.0,
    knee_bot_mus.l_M_opt: np.nan,
    knee_bot_mus.l_T_slack: np.nan,
    knee_top_mus.F_M_max: 1000.0,
    knee_top_mus.l_M_opt: np.nan,
    knee_top_mus.l_T_slack: np.nan,
}

p = np.array(list(par_map.keys()))
p_vals = np.array(list(par_map.values()))


# solve for maximal knee extension and flat foot
q1_ext = -par_map[lam]  # aligned with seat post
q2_ext = 3*np.pi/2  # foot perpendicular to crank
eval_holonomic = sm.lambdify((q, p), holonomic)
q3_ext, q4_ext = fsolve(lambda x: eval_holonomic([q1_ext, q2_ext, x[0], x[1]],
                                                 p_vals).squeeze(),
                        x0=np.deg2rad([-100.0, 20.0]))
q_ext = np.array([q1_ext, q2_ext, q3_ext, q4_ext])



eval_ankle_top_len = sm.lambdify((q, p), ankle_top_pathway.length)
eval_ankle_bot_len = sm.lambdify((q, p), ankle_bot_pathway.length)
eval_knee_top_len = sm.lambdify((q, p), knee_top_pathway.length)
eval_knee_bot_len = sm.lambdify((q, p), knee_bot_pathway.length)
# length of muscle path when fully extended
par_map[ankle_top_mus.l_T_slack] = eval_ankle_top_len(q_ext, p_vals)
par_map[ankle_bot_mus.l_T_slack] = eval_ankle_bot_len(q_ext, p_vals)
par_map[knee_top_mus.l_T_slack] = eval_knee_top_len(q_ext, p_vals)
par_map[knee_bot_mus.l_T_slack] = eval_knee_bot_len(q_ext, p_vals)
par_map[ankle_top_mus.l_M_opt] = par_map[ankle_top_mus.l_T_slack] + 0.01
par_map[ankle_bot_mus.l_M_opt] = par_map[ankle_bot_mus.l_T_slack] + 0.01
par_map[knee_top_mus.l_M_opt] = par_map[knee_top_mus.l_T_slack] + 0.01
par_map[knee_bot_mus.l_M_opt] = par_map[knee_bot_mus.l_T_slack] + 0.01

# solve for initial configuration
q1_0 = 0.0  # horizontal and forward
q2_0 = np.pi  # toe forward and foot aligned with the crank (horizontal)
eval_holonomic = sm.lambdify((q, p), holonomic)
q3_0, q4_0 = fsolve(lambda x: eval_holonomic([q1_0, q2_0, x[0], x[1]],
                                             p_vals).squeeze(),
                    x0=np.deg2rad([-90.0, 90.0]))
q_0 = np.array([q1_0, q2_0, q3_0, q4_0])

crank_revs = 10

instance_constraints = (
    q1.replace(t, 0*h) - q1_0,  # crank starts horizontal
    q2.replace(t, 0*h) - q2_0,  # foot/pedal aligned with crank
    q3.replace(t, 0*h) - q3_0,
    q4.replace(t, 0*h) - q4_0,
    q1.replace(t, (num_nodes - 1)*h) + crank_revs*2*np.pi,  # travel number of revolutions
    #u1.replace(t, 0*h),  # start stationary
    u2.replace(t, 0*h),  # start stationary
    u3.replace(t, 0*h),  # start stationary
    u4.replace(t, 0*h),  # start stationary
    knee_top_mus.a.replace(t, 0*h),
    knee_bot_mus.a.replace(t, 0*h),
    ankle_top_mus.a.replace(t, 0*h),
    ankle_bot_mus.a.replace(t, 0*h),
)

bounds = {
    q1: (-crank_revs*2*np.pi, 0.0),  # can only pedal forward
    # ankle angle, q3=-np.pi/4: ankle maximally plantar flexed, q3=-3*pi/2:
    # ankle maximally dorsiflexed
    q3: (-np.pi/2, -np.pi/4),
    # knee angle, q4 = 0: upper and lower leg aligned, q4 = pi/2: knee is
    # flexed 90 degs
    q4: (0.0, 3*np.pi/2),
    u1: (-20.0, 0.0),  # about 200 rpm
    ankle_bot_mus.e: (0.0, 1.0),
    ankle_top_mus.e: (0.0, 1.0),
    knee_bot_mus.e: (0.0, 1.0),
    knee_top_mus.e: (0.0, 1.0),
}

problem = Problem(
    obj,
    gradient,
    eom,
    state_vars,
    num_nodes,
    h,
    known_parameter_map=par_map,
    instance_constraints=instance_constraints,
    bounds=bounds,
)

# segmentation fault if I set initial guess to zero
initial_guess = np.random.random(problem.num_free)
q1_guess = np.linspace(0.0, -crank_revs*2*np.pi, num=num_nodes)
q2_guess = np.linspace(0.0, crank_revs*2*np.pi, num=num_nodes)
u1_guess = np.linspace(0.0, -6.0, num=num_nodes)
u1_guess[num_nodes//2:] = -3.0
u2_guess = np.linspace(0.0, 6.0, num=num_nodes)
u2_guess[num_nodes//2:] = 3.0
initial_guess[0*num_nodes:1*num_nodes] = q1_guess
initial_guess[1*num_nodes:2*num_nodes] = q2_guess
initial_guess[4*num_nodes:5*num_nodes] = u1_guess
initial_guess[5*num_nodes:6*num_nodes] = u2_guess

solution, info = problem.solve(initial_guess)

problem.plot_objective_value()
problem.plot_constraint_violations(solution)
problem.plot_trajectories(solution)

xs, us, ps = parse_free(solution, len(state_vars), 4, num_nodes)

plot_points = [P1, P2, P7, P3, P4, P6, P1]
coordinates = P1.pos_from(P1).to_matrix(N)
for Pi in plot_points[1:]:
    coordinates = coordinates.row_join(Pi.pos_from(P1).to_matrix(N))
eval_coordinates = sm.lambdify((q, p), coordinates)

mus_points = [P7, Co, P2, Co, P6]
mus_coordinates = P7.pos_from(P1).to_matrix(N)
for Pi in mus_points[1:]:
    mus_coordinates = mus_coordinates.row_join(Pi.pos_from(P1).to_matrix(N))
eval_mus_coordinates = sm.lambdify((q, p), mus_coordinates)


def plot_configuration(q_vals, p_vals, ax=None):
    if ax is None:
        fig, ax = plt.subplots(layout='constrained')

    x, y, _ = eval_coordinates(q_vals, p_vals)
    xm, ym, _ = eval_mus_coordinates(q_vals, p_vals)
    leg_lines, = ax.plot(x, y, 'o-')
    mus_lines, = ax.plot(xm, ym, 'o-', color='red',)
    crank_circle = plt.Circle((0.0, 0.0), par_map[lc], fill=False,
                              linestyle='--')
    knee_circle = plt.Circle((x[4], y[4]), par_map[rk], color='red',
                             fill=False)
    ax.add_patch(crank_circle)
    ax.add_patch(knee_circle)
    ax.set_aspect('equal')
    return ax, fig, leg_lines, mus_lines


plot_configuration(q_ext, p_vals)
ax, fig, leg_lines, mus_lines = plot_configuration(q_0, p_vals)


def animate(i):
    qi = xs[0:4, i]
    x, y, _ = eval_coordinates(qi, p_vals)
    xm, ym, _ = eval_mus_coordinates(qi, p_vals)
    leg_lines.set_data(x, y)
    mus_lines.set_data(xm, ym)


interval_value = solution[-1]
ani = animation.FuncAnimation(fig, animate, num_nodes,
                              interval=int(interval_value*1000))

plt.show()
