import sympy as sm
import sympy.physics.mechanics as me
import sympy.physics.biomechanics as bm

# bike frame, crank, pedal/foot, lower leg, upper leg
N, A, B, C, D = sm.symbols('N, A, B, C, D', cls=me.ReferenceFrame)

q1, q2, q3, q4 = me.dynamicsymbols('q1, q2, q3, q4')
u1, u2, u3, u4 = me.dynamicsymbols('u1, u2, u3, u4')
q = sm.Matrix([q1, q2, q3, q4])
u = sm.Matrix([u1, u2, u3, u4])

ls, lc, lf, ll, lu = sm.symbols('ls, lc, lf, ll, lu', real=True, positive=True)
lam, g, rk = sm.symbols('lambda, g, rk', real=True)
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
P6.set_pos(P1, -ls*sm.sin(lam)*N.x + ls*sm.cos(lam)*N.y)  # seat

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

resistance = me.Torque(
    crank,
    (-Cr*m*g*rw - sm.sign(G*u1*rw)*CD*rho*Ar*G**2*u1**2*rw**3/2)*N.z
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
knee_top_pathway = ExtensorPathway(P5, Co, P4, C.z, D.x, -C.x, rk, q4)
knee_top_act = bm.FirstOrderActivationDeGroote2016.with_defaults('knee_top')
knee_top_mus = bm.MusculotendonDeGroote2016.with_defaults('knee_top',
                                                          knee_top_pathway,
                                                          knee_top_act)
knee_bot_pathway = me.LinearPathway(P5, Co)
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

kane = me.KanesMethod(
    N,
    (q1, q2),
    (u1, u2),
    kd_eqs=(
        u1 - q1.diff(),
        u2 - q2.diff(),
        u3 - q3.diff(),
        u4 - q4.diff(),
    ),
    q_dependent=(q3, q4),
    configuration_constraints=holonomic,
    velocity_constraints=holonomic.diff(me.dynamicsymbols._t),
    u_dependent=(u3, u4),
)
bodies = (crank, foot, lower_leg, upper_leg)
Fr, Frs = kane.kanes_equations(bodies, loads)
