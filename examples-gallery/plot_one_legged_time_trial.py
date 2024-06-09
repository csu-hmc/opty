import sympy as sm
import sympy.physics.mechanics as me

# bike frame, crank, pedal/foot, lower leg, upper leg
N, A, B, C, D = sm.symbols('N, A, B, C, D', cls=me.ReferenceFrame)

q1, q2, q3, q4 = me.dynamicsymbols('q1, q2, q3, q4')
u1, u2, u3, u4 = me.dynamicsymbols('u1, u2, u3, u4')
q = sm.Matrix([q1, q2, q3, q4])
u = sm.Matrix([u1, u2, u3, u4])

ls, lc, lf, ll, lu = sm.symbols('ls, lc, lf, ll, lu', real=True, positive=True)
lam, g = sm.symbols('lambda, g')
mA, mB, mC, mD = sm.symbols('mA, mB, mC, mD')
IAzz, IBzz, ICzz, IDzz = sm.symbols('IAzz, IBzz, ICzz, IDzz')

A.orient_axis(N, N.z, q1)  # crank angle
B.orient_axis(A, A.z, q2)  # pedal/foot angle
C.orient_axis(B, B.z, q3)  # ankle angle
D.orient_axis(C, C.z, q4)  # knee angle

A.set_vel(N, u1*N.z)
B.set_vel(A, u2*N.z)
C.set_vel(B, u3*N.z)
D.set_vel(C, u4*N.z)

P1, P2, P3, P4, P5, P6 = sm.symbols('P1, P2, P3, P4, P5, P6', cls=me.Point)
Ao, Bo, Co, Do = sm.symbols('Ao, Bo, Co, Do', cls=me.Point)

Ao.set_pos_from(P1, 0*A.x)
P2.set_pos_from(P1, lc*A.x)  # pedal center
Bo.set_pos_from(P2, lf/2*B.x)  # foot mass center
P3.set_pos_from(P2, lf*B.x)  # ankle
Co.set_pos_from(P3, ll/2*C.x)  # lower leg mass center
P4.set_pos_from(P3, ll*C.x)  # knee
Do.set_pos_from(P4, lu/2*C.x)  # upper leg mass center
P5.set_pos_from(P4, lu*D.x)  # hip
P6.set_pos_from(P1, ls*sm.cos(lam)*N.x + ls*sm.sin(lam)*N.y)  # seat

P1.set_vel(N, 0)
P6.set_vel(N, 0)
Ao.v2pt_theory(P1, N, A)
P2.v2pt_theory(P1, N, A)
Bo.v2pt_theory(P2, N, B)
P3.v2pt_theory(P2, N, B)
Co.v2pt_theory(P3, N, C)
P4.v2pt_theory(P3, N, C)
Do.v2pt_theory(P4, N, D)
P5.v2pt_theory(P4, N, D)

kindiff = sm.Matrix([ui - qi.diff() for ui, qi in zip(u, q)])

holonomic = (P5.pos_from(P1) - P6.pos_from(P1)).to_matrix(N)[:2]

IA = me.Inertia.from_inertia_scalars(Ao, A, 0, 0, IAzz)
IB = me.Inertia.from_inertia_scalars(Bo, B, 0, 0, IBzz)
IC = me.Inertia.from_inertia_scalars(Co, C, 0, 0, ICzz)
ID = me.Inertia.from_inertia_scalars(Do, D, 0, 0, IDzz)

crank = me.RigidBody('crank', masscenter=Ao, frame=A, mass=mA, inertia=IA)
foot = me.RigidBody('foot', masscenter=Bo, frame=B, mass=mB, inertia=IB)
lower_leg = me.RigidBody('lower leg', masscenter=Co, frame=C, mass=mC,
                         inertia=IC)
upper_leg = me.RigidBody('upper leg', masscenter=Do, frame=D, mass=mD,
                         inertia=ID)

gravB = me.Force(foot, -mB*g*N.y)
gravC = me.Force(foot, -mC*g*N.y)
gravD = me.Force(foot, -mD*g*N.y)

load = me.Torque(crank, TA)

# add inertia due to bike and wheels to crank
# add resitance torque due to air drag and rolling resitance to crank
# omega = gear_ratio*ul
# T = TA/gear_ratio
# (2*J + m*r**2)*omega' = -sign(omega*r)*1/2*rho*Cd*(omega*r)**2 - Cr*m*g + T
# w*r = v -> w'*r = v' -> w' = v'/r

# four muscles
knee_top_pathway =
knee_bot_pathway = me.LinearPathway(butt, thigh)
ankle_top
ankle_bot_pathway = me.LinearPathway(calf, heel)
