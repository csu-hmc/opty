#!/usr/bin/env python

from collections import OrderedDict

from sympy import symbols, trigsimp, Matrix, simplify, solve
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point
from sympy.physics.mechanics import inertia, RigidBody, KanesMethod, msubs

sym_kwargs = {'positive': True, 'real': True}
dynamicsymbols._t = symbols('t', **sym_kwargs)


class PlanarStandingHumanOnMovingPlatform():
    """Generates the symbolic equations of motion of a 2D planar two body
    model representing a human standing on a antero-posteriorly moving
    platform similar to the one found in Park et. al 2004.

    Variables
    =========
    theta_a :
        angle between the ankle and the legs, plantar flexion is positive
    theta_h :
        angle between the legs and the hip, extension is positive
    omega_a :
        angular rate of the legs wrt to the ankle
    omega_h :
        angular rate of the torso wrt to the legs
    T_a :
        torque applied between the foot and the leg, positive torque causes
        plantar flexion
    T_h :
        torque applied between the leg and the torso, positive torque causes
        extension
    a :
        Antero-posterior acceleration of the platform and foot.

    Constant Parameters
    ===================

    l_L : leg length
    d_L : leg_com_length
    d_T : torso_com_length
    m_L : leg_mass
    m_T : torso_mass
    I_L : leg_inertia
    I_T : torso_inertia
    g : acceleration due to gravity

    Equations of Motion
    ===================

    The generalized coordinates:

    q = [theta_a, theta_h]

    q = [0, 0] is the nominal configuration: upright standing

    The generalized speeds:

    u = [omega_a, omega_h]

    The states:

    x = [q, u]

    The specified inputs:

    r = [T_a, T_h, a]

    x' = f(x, r)

    and a form linearized about the upright equilibrium:

    x' = A * x + B * r

    Closed Loop Equations of Motion
    ===============================

    Additional constants
    --------------------
    k_{T_a-theta_a}, k_{T_a-theta_h}, k_{T_a-omega_a}, k_{T_a-omega_h}
    k_{T_h-theta_a}, k_{T_h-theta_h}, k_{T_h-omega_a}, k_{T_h-omega_h}

    x = [q, u]
    r = [a]
    T = [T_a, T_h]

    x' = f(x, r)

    where

    T = -k * x


    """

    def create_states(self):

        theta_a, theta_h = dynamicsymbols('theta_a, theta_h')
        omega_a, omega_h = dynamicsymbols('omega_a, omega_h')

        self.coordinates = OrderedDict()
        self.coordinates['theta_a'] = theta_a
        self.coordinates['theta_h'] = theta_h

        self.speeds = OrderedDict()
        self.speeds['omega_a'] = omega_a
        self.speeds['omega_h'] = omega_h

    def create_specified(self):

        self.specified = OrderedDict()
        self.specified['platform_position'] = dynamicsymbols('x')
        self.specified['platform_speed'] = dynamicsymbols('v')
        self.specified['platform_acceleration'] = dynamicsymbols('a')
        self.specified['ankle_torque'] = dynamicsymbols('T_a')
        self.specified['hip_torque'] = dynamicsymbols('T_h')

    def create_parameters(self):

        leg_length = symbols('l_L', **sym_kwargs)
        leg_com_length, torso_com_length = symbols('d_L, d_T', **sym_kwargs)
        leg_mass, torso_mass = symbols('m_L, m_T', **sym_kwargs)
        leg_inertia, torso_inertia = symbols('I_Lz, I_Tz', **sym_kwargs)
        g = symbols('g', **sym_kwargs)

        self.parameters = OrderedDict()
        self.parameters['leg_length'] = leg_length
        self.parameters['leg_com_length'] = leg_com_length
        self.parameters['torso_com_length'] = torso_com_length
        self.parameters['leg_mass'] = leg_mass
        self.parameters['torso_mass'] = torso_mass
        self.parameters['leg_inertia'] = leg_inertia
        self.parameters['torso_inertia'] = torso_inertia
        self.parameters['g'] = g

    def create_reference_frames(self):

        self.frames = OrderedDict()
        self.frames['inertial'] = ReferenceFrame('I')
        self.frames['leg'] = ReferenceFrame('L')
        self.frames['torso'] = ReferenceFrame('T')

    def orient_reference_frames(self):

        self.frames['leg'].orient(self.frames['inertial'],
                                  'Axis',
                                  (self.coordinates['theta_a'],
                                   self.frames['inertial'].z))

        self.frames['torso'].orient(self.frames['leg'],
                                    'Axis',
                                    (self.coordinates['theta_h'],
                                     self.frames['leg'].z))

    def create_points(self):

        self.points = OrderedDict()
        self.points['origin'] = Point('O')
        self.points['ankle'] = Point('A')
        self.points['hip'] = Point('H')
        self.points['leg_mass_center'] = Point('L_o')
        self.points['torso_mass_center'] = Point('T_o')

    def set_positions(self):

        vec = self.specified['platform_position'] * self.frames['inertial'].x
        self.points['ankle'].set_pos(self.points['origin'], vec)

        vec = self.parameters['leg_length'] * self.frames['leg'].y
        self.points['hip'].set_pos(self.points['ankle'], vec)

        vec = self.parameters['leg_com_length'] * self.frames['leg'].y
        self.points['leg_mass_center'].set_pos(self.points['ankle'], vec)

        vec = self.parameters['torso_com_length'] * self.frames['torso'].y
        self.points['torso_mass_center'].set_pos(self.points['hip'], vec)

    def define_kin_diff_eqs(self):

        time = dynamicsymbols._t

        self.kin_diff_eqs = (self.speeds['omega_a'] -
                             self.coordinates['theta_a'].diff(time),
                             self.speeds['omega_h'] -
                             self.coordinates['theta_h'].diff(time))

    def set_angular_velocities(self):

        vec = self.speeds['omega_a'] * self.frames['inertial'].z
        self.frames['leg'].set_ang_vel(self.frames['inertial'], vec)

        vec = self.speeds['omega_h'] * self.frames['leg'].z
        self.frames['torso'].set_ang_vel(self.frames['leg'], vec)

    def set_linear_velocities(self):

        self.points['origin'].set_vel(self.frames['inertial'], 0)

        self.points['ankle'].set_vel(self.frames['inertial'],
                                     self.specified['platform_speed'] *
                                     self.frames['inertial'].x)

        self.points['leg_mass_center'].v2pt_theory(self.points['ankle'],
                                                   self.frames['inertial'],
                                                   self.frames['leg'])

        self.points['hip'].v2pt_theory(self.points['ankle'],
                                       self.frames['inertial'],
                                       self.frames['leg'])

        self.points['torso_mass_center'].v2pt_theory(self.points['hip'],
                                                     self.frames['inertial'],
                                                     self.frames['torso'])

    def create_inertia_dyadics(self):

        leg_inertia_dyadic = inertia(self.frames['leg'], 0, 0,
                                     self.parameters['leg_inertia'])

        torso_inertia_dyadic = inertia(self.frames['torso'], 0, 0,
                                       self.parameters['torso_inertia'])

        self.central_inertias = OrderedDict()
        self.central_inertias['leg'] = (leg_inertia_dyadic,
                                        self.points['leg_mass_center'])
        self.central_inertias['torso'] = (torso_inertia_dyadic,
                                          self.points['torso_mass_center'])

    def create_rigid_bodies(self):

        self.rigid_bodies = OrderedDict()

        self.rigid_bodies['leg'] = RigidBody('Leg',
                                             self.points['leg_mass_center'],
                                             self.frames['leg'],
                                             self.parameters['leg_mass'],
                                             self.central_inertias['leg'])

        self.rigid_bodies['torso'] = RigidBody('Torso',
                                               self.points['torso_mass_center'],
                                               self.frames['torso'],
                                               self.parameters['torso_mass'],
                                               self.central_inertias['torso'])

    def create_loads(self):

        self.loads = OrderedDict()

        g = self.parameters['g']

        vec = -self.parameters['leg_mass'] * g * self.frames['inertial'].y
        self.loads['leg_force'] = (self.points['leg_mass_center'], vec)

        vec = -self.parameters['torso_mass'] * g * self.frames['inertial'].y
        self.loads['torso_force'] = (self.points['torso_mass_center'], vec)

        vec = (self.specified['ankle_torque'] * self.frames['inertial'].z -
               self.specified['hip_torque'] * self.frames['inertial'].z)
        self.loads['leg_torque'] = (self.frames['leg'], vec)

        self.loads['torso_torque'] = (self.frames['torso'],
                                      self.specified['hip_torque'] *
                                      self.frames['inertial'].z)

    def setup_problem(self):
        self.create_states()
        self.create_specified()
        self.create_parameters()
        self.create_reference_frames()
        self.orient_reference_frames()
        self.create_points()
        self.set_positions()
        self.define_kin_diff_eqs()
        self.set_angular_velocities()
        self.set_linear_velocities()
        self.create_inertia_dyadics()
        self.create_rigid_bodies()
        self.create_loads()

    def generate_eoms(self):

        self.kane = KanesMethod(self.frames['inertial'],
                                self.coordinates.values(),
                                self.speeds.values(),
                                self.kin_diff_eqs)

        fr, frstar = self.kane.kanes_equations(self.loads.values(),
                                               self.rigid_bodies.values())

        sub = {self.specified['platform_speed'].diff(dynamicsymbols._t):
               self.specified['platform_acceleration']}

        self.fr_plus_frstar = trigsimp(fr + frstar).subs(sub)

    def generate_rhs(self):

        udots = Matrix([s.diff() for s in self.speeds.values()])

        m1 = self.fr_plus_frstar.diff(udots[0])
        m2 = self.fr_plus_frstar.diff(udots[1])

        self.mass_matrix = m1.row_join(m2)
        self.forcing_vector = simplify(self.fr_plus_frstar -
                                       self.mass_matrix * udots)

        udot = self.mass_matrix.LUsolve(self.forcing_vector)
        qdot_map = self.kane.kindiffdict()
        qdot = Matrix([qdot_map[q.diff()] for q in self.coordinates.values()])
        self.rhs = qdot.col_join(udot)

    def derive(self):
        self.setup_problem()
        self.generate_eoms()
        self.generate_rhs()

    def create_symbolic_controller(self):

        states = self.coordinates.values() + self.speeds.values()
        inputs = self.specified.values()[-2:]

        num_states = len(states)
        num_inputs = len(inputs)

        xeq = Matrix([0 for x in states])

        K = Matrix(num_inputs, num_states, lambda i, j:
                   symbols('k_{}{}'.format(i, j)))
        self.gain_matrix = K

        x = Matrix(states)
        T = Matrix(inputs)

        self.gain_symbols = [k for k in K]

        # T = K * (xeq - x) -> 0 = T - K * (xeq - x)

        self.controller_dict = solve(T - K * (xeq - x), inputs)

    def generate_closed_loop_eoms(self):

        self.create_symbolic_controller()

        self.fr_plus_frstar_closed = msubs(self.fr_plus_frstar,
                                           self.controller_dict)
