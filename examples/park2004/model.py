#!/usr/bin/env python

from collections import OrderedDict

import numpy as np
from scipy.interpolate import interp1d
import sympy as sy
import sympy.physics.mechanics as me
import yeadon
from pydy.codegen.code import generate_ode_function

sym_kwargs = {'positive': True, 'real': True}
me.dynamicsymbols._t = sy.symbols('t', **sym_kwargs)


class PlanarStandingHumanOnMovingPlatform():
    """Generates the symbolic equations of motion of a 2D planar two body
    model representing a human standing on a antero-posteriorly moving
    platform similar to the one found in [Park2004]_.

    References
    ==========

    .. [Park2004] Park, Sukyung, Fay B. Horak, and Arthur D. Kuo. "Postural
       Feedback Responses Scale with Biomechanical Constraints in Human
       Standing." Experimental Brain Research 154, no. 4 (February 1, 2004):
       417-27. doi:10.1007/s00221-003-1674-3.

    Model Description
    =================

    Time Varying Parameters
    -----------------------
    theta_a :
        Angle of legs wrt to foot, plantar flexion is positive.
    theta_h :
        Angle of torso wrt to legs, extension is positive.
    omega_a :
        Angular rate of the legs wrt the foot.
    omega_h :
        Angular rate of the torso wrt the legs.
    a :
        Antero-posterior acceleration of the platform and foot, positive is
        forward.
    T_a :
        Torque applied between the foot and the legs, positive torque causes
        plantar flexion.
    T_h :
        Torque applied between the legs and the torso, positive torque
        causes extension.

    Constant Parameters
    -------------------

    l_L : distance from ankle to hip
    d_L : distance from ankle to legs center of mass
    d_T : distance from hip to torso center of mass
    m_L : mass of legs
    m_T : mass of torso
    I_L : moment of inertia of the legs
    I_T : moment of inertia of the torso
    g : acceleration due to gravity

    Equations of Motion
    -------------------

    The generalized coordinates:

    q = [theta_a, theta_h]

    q = [0, 0] is the upright standing configuration.

    The generalized speeds:

    u = [omega_a, omega_h]

    The states:

    x = [q, u]

    The specified inputs:

    r = [a, T_a, T_h]

    The first order explicit form of the equations of motion are:

    x' = f(x, r)

    The class can also output the first order implicit form:

    0 = f(x', x, r)

    Linearized Equations of Motion
    ------------------------------

    This class also generates a form linearized about the upright
    equilibrium where the only inputs are the joint torques.

    r = [T_a, T_h]

    x' = A * x + B * r

    Closed Loop Equations of Motion
    -------------------------------

    The open loop system can be controlled by a simple full state feedback
    controller.

    K = [k_00, k_01, k_02, k_03]
        [k_10, k_11, k_12, k_13]

    and

    S = [s_00, s_01, s_02, s_03]
        [s_10, s_11, s_12, s_13]

    x = [q, u]
    r = [a]
    T = [T_a, T_h]

    x' = f(x, r)

    where

    T = - S .* K * x

    where .* is the Hadamard product.

    S is a scaling factor for the gains that is necessary for convergence
    when the model is used in an NLP optimization.

    """

    def _create_states(self):

        self.time = me.dynamicsymbols._t

        syms = 'theta_a, theta_h, omega_a, omega_h'
        time_varying = [s(self.time) for s in
                        sy.symbols(syms, cls=sy.Function, real=True)]

        self.coordinates = OrderedDict()
        self.coordinates['ankle_angle'] = time_varying[0]
        self.coordinates['hip_angle'] = time_varying[1]

        self.speeds = OrderedDict()
        self.speeds['ankle_rate'] = time_varying[2]
        self.speeds['hip_rate'] = time_varying[3]

    def _create_specified(self):

        time_varying = [s(self.time)
                        for s in sy.symbols('x, v, a, T_h, T_a',
                                            cls=sy.Function, real=True)]

        self.specified = OrderedDict()
        self.specified['platform_position'] = time_varying[0]
        self.specified['platform_speed'] = time_varying[1]
        self.specified['platform_acceleration'] = time_varying[2]
        self.specified['ankle_torque'] = time_varying[3]
        self.specified['hip_torque'] = time_varying[4]

    def _create_parameters(self):

        self.parameters = OrderedDict()
        self.parameters['leg_length'] = sy.symbols('l_L', **sym_kwargs)
        self.parameters['leg_com_length'] = sy.symbols('d_L', **sym_kwargs)
        self.parameters['torso_com_length'] = sy.symbols('d_T', **sym_kwargs)
        self.parameters['leg_mass'] = sy.symbols('m_L', **sym_kwargs)
        self.parameters['torso_mass'] = sy.symbols('m_T', **sym_kwargs)
        self.parameters['leg_inertia'] = sy.symbols('I_L', **sym_kwargs)
        self.parameters['torso_inertia'] = sy.symbols('I_T', **sym_kwargs)
        self.parameters['g'] = sy.symbols('g', **sym_kwargs)

    def _create_reference_frames(self):

        self.frames = OrderedDict()
        self.frames['inertial'] = me.ReferenceFrame('I')
        self.frames['leg'] = me.ReferenceFrame('L')
        self.frames['torso'] = me.ReferenceFrame('T')

    def _orient_reference_frames(self):

        self.frames['leg'].orient(self.frames['inertial'],
                                  'Axis',
                                  (self.coordinates['ankle_angle'],
                                   self.frames['inertial'].z))

        self.frames['torso'].orient(self.frames['leg'],
                                    'Axis',
                                    (self.coordinates['hip_angle'],
                                     self.frames['leg'].z))

    def _create_points(self):

        self.points = OrderedDict()
        self.points['origin'] = me.Point('O')
        self.points['ankle'] = me.Point('A')
        self.points['hip'] = me.Point('H')
        self.points['leg_mass_center'] = me.Point('L_o')
        self.points['torso_mass_center'] = me.Point('T_o')

    def _set_positions(self):

        vec = self.specified['platform_position'] * self.frames['inertial'].x
        self.points['ankle'].set_pos(self.points['origin'], vec)

        vec = self.parameters['leg_length'] * self.frames['leg'].y
        self.points['hip'].set_pos(self.points['ankle'], vec)

        vec = self.parameters['leg_com_length'] * self.frames['leg'].y
        self.points['leg_mass_center'].set_pos(self.points['ankle'], vec)

        vec = self.parameters['torso_com_length'] * self.frames['torso'].y
        self.points['torso_mass_center'].set_pos(self.points['hip'], vec)

    def _define_kin_diff_eqs(self):

        self.kin_diff_eqs = (self.speeds['ankle_rate'] -
                             self.coordinates['ankle_angle'].diff(self.time),
                             self.speeds['hip_rate'] -
                             self.coordinates['hip_angle'].diff(self.time))

    def _set_angular_velocities(self):

        vec = self.speeds['ankle_rate'] * self.frames['inertial'].z
        self.frames['leg'].set_ang_vel(self.frames['inertial'], vec)

        vec = self.speeds['hip_rate'] * self.frames['leg'].z
        self.frames['torso'].set_ang_vel(self.frames['leg'], vec)

    def _set_linear_velocities(self):

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

    def _set_linear_accelerations(self):

        self.points['origin'].set_vel(self.frames['inertial'], 0)

        # Note : This doesn't acutally populate through in the KanesMethod
        # classe. See https://github.com/sympy/sympy/issues/8244.
        vec = (self.specified['platform_acceleration'] *
               self.frames['inertial'].x)
        self.points['ankle'].set_acc(self.frames['inertial'], vec)

    def _create_inertia_dyadics(self):

        leg_inertia_dyadic = me.inertia(self.frames['leg'], 0, 0,
                                        self.parameters['leg_inertia'])

        torso_inertia_dyadic = me.inertia(self.frames['torso'], 0, 0,
                                          self.parameters['torso_inertia'])

        self.central_inertias = OrderedDict()
        self.central_inertias['leg'] = (leg_inertia_dyadic,
                                        self.points['leg_mass_center'])
        self.central_inertias['torso'] = (torso_inertia_dyadic,
                                          self.points['torso_mass_center'])

    def _create_rigid_bodies(self):

        self.rigid_bodies = OrderedDict()

        self.rigid_bodies['leg'] = \
            me.RigidBody('Leg',
                         self.points['leg_mass_center'],
                         self.frames['leg'],
                         self.parameters['leg_mass'],
                         self.central_inertias['leg'])

        self.rigid_bodies['torso'] = \
            me.RigidBody('Torso',
                         self.points['torso_mass_center'],
                         self.frames['torso'],
                         self.parameters['torso_mass'],
                         self.central_inertias['torso'])

    def _create_loads(self):

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

    def _setup_problem(self):
        self._create_states()
        self._create_specified()
        self._create_parameters()
        self._create_reference_frames()
        self._orient_reference_frames()
        self._create_points()
        self._set_positions()
        self._define_kin_diff_eqs()
        self._set_angular_velocities()
        self._set_linear_velocities()
        self._set_linear_accelerations()
        self._create_inertia_dyadics()
        self._create_rigid_bodies()
        self._create_loads()

    def _generate_eoms(self):

        self.kane = me.KanesMethod(self.frames['inertial'],
                                   self.coordinates.values(),
                                   self.speeds.values(),
                                   self.kin_diff_eqs)

        fr, frstar = self.kane.kanes_equations(self.loads.values(),
                                               self.rigid_bodies.values())

        sub = {self.specified['platform_speed'].diff(self.time):
               self.specified['platform_acceleration']}

        self.fr_plus_frstar = sy.trigsimp(fr + frstar).subs(sub)

        udots = sy.Matrix([s.diff(self.time) for s in self.speeds.values()])

        m1 = self.fr_plus_frstar.diff(udots[0])
        m2 = self.fr_plus_frstar.diff(udots[1])

        # M x' = F
        self.mass_matrix = -m1.row_join(m2)
        self.forcing_vector = sy.simplify(self.fr_plus_frstar +
                                          self.mass_matrix * udots)

        M_top_rows = sy.eye(2).row_join(sy.zeros(2))
        F_top_rows = sy.Matrix(self.speeds.values())

        tmp = sy.zeros(2).row_join(self.mass_matrix)
        self.mass_matrix_full = M_top_rows.col_join(tmp)
        self.forcing_vector_full = F_top_rows.col_join(self.forcing_vector)

    def _generate_rhs(self):

        udot = self.mass_matrix.LUsolve(self.forcing_vector)
        qdot_map = self.kane.kindiffdict()
        qdot = sy.Matrix([qdot_map[q.diff(self.time)] for q in
                          self.coordinates.values()])
        self.rhs = qdot.col_join(udot)

    def _create_symbolic_controller(self):

        states = self.coordinates.values() + self.speeds.values()
        inputs = self.specified.values()[-2:]

        num_states = len(states)
        num_inputs = len(inputs)

        # The equilibrium point is the nominal upright configuration and
        # zero angular velocity.
        xeq = sy.Matrix([0 for x in states])

        K = sy.Matrix(num_inputs, num_states, lambda i, j:
                      sy.symbols('k_{}{}'.format(i, j)))
        self.gain_matrix = K

        S = sy.Matrix(num_inputs, num_states, lambda i, j:
                      sy.symbols('s_{}{}'.format(i, j)))
        self.scale_matrix = S

        x = sy.Matrix(states)
        T = sy.Matrix(inputs)

        self.gain_symbols = [k for k in K]
        self.scale_symbols = [s for s in S]

        # T = K * (xeq - x) -> 0 = T - S .* K * (xeq - x)

        self.controller_dict = sy.solve(T - S.multiply_elementwise(K) *
                                        (xeq - x), inputs)

    def _generate_closed_loop_eoms(self):

        self.fr_plus_frstar_closed = me.msubs(self.fr_plus_frstar,
                                              self.controller_dict)

    def _numerical_parameters(self):

        h = yeadon.Human('JasonYeadonMeas.txt')

        hip_pos = h.J1.pos
        ankle_pos = h.J2.solids[1].pos
        leg_length = np.sqrt(np.sum(np.asarray(ankle_pos - hip_pos)**2))

        leg_mass, leg_com, leg_inertia = \
            h.combine_inertia(['j0', 'j1', 'j2', 'j3', 'j4',
                               'k0', 'k1', 'k2', 'k3', 'k4'])

        leg_com_length = np.sqrt(np.sum(np.asarray(ankle_pos - leg_com)**2))

        torso_mass, torso_com, torso_inertia = \
            h.combine_inertia(['P', 'T', 'C', 'A1', 'A2', 'B1', 'B2'])

        p = {}
        p['leg_length'] = leg_length
        p['leg_com_length'] = leg_com_length
        p['leg_mass'] = leg_mass
        p['leg_inertia'] = leg_inertia[0, 0]
        p['torso_com_length'] = torso_com[2, 0]
        p['torso_mass'] = torso_mass
        p['torso_inertia'] = torso_inertia[0, 0]
        p['g'] = 9.81

        self.open_loop_par_map = OrderedDict()

        for k, v in self.parameters.items():
            self.open_loop_par_map[v] = p[k]

        # These are taken from Samin's paper. She may have gotten them from
        # the Park paper.
        self.numerical_gains = np.array([[950.0, 175.0, 185.0, 50.0],
                                         [45.0, 290.0, 60.0, 26.0]])
        # We are going to scale the gains so that the values we search for
        # with IPOPT are all close to 0.5 instead of the large gain values.
        self.numerical_gains_scales = self.numerical_gains / 0.5

        self.closed_loop_par_map = self.open_loop_par_map.copy()

        for k, v in zip(self.scale_symbols,
                        self.numerical_gains_scales.flatten()):
            self.closed_loop_par_map[k] = v

    def _linearize(self):

        # x = [theta_a, theta_h, omega_a, omega_h]
        states = self.coordinates.values() + self.speeds.values()
        # r = [T_a, T_h]
        specified = self.specified.values()[-2:]

        # We are only concerned about the upright standing equilibrium
        # point.
        equilibrium = {s: 0 for s in states}

        F_A = self.forcing_vector.jacobian(states).subs(equilibrium)
        F_B = self.forcing_vector.jacobian(specified).subs(equilibrium)

        A_top_rows = sy.zeros(2).row_join(sy.eye(2))
        B_top_rows = sy.zeros(2)

        M = self.mass_matrix.subs(equilibrium)

        A = A_top_rows.col_join(M.LUsolve(F_A))
        B = B_top_rows.col_join(M.LUsolve(F_B))

        self.A = sy.simplify(A)
        self.B = sy.simplify(B)

    def derive(self):
        self._setup_problem()
        self._generate_eoms()
        self._generate_rhs()
        self._create_symbolic_controller()
        self._generate_closed_loop_eoms()
        self._numerical_parameters()
        self._linearize()

    def numerical_linear(self):

        return (sy.matrix2numpy(self.A.subs(self.open_loop_par_map), dtype=float),
                sy.matrix2numpy(self.B.subs(self.open_loop_par_map), dtype=float))

    def closed_loop_ode_func(self, time, reference_noise,
                             platform_acceleration):
        """Returns a function that evaluates the continous closed loop
        system first order ODEs.

        Parameters
        ----------
        time : ndarray, shape(N,)
            The monotonically increasing time values.
        reference_noise : ndarray, shape(N, 4)
            The reference noise vector at each time.
        platform_acceleration : ndarray, shape(N,)
            The acceleration of the platform at each time.

        Returns
        -------
        rhs : function
            A function that evaluates the right hand side of the first order
            ODEs in a form easily used with scipy.integrate.odeint.
        args : dictionary
            A dictionary containing the model constant values and the
            controller function.

        """

        controls = np.empty(3, dtype=float)

        all_sigs = np.hstack((reference_noise,
                              np.expand_dims(platform_acceleration, 1)))
        interp_func = interp1d(time, all_sigs, axis=0)

        def controller(x, t):
            """
            x = [theta_a, theta_h, omega_a, omega_h]
            r = [a, T_a, T_h]
            """
            # TODO : This interpolation call is the most expensive thing
            # when running odeint.
            if t > time[-1]:
                result = interp_func(time[-1])
            else:
                result = interp_func(t)

            controls[0] = result[-1]
            controls[1:] = np.dot(0.5 * self.numerical_gains_scales,
                                  result[:-1] - x)

            return controls

        rhs = generate_ode_function(self.mass_matrix_full,
                                    self.forcing_vector_full,
                                    self.parameters.values(),
                                    self.coordinates.values(),
                                    self.speeds.values(),
                                    self.specified.values()[-3:],
                                    generator='cython')

        args = {'constants': np.array(self.open_loop_par_map.values()),
                'specified': controller}

        return rhs, args

    def first_order_implicit(self):
        return sy.Matrix(self.kin_diff_eqs).col_join(self.fr_plus_frstar_closed)

    def states(self):
        return self.coordinates.values() + self.speeds.values()
