#!/usr/bin/env python

import sympy as sym
import sympy.physics.mechanics as me
from opty.utils import state_derivatives


def n_link_pendulum_on_cart(n, cart_force=True, joint_torques=False,
                            spring_damper=False):
    """Returns the the symbolic first order equations of motion for a 2D
    n-link pendulum on a sliding cart under the influence of gravity in this
    form:

        M(x) x(t) = F(x, u, t)

    Parameters
    ----------
    n : integer
        The number of links in the pendulum.
    cart_force : boolean, default=True
        If true an external specified lateral force is applied to the cart.
    joint_torques : boolean, default=False
        If true joint torques will be added as specified inputs at each
        joint.
    spring_damper : boolean, default=False
        If true a linear spring and damper are added to constrain the cart
        to the origin.

    Returns
    -------
    mass_matrix : sympy.MutableMatrix, shape(2 * (n + 1), 2 * (n + 1))
        The symbolic mass matrix of the system which are linear in u' and q'.
    forcing_vector : sympy.MutableMatrix, shape(2 * (n + 1), 1)
        The forcing vector of the system.
    constants : list
        A sequence of all the symbols which are constants in the equations
        of motion.
    coordinates : list
        A sequence of all the dynamic symbols, i.e. functions of time, which
        describe the configuration of the system.
    speeds : list
        A sequence of all the dynamic symbols, i.e. functions of time, which
        describe the generalized speeds of the system.
    specfied : list
        A sequence of all the dynamic symbols, i.e. functions of time, which
        describe the specified inputs to the system.

    Notes
    -----
    The degrees of freedom of the system are n + 1, i.e. one for each
    pendulum link and one for the lateral motion of the cart.

    M x' = F, where x = [u0, ..., un+1, q0, ..., qn+1]

    The joint angles are all defined relative to the ground where the x axis
    defines the ground line and the y axis points up. The joint torques are
    applied between each adjacent link and the between the cart and the
    lower link where a positive torque corresponds to positive angle.

    """
    if n <= 0:
        raise ValueError('The number of links must be a positive integer.')

    q = me.dynamicsymbols('q:{}'.format(n + 1))
    u = me.dynamicsymbols('u:{}'.format(n + 1))

    if joint_torques is True:
        T = me.dynamicsymbols('T1:{}'.format(n + 1))

    m = sym.symbols('m:{}'.format(n + 1))
    l = sym.symbols('l:{}'.format(n))
    g, t = sym.symbols('g t')

    I = me.ReferenceFrame('I')
    O = me.Point('O')
    O.set_vel(I, 0)

    P0 = me.Point('P0')
    P0.set_pos(O, q[0] * I.x)
    P0.set_vel(I, u[0] * I.x)
    Pa0 = me.Particle('Pa0', P0, m[0])

    frames = [I]
    points = [P0]
    particles = [Pa0]
    if spring_damper:
        k, c = sym.symbols('k, c')
        forces = [(P0, -m[0] * g * I.y - k * q[0] * I.x - c * u[0] * I.x)]
    else:
        forces = [(P0, -m[0] * g * I.y)]
    kindiffs = [q[0].diff(t) - u[0]]

    if cart_force is True or joint_torques is True:
        specified = []
    else:
        specified = None

    for i in range(n):
        Bi = I.orientnew('B{}'.format(i), 'Axis', [q[i + 1], I.z])
        Bi.set_ang_vel(I, u[i + 1] * I.z)
        frames.append(Bi)

        Pi = points[-1].locatenew('P{}'.format(i + 1), l[i] * Bi.y)
        Pi.v2pt_theory(points[-1], I, Bi)
        points.append(Pi)

        Pai = me.Particle('Pa' + str(i + 1), Pi, m[i + 1])
        particles.append(Pai)

        forces.append((Pi, -m[i + 1] * g * I.y))

        if joint_torques is True:

            specified.append(T[i])

            if i == 0:
                forces.append((I, -T[i] * I.z))

            if i == n - 1:
                forces.append((Bi, T[i] * I.z))
            else:
                forces.append((Bi, T[i] * I.z - T[i + 1] * I.z))

        kindiffs.append(q[i + 1].diff(t) - u[i + 1])

    if cart_force is True:
        F = me.dynamicsymbols('F')
        forces.append((P0, F * I.x))
        specified.append(F)

    kane = me.KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kindiffs)
    kane.kanes_equations(particles, forces)

    mass_matrix = kane.mass_matrix_full
    forcing_vector = kane.forcing_full
    coordinates = [x for x in kane._q]
    speeds = [x for x in kane._u]

    if spring_damper:
        constants = [k, c, g, m[0]]
    else:
        constants = [g, m[0]]
    for i in range(n):
        constants += [l[i], m[i + 1]]

    return (mass_matrix, forcing_vector, constants, coordinates, speeds,
            specified)


def create_symbolic_controller(states, inputs):
    """"Returns a dictionary with keys that are the joint torque inputs and
    the values are the controller expressions. This can be used to convert
    the symbolic equations of motion from 0 = f(x', x, u, t) to a closed
    loop form 0 = f(x', x, t).

    Parameters
    ----------
    states : sequence of len 2 * (n + 1)
        The SymPy time dependent functions for the system states where n are
        the number of links.
    inputs : sequence of len n
        The SymPy time depednent functions for the system joint torque
        inputs (should not include the lateral force).

    Returns
    -------
    controller_dict : dictionary
        Maps joint torques to control expressions.
    gain_symbols : list of SymPy Symbols
        The symbols used in the gain matrix.
    xeq : list of SymPy Symbols
        The symbols for the equilibrium point.

    """
    num_states = len(states)
    num_inputs = len(inputs)

    xeq = sym.Matrix([x.__class__.__name__ + '_eq' for x in states])

    K = sym.Matrix(num_inputs, num_states, lambda i, j:
                   sym.Symbol('k_{}{}'.format(i, j)))

    x = sym.Matrix(states)
    T = sym.Matrix(inputs)

    gain_symbols = [k for k in K]

    # T = K * (xeq - x) -> 0 = T - K * (xeq - x)

    controller_dict = sym.solve(T - K * (xeq - x), inputs)

    return controller_dict, gain_symbols, xeq


def symbolic_constraints(mass_matrix, forcing_vector, states,
                         controller_dict, equilibrium_dict=None):
    """Returns a vector expression of the zero valued closed loop system
    equations of motion: M * x' - F.

    Parameters
    ----------
    mass_matrix : sympy.Matrix, shape(n, n)
        The system mass matrix, M.
    forcing_vector : sympy.Matrix, shape(n, 1)
        The system forcing vector, F.
    states : iterable of sympy.Function, len(n)
        The functions of time representing the states.
    controll_dict : dictionary
        Maps any input forces in the forcing vector to the symbolic
        controller expressions.
    equilibrium_dit : dictionary
        A dictionary of equilibrium values to substitute.

    Returns
    -------
    constraints : sympy.Matrix, shape(n, 1)
        The closed loop constraint expressions.

    """

    xdot = sym.Matrix(state_derivatives(states))

    if equilibrium_dict is not None:
        for k, v in controller_dict.items():
            controller_dict[k] = v.subs(equilibrium_dict)

    # M * x' = F -> M * x' - F = 0
    system = mass_matrix * xdot - forcing_vector.subs(controller_dict)

    return system


def symbolic_constraints_solved(mass_matrix, forcing_vector, states,
                                controller_dict, equilibrium_dict=None):
    """Returns a vector expression of the zero valued closed loop system
    equations of motion: x' - M^-1 * F.

    Parameters
    ----------
    mass_matrix : sympy.Matrix, shape(n, n)
        The system mass matrix, M.
    forcing_vector : sympy.Matrix, shape(n, 1)
        The system forcing vector, F.
    states : iterable of sympy.Function, len(n)
        The functions of time representing the states.
    controll_dict : dictionary
        Maps any input forces in the forcing vector to the symbolic
        controller expressions.
    equilibrium_dit : dictionary
        A dictionary of equilibrium values to substitute.

    Returns
    -------
    constraints : sympy.Matrix, shape(n, 1)
        The closed loop constraint expressions.

    Notes
    -----
    The mass matrix is symbolically inverted, so this can be potentailly be
    slow for large systems.

    """

    xdot = sym.Matrix(state_derivatives(states))

    if equilibrium_dict is not None:
        for k, v in controller_dict.items():
            controller_dict[k] = v.subs(equilibrium_dict)

    F = forcing_vector.subs(controller_dict)
    constraints = xdot - mass_matrix.LUsolve(F)

    return constraints
