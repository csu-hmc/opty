"""This example replicates some of the work presented in Ackermann and van
den Bogert 2010."""

import sympy as sm
import numpy as np
from pygait2d import derive, simulate
from pygait2d.segment import time_symbol
from opty import Problem, parse_free
from opty.utils import f_minus_ma

speed = 1.3250  # m/s
num_nodes = 60
h = sm.symbols('h', real=True, positive=True)
duration = (num_nodes - 1)*h

symbolics = derive.derive_equations_of_motion()

mass_matrix = symbolics[0]
forcing_vector = symbolics[1]
kane = symbolics[2]
constants = symbolics[3]
coordinates = symbolics[4]
speeds = symbolics[5]
states = coordinates + speeds
specified = symbolics[6]

num_states = len(states)

eom = f_minus_ma(mass_matrix, forcing_vector, coordinates + speeds)

# right: b, c, d
# left: e, f, g
qax, qay, qa, qb, qc, qd, qe, qf, qg = coordinates
uax, uay, ua, ub, uc, ud, ue, uf, ug = speeds
Fax, Fay, Ta, Tb, Tc, Td, Te, Tf, Tg = specified

par_map = simulate.load_constants(constants, 'example_constants.yml')

# Hand of god is nothing.
traj_map = {
    Fax: np.zeros(num_nodes),
    Fay: np.zeros(num_nodes),
    Ta: np.zeros(num_nodes),
}

bounds = {
    h: (0.001, 0.1),
    qax: (0.0, 10.0),
    qay: (0.5, 1.5),
    qa: (-np.pi/3.0, np.pi/3.0),  # +/- 60 deg
    uax: (0.0, 10.0),
    uay: (-10.0, 10.0),
}
bounds.update({k: (-np.pi/4.0, 3.0*np.pi/4.0) for k in [qb, qe]})  # hip
bounds.update({k: (-3.0/np.pi/4.0, 0.0) for k in [qc, qf]})  # knee
bounds.update({k: (-np.pi/4.0, np.pi/4.0) for k in [qd, qg]})  # foot
bounds.update({k: (-6.0, 6.0) for k in [ua, ub, uc, ud, ue, uf, ug]})  # ~200 deg/s
bounds.update({k: (-1000.0, 1000.0) for k in [Tb, Tc, Td, Te, Tf, Tg]})

# Specify the symbolic instance constraints, i.e. initial and end
# conditions.
uneval_states = [s.__class__ for s in states]
(qax, qay, qa, qb, qc, qd, qe, qf, qg, uax, uay, ua, ub, uc, ud, ue, uf, ug) = uneval_states

instance_constraints = (
    qax(0*h),
    qay(0*h) - 0.95,
    qb(0*h) - qe(duration),
    qc(0*h) - qf(duration),
    qd(0*h) - qg(duration),
    ub(0*h) - ue(duration),
    uc(0*h) - uf(duration),
    ud(0*h) - ug(duration),
    # TODO : need support for including h outside of a
    # Function argument.
    #qax(duration) - speed * duration,
    uax(0*h) - speed,
    uax(duration) - speed,
)


# Specify the objective function and it's gradient.
def obj(free):
    """Minimize the sum of the squares of the control torque."""
    T, h = free[num_states*num_nodes:-1], free[-1]
    return h*np.sum(T**2)


def obj_grad(free):
    T, h = free[num_states*num_nodes:-1], free[-1]
    grad = np.zeros_like(free)
    grad[num_states*num_nodes:-1] = 2.0*h*T
    grad[-1] = np.sum(T**2)
    return grad


# Create an optimization problem.
prob = Problem(obj, obj_grad, eom, states, num_nodes, h,
               known_parameter_map=par_map,
               known_trajectory_map=traj_map,
               instance_constraints=instance_constraints,
               bounds=bounds,
               time_symbol=time_symbol,
               tmp_dir='ufunc')

# Use a random positive initial guess.
initial_guess = prob.lower_bound + (prob.upper_bound - prob.lower_bound) * np.random.randn(prob.num_free)
initial_guess = np.zeros(prob.num_free)

# Find the optimal solution.
solution, info = prob.solve(initial_guess)


state_vals, _, _, h_val = parse_free(solution, num_states, len(specified) -
                                     len(traj_map), num_nodes,
                                     variable_duration=True)
np.savez('solution', x=state_vals, h=h_val, n=num_nodes)
