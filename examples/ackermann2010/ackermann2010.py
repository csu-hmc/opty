"""This example replicates some of the work presented in Ackermann and van
den Bogert 2010."""

import numpy as np
from pygait2d import derive, simulate
from pygait2d.segment import time_symbol
from opty.direct_collocation import Problem

from model import f_minus_ma

speed = 1.3250
speed = 0.0
duration = 0.5
num_nodes = 60

interval_value = duration / (num_nodes - 1)

symbolics = derive.derive_equations_of_motion()

mass_matrix = symbolics[0]
forcing_vector = symbolics[1]
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
traj_map = {Fax: np.zeros(num_nodes),
            Fay: np.zeros(num_nodes),
            Ta: np.zeros(num_nodes)}

bounds = {qax: (-3.0, 3.0), qay: (0.5, 1.5), qa: (-np.pi / 3.0, np.pi / 3.0)}
bounds.update({k: (-np.pi, np.pi) for k in [qb, qc, qd, qe, qf, qg]})
bounds.update({k: (-10.0, 10.0) for k in [uax, uay]})
bounds.update({k: (-20.0, 20.0) for k in [ua, ub, uc, ud, ue, uf, ug]})
bounds.update({k: (-300.0, 300.0) for k in [Tb, Tc, Td, Te, Tf, Tg]})

# Specify the symbolic instance constraints, i.e. initial and end
# conditions.
uneval_states = [s.__class__ for s in states]
(qax, qay, qa, qb, qc, qd, qe, qf, qg, uax, uay, ua, ub, uc, ud, ue, uf, ug) = uneval_states

instance_constraints = (qb(0.0) - qe(duration),
                        qc(0.0) - qf(duration),
                        qd(0.0) - qg(duration),
                        ub(0.0) - ue(duration),
                        uc(0.0) - uf(duration),
                        ud(0.0) - ug(duration),
                        qax(duration) - speed * duration,
                        qax(0.0))


# Specify the objective function and it's gradient.
def obj(free):
    """Minimize the sum of the squares of the control torque."""
    T = free[num_states * num_nodes:]
    return np.sum(T**2)


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[num_states * num_nodes:] = 2.0 * interval_value * free[num_states *
                                                                num_nodes:]
    return grad

# Create an optimization problem.
prob = Problem(obj, obj_grad, eom, states, num_nodes, interval_value,
               known_parameter_map=par_map,
               known_trajectory_map=traj_map,
               instance_constraints=instance_constraints,
               bounds=bounds,
               time_symbol=time_symbol,
               tmp_dir='ufunc')

# Use a random positive initial guess.
initial_guess = prob.lower_bound + (prob.upper_bound - prob.lower_bound) * np.random.randn(prob.num_free)

# Find the optimal solution.
solution, info = prob.solve(initial_guess)
