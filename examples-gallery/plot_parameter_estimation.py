# %%
"""

Parameter Estimation
====================

Four noisy measurements of the location of a simple system consisting of a mass
connected to a fixed point by a spring and a damper. The movement is in a
horizontal direction. The the spring constant and the damping coefficient
are to be estimated.

The idea is to set up four sets of eoms, one for each of the measurements, with
identical parameters, and let opty estimate the parameters.

**State Variables**

- :math:`x_1`: position of the mass of the first system [m]
- :math:`x_2`: position of the mass of the second system [m]
- :math:`x_3`: position of the mass of the third system [m]
- :math:`x_4`: position of the mass of the fourth system [m]
- :math:`u_1`: speed of the mass of the first system [m/s]
- :math:`u_2`: speed of the mass of the second system [m/s]
- :math:`u_3`: speed of the mass of the third system [m/s]
- :math:`u_4`: speed of the mass of the fourth system [m/s]

**Parameters**

- :math:`m`: mass for both systems system [kg]
- :math:`c`: damping coefficient for both systems [Ns/m]
- :math:`k`: spring constant for both systems [N/m]
- :math:`l_0`: natural length of the spring [m]

"""
# %%
# Set up the equations of motion and integrate them to get the noisy measurements.
#
import sympy as sm
import numpy as np
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from opty import Problem
from opty.utils import parse_free

N = me.ReferenceFrame('N')
O, P1, P2, P3, P4 = sm.symbols('O P1 P2 P3 P4', cls=me.Point)

O.set_vel(N, 0)
t = me.dynamicsymbols._t

x1, x2, x3, x4 = me.dynamicsymbols('x1 x2 x3 x4')
u1, u2, u3, u4 = me.dynamicsymbols('u1 u2 h3 u4')
m, c, k, l0 = sm.symbols('m c k l0')

P1.set_pos(O, x1 * N.x)
P2.set_pos(O, x2 * N.x)
P3.set_pos(O, x3 * N.x)
P4.set_pos(O, x4 * N.x)
P1.set_vel(N, u1 * N.x)
P2.set_vel(N, u2 * N.x)
P3.set_vel(N, u3 * N.x)
P4.set_vel(N, u4 * N.x)

body1 = me.Particle('body1', P1, m)
body2 = me.Particle('body2', P2, m)
body3 = me.Particle('body3', P3, m)
body4 = me.Particle('body4', P4, m)
bodies = [body1, body2, body3, body4]

forces = [(P1, -k * (x1 - l0) * N.x - c * u1 * N.x),
    (P2, -k * (x2 - l0) * N.x - c * u2 * N.x), (P3, -k * (x3 - l0) * N.x - c * u3 * N.x),
    (P4, -k * (x4 - l0) * N.x - c * u4 * N.x)]

kd = sm.Matrix([u1 - x1.diff(), u2 - x2.diff(), u3 - x3.diff(), u4 - x4.diff()])

q_ind = [x1, x2, x3, x4]
u_ind = [u1, u2, u3, u4]

KM = me.KanesMethod(N, q_ind, u_ind, kd_eqs=kd)
fr, frstar = KM.kanes_equations(bodies, forces)
eom = kd.col_join(fr + frstar)
sm.pprint(eom)

rhs = KM.rhs()

qL = q_ind + u_ind
pL = [m, c, k, l0]

rhs_lam = sm.lambdify(qL + pL, rhs)


def gradient(t, x, args):
    return rhs_lam(*x, *args).reshape(8)

t0, tf = 0, 20
num_nodes = 500
times = np.linspace(t0, tf, num_nodes)
t_span = (t0, tf)

x0 = np.array([3, 3, 3, 3, 0, 0, 0, 0])
pL_vals = [1.0, 0.25, 1.0, 1.0]

resultat1 = solve_ivp(gradient, t_span, x0, t_eval = times, args=(pL_vals,))
resultat = resultat1.y.T

noisy = []
np.random.seed(123)
for i in range(4):
    noisy.append(resultat[:, i] + np.random.randn(resultat.shape[0]) * 0.5 +
        np.random.randn(1)*2)

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(4):
    ax.plot(times, noisy[i], label=f'measurement {i+1}', lw=0.5)
plt.xlabel('Time')
ax.set_title('Measurements of the position of the mass')
ax.legend();

plt.show()


# %%
# Set up the Estimation Problem.
# --------------------------------
#
# If some measurement is considered more reliable, its weight w can be increased.
#
# objective = :math:`\int_{t_0}^{t_f} (w_1 (x_1 - noisy_{x_1})^2 + w_2 (x_2 - noisy_{x_2})^2 + w_3 (x_3 - noisy_{x_3})^2 + w_4 (x_4 - noisy_{x_4})^2)\, dt`
#
state_symbols = [x1, x2, x3, x4, u1, u2, u3, u4]
unknown_parameters = [c, k]

interval_value = (tf - t0) / (num_nodes - 1)
par_map = {m: pL_vals[0], l0: pL_vals[3]}

w =[1, 1, 1, 1]
def obj(free):
    return interval_value *np.sum((w[0] * free[:num_nodes] - noisy[0])**2 +
            w[1] * (free[num_nodes:2*num_nodes] - noisy[1])**2 +
            w[2] * (free[2*num_nodes:3*num_nodes] - noisy[2])**2 +
            w[3] * (free[3*num_nodes:4*num_nodes] - noisy[3])**2
)


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[:num_nodes] = 2 * w[0] * interval_value * (free[:num_nodes] - noisy[0])
    grad[num_nodes:2*num_nodes] = 2 * w[1] * (interval_value *
                    (free[num_nodes:2*num_nodes] - noisy[1]))
    grad[2*num_nodes:3*num_nodes] = 2 * w[2] * (interval_value *
                    (free[2*num_nodes:3*num_nodes] - noisy[2]))
    grad[3*num_nodes:4*num_nodes] = 2 * w[3] * (interval_value *
                    (free[3*num_nodes:4*num_nodes] - noisy[3]))
    return grad

instance_constraints = (
    x1.subs({t: t0}) - x0[0],
    x2.subs({t: t0}) - x0[1],
    x3.subs({t: t0}) - x0[2],
    x4.subs({t: t0}) - x0[3],
    u1.subs({t: t0}) - x0[4],
    u2.subs({t: t0}) - x0[5],
    u3.subs({t: t0}) - x0[6],
    u4.subs({t: t0}) - x0[7],
)

problem = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    num_nodes,
    interval_value,
    known_parameter_map=par_map,
    instance_constraints=instance_constraints,
    time_symbol=me.dynamicsymbols._t,
)

# %%
# Initial guess.
#
initial_guess = np.array(list(np.random.randn(8*num_nodes + 2)))
# %%
# Solve the Optimization Problem.
#
solution, info = problem.solve(initial_guess)
print(info['status_msg'])
problem.plot_objective_value()
# %%
problem.plot_constraint_violations(solution)
# %%
# Results obtained.
#------------------
#
state_sol, input_sol, _ = parse_free(solution, len(state_symbols),
    0, num_nodes)
state_sol = state_sol.T
error_x1 = (state_sol[:, 0] - resultat[:, 0]) / np.max(resultat[:, 0]) * 100
error_x2 = (state_sol[:, 1] - resultat[:, 1]) / np.max(resultat[:, 1]) * 100
error_x3 = (state_sol[:, 2] - resultat[:, 2]) / np.max(resultat[:, 2]) * 100
error_x4 = (state_sol[:, 3] - resultat[:, 3]) / np.max(resultat[:, 3]) * 100
error_max = max(np.max(error_x1), np.max(error_x2), np.max(error_x3),
    np.max(error_x4))
print(f'Estimate of damping parameter is  {solution[-2]:.2f} %')
print(f'Estimate ofthe spring constant is {solution[-1]:.2f} %')