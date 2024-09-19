"""
Parameter Identification from Non-Contiguous Measurements.
==========================================================

For parameter estimation it is common to collect measurements of a system's
trajectories for distinct experiments. for example, if you are identifying the
parameters of a mass-spring-damper system, you will exite the system with
different initial conditions multiple times. The date cannot simply be stacked
and the identification run because the measurement data would be discontinuous
between trials.
A work around in opty is to creat a set of differential equations with unique
state variables. For each measurement trial that all share the same constant
parameters. You can then identify the parameters from all measurement trials
simultaneously by passing the uncoupled differential equations to opty.

For exaple:
Four measurements of the location of a simple system consisting of a mass
connected to a fixed point by a spring and a damper are done. The movement
is in horizontal direction. The the spring constant and the damping coefficient
will be identified.


**State Variables**

- :math:`x_1`: position of the mass of the first measurement trial [m]
- :math:`x_2`: position of the mass of the second measurement trial [m]
- :math:`x_3`: position of the mass of the third measurement trial [m]
- :math:`x_4`: position of the mass of the fourth measurement trial [m]
- :math:`u_1`: speed of the mass of the first measurement trial [m/s]
- :math:`u_2`: speed of the mass of the second measurement trial [m/s]
- :math:`u_3`: speed of the mass of the third measurement trial [m/s]
- :math:`u_4`: speed of the mass of the fourth measurement trial [m/s]

**Parameters**

- :math:`m`: mass for both systems system [kg]
- :math:`c`: damping coefficient for both systems [Ns/m]
- :math:`k`: spring constant for both systems [N/m]
- :math:`l_0`: natural length of the spring [m]

"""
# %%
# Set up the equations of motion and integrate them to get the measurements.
#
import sympy as sm
import numpy as np
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from opty import Problem
from opty.utils import parse_free

x1, x2, x3, x4, u1, u2, u3, u4 = me.dynamicsymbols('x1, x2, x3, x4, u1, u2, u3, u4')
m, c, k, l0 = sm.symbols('m, c, k, l0')
t = me.dynamicsymbols._t

eom = sm.Matrix([
    x1.diff(t) - u1,
    x2.diff(t) - u2,
    x3.diff(t) - u3,
    x4.diff(t) - u4,
    m*u1.diff(t) + c*u1 + k*(x1 - l0),
    m*u2.diff(t) + c*u2 + k*(x2 - l0),
    m*u3.diff(t) + c*u3 + k*(x3 - l0),
    m*u4.diff(t) + c*u4 + k*(x4 - l0),
])
# %%
# Equations of motion.
sm.pprint(eom)

# %%
# Create the measurements for this example. To get the measurements, the
# equations of motion are integrated, and then noise is added to each point in
# time of the solution. Also a random shift is added to each measurement.
#
rhs = np.array([
    u1,
    u2,
    u3,
    u4,
    1/m * (-c*u1 - k*(x1 - l0)),
    1/m * (-c*u2 - k*(x2 - l0)),
    1/m * (-c*u3 - k*(x3 - l0)),
    1/m * (-c*u4 - k*(x4 - l0)),
])

qL = [x1, x2, x3, x4, u1, u2, u3, u4]
pL = [m, c, k, l0]

rhs_lam = sm.lambdify(qL + pL, rhs)

def gradient(t, x, args):
    return rhs_lam(*x, *args).reshape(8)

t0, tf = 0, 20
num_nodes = 500
times = np.linspace(t0, tf, num_nodes)
t_span = (t0, tf)

x0 = np.array([2, 3, 4, 5, 0, 0, 0, 0])
pL_vals = [1.0, 0.25, 1.0, 1.0]

resultat1 = solve_ivp(gradient, t_span, x0, t_eval = times, args=(pL_vals,))
resultat = resultat1.y.T

measurements = []
np.random.seed(123)
for i in range(4):
    measurements.append(resultat[:, i] + np.random.randn(resultat.shape[0]) * 0.5 +
        np.random.randn(1)*2)

# %%
# Set up the Identification Problem.
# ----------------------------------
#
# If some measurement is considered more reliable, its weight w may be increased.
#
# objective = :math:`\int_{t_0}^{t_f} (w_1 (x_1 - x_1^m)^2 + w_2 (x_2 - x_2^m)^2 + w_3 (x_3 - x_3^m)^2 + w_4 (x_4 - x_4^m)^2)\, dt`
#
state_symbols = [x1, x2, x3, x4, u1, u2, u3, u4]
unknown_parameters = [c, k]

interval_value = (tf - t0) / (num_nodes - 1)
par_map = {m: pL_vals[0], l0: pL_vals[3]}

w =[1, 1, 1, 1]
def obj(free):
    return interval_value *np.sum((w[0] * free[:num_nodes] - measurements[0])**2 +
            w[1] * (free[num_nodes:2*num_nodes] - measurements[1])**2 +
            w[2] * (free[2*num_nodes:3*num_nodes] - measurements[2])**2 +
            w[3] * (free[3*num_nodes:4*num_nodes] - measurements[3])**2
)


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[:num_nodes] = 2 * w[0] * interval_value * (free[:num_nodes] -
                            measurements[0])
    grad[num_nodes:2*num_nodes] = 2 * w[1] * (interval_value *
                    (free[num_nodes:2*num_nodes] - measurements[1]))
    grad[2*num_nodes:3*num_nodes] = 2 * w[2] * (interval_value *
                    (free[2*num_nodes:3*num_nodes] - measurements[2]))
    grad[3*num_nodes:4*num_nodes] = 2 * w[3] * (interval_value *
                    (free[3*num_nodes:4*num_nodes] - measurements[3]))
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

bounds = {
    c: (0, 2),
    k: (1, 3)
}

problem = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    num_nodes,
    interval_value,
    known_parameter_map=par_map,
    bounds=bounds,
    time_symbol=me.dynamicsymbols._t,
)

# %%
# Initial guess.
# It is reasonable to use the measurements as initial guess for the state.
#
initial_guess = np.array(list(measurements[0]) + list(measurements[1]) +
    list(measurements[2]) +list(measurements[3]) + list(np.zeros(4*num_nodes))
    + [0.1, 0.1])

# %%
# Solve the optimization problem.
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
print(f'Estimate of damping parameter is  {solution[-2]:.2f}')
print(f'Estimate ofthe spring constant is {solution[-1]:.2f}')

# %%
# Plot the Measurements and the Estimated Trajectories.
# -----------------------------------------------------
#
fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
for i in range(4):
    ax[i].plot(times, measurements[i], label=str(qL[i]))
    ax[i].set_ylabel(f'measurement {i+1}')
ax[0].set_title('Measurements')
ax[-1].set_xlabel('Time [sec]')
prevent_output = 0

# %%
problem.plot_trajectories(solution)
#
# sphinx_gallery_thumbnail_number = 3

plt.show()