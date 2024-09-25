"""
Parameter Identification from Noncontiguous Measurements with Bias and added Noise
==================================================================================

In parameter estimation it is common to collect measurements of a system's
trajectories from distinct experiments. For example, if you are identifying the
parameters of a mass-spring-damper system, you may excite the system with
different initial conditions multiple times and measure the position of the
mass. The data cannot simply be stacked end-to-end in time and the
identification run because the measurement data would be discontinuous between
each measurement trial.

A workaround in opty is to create a set of differential equations with unique
state variables for each measurement trial that all share the same constant
parameters. You can then identify the parameters from all measurement trials
simultaneously by passing the uncoupled differential equations to opty.
This idea is due to Jason Moore.

In addition, for each measurement, one may create a set of differential
equations with unique state variables that again share the same constant
parameters. In addition, one adds noise to these equations. (Intuitively this
noise corresponds to a force acting on the system).
This idea is due to Huawei Wang and A. J. van den Bogert.

So, if one has 'number_of_measurements' measurements, and one creates
'number_of_repeats' differential equations for each measurement, one will have
'number_of_measurements' * 'number_of_repeats' uncoupled differential equations,
all sharing the same constant parameters.

Mass-spring-damper-friction Example
===================================

The position of a simple system consisting of a mass connected to a fixed point
by a spring and a damper and subject to friction is simulated and recorded as
noisy measurements with bias. The spring constant, the damping coefficient
and the coefficient of friction will be identified.

**State Variables**

- :math:`x_{i, j}`: position of the mass of the i-th measurement trial [m]
- :math:`u_{i, j}`: speed of the mass of the i-th measurement trial [m/s]

**Noise Variables**

- :math:`n_{i, j}`: noise added [N]

**Parameters**

- :math:`m`: mass [kg]
- :math:`c`: linear damping coefficient [Ns/m]
- :math:`k`: linear spring constant [N/m]
- :math:`l_0`: natural length of the spring [m]
- :math:`friction`: friction coefficient [N]

"""
import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
from scipy.integrate import solve_ivp
from opty import Problem
from opty.utils import parse_free
import matplotlib.pyplot as plt

# %%
# Equations of Motion.
# --------------------
#=========================================
# Basic data may be set here
number_of_measurements = 3
number_of_repeats = 5
t0, tf = 0.0, 10.0
num_nodes = 500
#=========================================

m, c, k, l0, friction = sm.symbols('m, c, k, l0, friction')

t = me.dynamicsymbols._t
x = sm.Matrix([me.dynamicsymbols([f'x{i}_{j}' for j in range(number_of_repeats)])
        for i in range(number_of_measurements)])
u = sm.Matrix([me.dynamicsymbols([f'u{i}_{j}' for j in range(number_of_repeats)])
        for i in range(number_of_measurements)])

n = sm.Matrix([me.dynamicsymbols([f'n{i}_{j}' for j in range(number_of_repeats)])
        for i in range(number_of_measurements)])

xh, uh = me.dynamicsymbols('xh, uh')

# %%
# Form the equations of motion, such that the first number_of_repeats equations
# belong to the first measurement, the second number_of_repeats equations belong
# to the second measurement, and so on.
def kd_eom_rh(xh, uh, m, c, k, l0, friction):
    """" sets up the eoms for the system"""
    N = me.ReferenceFrame('N')
    O, P = sm.symbols('O, P', cls=me.Point)
    O.set_vel(N, 0)

    P.set_pos(O, xh*N.x)
    P.set_vel(N, uh*N.x)
    bodies = [me.Particle('part', P, m)]
    # :math:`\tanh(\alpha \cdot x) \approx sign(x)` for large :math:`\alpha`, and
    # is differentiale everywhere.
    forces = [(P, -c*uh*N.x - k*(xh-l0)*N.x - friction * sm.tanh(30*uh)*N.x)]
    kd = sm.Matrix([uh - xh.diff(t)])
    KM = me.KanesMethod(N,
                        q_ind=[xh],
                        u_ind=[uh],
                        kd_eqs=kd
    )
    fr, frstar = KM.kanes_equations(bodies, forces)
    rhs = KM.rhs()
    return (kd, fr + frstar, rhs)

kd, fr_frstar, rhs = kd_eom_rh(xh, uh, m, c, k, l0, friction)

# %%
# Stack the equations appropriately.
kd_total = sm.Matrix([])
eom_total = sm.Matrix([])
rhs_total = sm.Matrix([])
for i in range(number_of_measurements):
    for j in range(number_of_repeats):
        eom_h = me.msubs(fr_frstar, {xh: x[i, j], uh: u[i, j], uh.diff(t): u[i, j].diff(t)})
        eom_h = eom_h + sm.Matrix([n[i, j]])
        kd_h = kd.subs({xh: x[i, j], uh: u[i, j]})
        kd_total = kd_total.col_join(kd_h)
        eom_total = eom_total.col_join(eom_h)
        rhs_h = sm.Matrix([rhs[1].subs({xh: x[i, j], uh: u[i, j]})])
        rhs_total = rhs_total.col_join(rhs_h)

eom = kd_total.col_join(eom_total)

uh =sm.Matrix([u[i, j] for i in range(number_of_measurements)
    for j in range(number_of_repeats)])
rhs = uh.col_join(rhs_total)

for i in range(number_of_repeats):
    sm.pprint(eom[i])
for i in range(3):
    print('.........')
for i in range(number_of_repeats):
    sm.pprint(eom[-(number_of_repeats-i)])

# %%
# Generate Noisy Measurement
# --------------------------
# Create 'number_of_measurements' sets of measurements with different initial
# conditions. To get the measurements, the equations of motion are integrated,
# and then noise is added to each point in time of the solution.
#

states = [x[i, j] for i in range(number_of_measurements)
                for j in range(number_of_repeats)] + \
        [u[i, j] for i in range(number_of_measurements)
                for j in range(number_of_repeats)]
parameters = [m, c, k, l0, friction]
par_vals = [1.0, 0.25, 2.0, 1.0, 0.5]

eval_rhs = sm.lambdify(states + parameters, rhs)

times = np.linspace(t0, tf, num=num_nodes)

measurements = []
# %%
# Integrate the differential equations. If np.random.seed(seed) is used, it
# the seed must be changed for every measurement to ensure they are independent.
for i in range(number_of_measurements):
    for j in range(number_of_repeats):
        seed = 1234*(i+1)*(j+1)
        np.random.seed(seed)
        x0 = 4*np.random.randn(2*number_of_measurements * number_of_repeats)
        sol = solve_ivp(lambda t, x, p: eval_rhs(*x, *p).squeeze(),
                    (t0, tf), x0, t_eval=times, args=(par_vals,))
    seed = 10000 + 12345*i
    np.random.seed(seed)
    measurements.append(sol.y[0, :] + 1.0*np.random.randn(num_nodes))
measurements = np.array(measurements)
print('shsape of measurement array', measurements.shape)
print('shape of solution array', sol.y.shape)

# %%
# Setup the Identification Problem
# --------------------------------
#
# The goal is to identify the damping coefficient :math:`c`, the spring
# constant :math:`k` and the coefficient of friction :math:`friction`.
# The objective :math:`J` is to minimize the least square
# difference in the optimal simulation as compared to the measurements.  If
# some measurement is considered more reliable, its weight :math:`w` may be
# increased relative to the other measurements.
#
#
# objective = :math:`\int_{t_0}^{t_f} \sum_{i=1}^{\text{number_of_measurements}} \left[ \sum_{s=1}^{\text{number_of_repeats}} (w_i (x_{i, s} - x_{i,s}^m)^2 \right] \hspace{2pt} dt`
#
interval_value = (tf - t0) / (num_nodes - 1)

w =[1 for _ in range(number_of_measurements)]

nm = number_of_measurements
nr = number_of_repeats
def obj(free):
    sum = 0.0
    for i in range(nm):
        for j in range(nr):
            sum += np.sum((w[i]*(free[(i*nr+j)*num_nodes:(i*nr+j+1)*num_nodes] -
                measurements[i])**2))
    return sum


def obj_grad(free):
    grad = np.zeros_like(free)
    for i in range(nm):
        for j in range(nr):
            grad[(i*nr+j)*num_nodes:(i*nr+j+1)*num_nodes] = 2*w[i]*interval_value*(
                free[(i*nr+j)*num_nodes:(i*nr+j+1)*num_nodes] - measurements[i]
            )
    return grad


# %%
# By not including :math:`c`, :math:`k`, :math:`friction`
# in the parameter map, they will be treated as unknown parameters.
par_map = {m: par_vals[0], l0: par_vals[3]}

bounds = {
    c: (0.01, 2.0),
    k: (0.1, 10.0),
    friction: (0.1, 10.0),
}
# %%
# Set up the known trajectory map. If np.random.seed(seed) is used, the
# seed must be changed for every map to ensure they are idependent.
# noise_scale gives the 'strength' of the noise.
noise_scale = 1.0
known_trajectory_map = {}
for i in range(number_of_measurements):
    for j in range(number_of_repeats):
        seed = 10000 + 12345*(i+1)*(j+1)
        np.random.seed(seed)
        known_trajectory_map = (known_trajectory_map |
            {n[i, j]: noise_scale*np.random.randn(num_nodes)})

# %%
# Set up the problem.
problem = Problem(
    obj,
    obj_grad,
    eom,
    states,
    num_nodes,
    interval_value,
    known_parameter_map=par_map,
    time_symbol=me.dynamicsymbols._t,
    integration_method='midpoint',
    bounds=bounds,
    known_trajectory_map=known_trajectory_map
)
# %%
# This give the sequence of the unknown parameters at the tail of solution.
print(problem.collocator.unknown_parameters)
# %%
# Create an Initial Guess
# -----------------------
#
# It is reasonable to use the measurements as initial guess for the states
# because they would be available. Here, only the measurements of the position
# are used and the speeds are set to zero. The last values are the guesses
# for :math:`c`, :math:`friction` and :math:`k`, respectively.
#
initial_guess = np.array([])
for i in range(number_of_measurements):
    for j in range(number_of_repeats):
        initial_guess = np.concatenate((initial_guess, measurements[i, :]))
initial_guess = np.hstack((initial_guess,
                np.zeros(number_of_measurements*number_of_repeats*num_nodes),
                [0.1, 3.0, 2.0]))

# %%
# Solve the Optimization Problem
# ------------------------------
#
solution, info = problem.solve(initial_guess)
print(info['status_msg'])
print(f'final value of the objective function is {info['obj_val']:.2f}' )

# %%
problem.plot_objective_value()

# %%
problem.plot_constraint_violations(solution)


# %%
# The identified parameters are:
# ------------------------------
#
print(f'Estimate of damping coefficient is      {solution[-3]: 1.2f}' +
      f' Percentage error is {(solution[-3]-par_vals[1])/solution[-3]*100:1.2f} %')
print(f'Estimate of the spring constant is      {solution[-1]: 1.2f}' +
      f' Percentage error is {(solution[-1]-par_vals[2])/solution[-1]*100:1.2f} %')
print(f'Estimate of the friction coefficient is {solution[-2]: 1.2f}'
      f' Percentage error is {(solution[-2]-par_vals[-1])/solution[-2]*100:1.2f} %')

# %%
# Plot the measurements and the trajectories calculated.
#-------------------------------------------------------
#
sol_parsed, _, _ = parse_free(solution, len(states), 0, num_nodes )
if number_of_measurements > 1:
    fig, ax = plt. subplots(number_of_measurements, 1,
        figsize=(6.4, 1.25*number_of_measurements), sharex=True, constrained_layout=True)

    for i in range(number_of_measurements):
        ax[i].plot(times, sol_parsed[i*number_of_repeats, :], color='red')
        ax[i].plot(times, measurements[i], color='blue', lw=0.5)
        ax[i].set_ylabel(f'{states[i]}')
    ax[-1].set_xlabel('Time [s]')
    ax[0].set_title('Trajectories')

else:
    fig, ax = plt. subplots(1, 1, figsize=(6.4, 1.25))
    ax.plot(times, sol_parsed[0, :], color='red')
    ax.plot(times, measurements[0], color='blue', lw=0.5)
    ax.set_ylabel(f'{states[0]}')
    ax.set_xlabel('Time [s]')
    ax.set_title('Trajectories')


prevent_output = True

# %%
#Plot the Trajectories
problem.plot_trajectories(solution)
# %%
#
# sphinx_gallery_thumbnail_number = 3
plt.show()
