# %%
"""
Parameter Identification from Noncontiguous Measurements with Bias
==================================================================

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

If the measurements are biased in addition to being noisy, one way to handle
this is to double the set of differential equations created above, where the
second set is a shifted copy of the first set, shifted by twice the expected
bias.

Mass-spring-damper Example
==========================

The position of a simple system consisting of a mass connected to a fixed point
by a spring and a damper is simulated and recorded as noisy measurements with
bias. The spring constant and the damping coefficient will be identified.

**State Variables**

- :math:`x_i`: position of the mass of the i-th measurement trial [m]
- :math:`u_i`: speed of the mass of the i-th measurement trial [m/s]

**Parameters**

- :math:`m`: mass [kg]
- :math:`c`: linear damping coefficient [Ns/m]
- :math:`k`: linear spring constant [N/m]
- :math:`l_0`: natural length of the spring [m]

"""
import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
from scipy.integrate import solve_ivp
from opty import Problem
import matplotlib.pyplot as plt

# %%
# Equations of Motion.
# --------------------
#
# Set up as twice as many of equations of motion as there are measurements.
# The second part of these eoms are dependent on the first part by
# configuration constraints.
#
number_of_measurements = 6
t0, tf = 0.0, 10.0
num_nodes = 500

m, c, k, l0 = sm.symbols('m, c, k, l0')
bias_ident = sm.symbols(f'bias:{number_of_measurements}')

t = me.dynamicsymbols._t

x = me.dynamicsymbols(f'x:{2*number_of_measurements}')
u = me.dynamicsymbols(f'u:{2*number_of_measurements}')

N = me.ReferenceFrame('N')
O = me.Point('O')
O.set_vel(N, 0)

Points = sm.symbols(f'P:{2*number_of_measurements}', cls=me.Point)
for i in range(2*number_of_measurements):
    Points[i].set_pos(O, x[i]*N.x)
    Points[i].set_vel(N, u[i]*N.x)
bodies = [me.Particle('part_' + str(i), Points[i], m) for i in range(2*number_of_measurements)]
forces = [(Points[i], -c*u[i]*N.x - k*(x[i]-l0)*N.x) for i in range(2*number_of_measurements)]

q_ind = x[: number_of_measurements]
q_dep = x[number_of_measurements :]
u_ind = u[: number_of_measurements]
u_dep = u[number_of_measurements :]

kd = sm.Matrix([u[i] - x[i].diff(t) for i in range(2*number_of_measurements)])
config_constr = sm.Matrix([x[i] - x[number_of_measurements + i] + 2*bias_ident[i]
        for i in range(number_of_measurements)])
speed_constr = config_constr.diff(t)

KM = me.KanesMethod(
    N,
    q_ind=q_ind,
    q_dependent=q_dep,
    u_ind=u_ind,
    u_dependent=u_dep,
    kd_eqs=kd,
    velocity_constraints=speed_constr,
    configuration_constraints=config_constr,
)

fr, frstar = KM.kanes_equations(bodies, forces)
eom =kd.col_join(fr+frstar)
eom = eom.col_join(config_constr)

# %%
# Generate Noisy Measurement Data with Bias
# -----------------------------------------
# Create 'number_of_measurements' sets of measurements with different initial
# conditions. To get the measurements, the equations of motion are integrated,
# and then noise is added to each point in time of the solution. Finally a bias
# is added.
#
rhs = KM.rhs()
states = x + u
parameters = [m, c, k, l0] + list(bias_ident)
par_vals = [1.0, 0.5, 1.0, 1.0] + [0] * number_of_measurements

eval_rhs = sm.lambdify(states + parameters, rhs)

times = np.linspace(t0, tf, num=num_nodes)

measurements = []
np.random.seed(1234)
# %%
# As the second half of x, u are a shifted copy of the first half, they must get
# identical initial conditions.
for i in range(2*number_of_measurements):
    start1 = 4.0*np.random.randn(number_of_measurements)
    start2 = 4.0*np.random.randn(number_of_measurements)
    x0 = np.hstack((start1, start1, start2, start2))
    sol = solve_ivp(lambda t, x, p: eval_rhs(*x, *p).squeeze(),
                    (t0, tf), x0, t_eval=times, args=(par_vals,))
    if i < number_of_measurements:
        measurements.append(sol.y[0, :] +
                        1.0*np.random.randn(len(sol.t)))

# %%
# Add bias to the measurements
bias = 10.0 *np.random.uniform(0., 1., size=number_of_measurements)
measurements = np.array(measurements) + bias[:, np.newaxis]

print(measurements.shape)

# %%
# Setup the Identification Problem
# --------------------------------
#
# The goal is to identify the damping coefficient :math:`c` and the spring
# constant :math:`k`. The objective :math:`J` is to minimize the least square
# difference in the optimal simulation as compared to the measurements.  If
# some measurement is considered more reliable, its weight :math:`w` may be
# increased relative to the other measurements.
#
#
# objective = :math:`\int_{t_0}^{t_f} \left[ \sum_{i=1}^{i = \text{number_of_measurements}} (w_i (x_i - x_i^m)^2 \right] \hspace{2pt} dt`
#
interval_value = (tf - t0) / (num_nodes - 1)

w =[1 for _ in range(number_of_measurements)]

nm = number_of_measurements
def obj(free):
    return interval_value * np.sum([w[i] * np.sum(
        (free[(nm+i)*num_nodes:(nm+i+1)*num_nodes] - measurements[i])**2)
        for i in range(nm)]
)

def obj_grad(free):
    grad = np.zeros_like(free)
    for i in range(nm):
        grad[(nm+i)*num_nodes:(nm+i+1)*num_nodes] = 2*w[i]*interval_value*(
            free[(nm+i)*num_nodes:(nm+i+1)*num_nodes] - measurements[i]
)
    return grad


# %%
# By not including :math:`c`, :math:`k`, :math:`bias_i` in the parameter map,
# they will be treated as unknown parameters.
par_map = {m: par_vals[0], l0: par_vals[3]}

# %%
bounds = {
    c: (0.01, 2.0),
    k: (0.1, 10.0),
} | {u[i]: (-10, 10) for i in range(2*number_of_measurements)}

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
# for :math:`bias_i`, :math:`c` and :math:`k`, respectively.
#
initial_guess = np.hstack((np.random.randn(number_of_measurements*num_nodes), #x2
                           np.array(measurements).flatten(),  # x1
                           np.zeros(2*number_of_measurements*num_nodes),  # u
                           [0.5 for _ in range(number_of_measurements)], # bias
                           [0.1, 3.0],  # c, k
))


# %%
# Solve the Optimization Problem
# ------------------------------
#
initial_guess = np.load('non_contiguous_parameter_identification_bias_solution.npy')
solution, info = problem.solve(initial_guess)
# %%
# This is how the solution may be saved for a future initial guess
# ```np.save('non_contiguous_parameter_identification_bias_solution', solution)```
print(info['status_msg'])

# %%
problem.plot_objective_value()

# %%
problem.plot_constraint_violations(solution)


# %%
# The identified parameters are:
# ------------------------------
#
print(f'Estimate of damping coefficient is {solution[-2]: 1.2f}')
print(f'Estimate of the spring constant is {solution[-1]: 1.2f} \n')
for i in range(number_of_measurements):
    print(f'estimated {i}-th bias is {solution[-8+i]:.2f}, true bias is {bias[i]:.2f}')

# %%
# Plot the measurements and the trajectories calculated.
#
fig, ax = plt. subplots(number_of_measurements, 1,
    figsize=(6.4, 1.25*number_of_measurements), sharex=True, constrained_layout=True)
for i in range(number_of_measurements):
    ax[i].plot(times, solution[(nm+i)*num_nodes:(nm+i+1)*num_nodes], color='red')
    ax[i].plot(times, measurements[i], color='blue', lw=0.5)
    ax[i].set_ylabel(f'{states[nm+i]}')
ax[-1].set_xlabel('Time [s]')
ax[0].set_title('Trajectories')
prevent_output = True

# %%
#
# sphinx_gallery_thumbnail_number = 3
plt.show()
