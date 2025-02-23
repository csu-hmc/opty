from opty import Problem
from opty.utils import MathJaxRepr
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me

m, h, a, b, v, g, I1, I2, I3 = sm.symbols('m, h a, b, v, g, I1, I2, I3',
                                          real=True)
dt = sm.symbols('dt', real=True)

delta, deltadot, theta, thetadot, x, y, psi = me.dynamicsymbols(
    'delta, deltadot, theta, thetadot, x, y, psi')
t = me.dynamicsymbols._t

eom = sm.Matrix([
    theta.diff(t) - thetadot,
    (I1 + m*h**2)*thetadot.diff(t) +
    (I3 - I2 - m*h**2)*(v*sm.tan(delta)/b)**2*sm.sin(theta)*sm.cos(theta) -
    m*g*h*sm.sin(theta) +
    m*h*sm.cos(theta)*(a*v/b/sm.cos(delta)**2*delta.diff(t) +
                       v**2/v*sm.tan(delta)),
    x.diff(t) - v*sm.cos(psi),
    y.diff(t) - v*sm.sin(psi),
    psi.diff(t) - v/b*sm.tan(delta),
])

MathJaxRepr(eom)

# %%
# Set up the time discretization.
num_nodes = 201
duration = (num_nodes - 1)*dt

# %%
# Provide some reasonably realistic values for the constants.
par_map = {
    I1: 3.28,  # kg m^2
    # TODO : get some realistic values for I2 and I3
    I2: 1.0,  # kg m^2
    I3: 1.0,  # kg m^2
    a: 0.5,  # m
    b: 1.0,  # m
    g: 9.81,  # m/s^2
    h: 1.0,  # m
    m: 87.0,  # kg
    v: 5.0,  # m/s
}

state_symbols = (theta, thetadot, x, y, psi)
specified_symbols = (delta, )


# %%
# Specify the objective function and form the gradient.
def objective(free):
    """Return h (always the last element in the free variables)."""
    return free[-1]


def gradient(free):
    """Return the gradient of the objective."""
    grad = np.zeros_like(free)
    grad[-1] = 1.0
    return grad


# %%
# Specify the symbolic instance constraints, i.e. initial and end conditions.
# The car should be stationary at start and stop but laterally displaced 2
# meters (car width).
instance_constraints = (
    x.func(0*h),
    y.func(0*h),
    psi.func(0*h),
    # can't put an instance constraint on an input
    #delta.func(0*h),
    theta.func(0*h),
    thetadot.func(0*h),
    #deltadot.func(0*h),
    theta.func(duration),
    #delta.func(duration),
    psi.func(duration) - np.deg2rad(90.0),
    thetadot.func(duration),
    #deltadot.func(duration),
)

# %%
# Add some physical limits to some variables.
bounds = {
    delta: (np.deg2rad(-45.0), np.deg2rad(45.0)),
    #deltadot: (np.deg2rad(-200.0), np.deg2rad(200.0)),
}

# %%
# Create the optimization problem and set any options.
prob = Problem(objective, gradient, eom, state_symbols, num_nodes, dt,
               known_parameter_map=par_map,
               instance_constraints=instance_constraints, bounds=bounds,
               time_symbol=t)

# %%
# Give some rough estimates for the x and y trajectories.
#x_guess = 3.0/duration*2.0*time
#x_guess[num_nodes//2:] = 6.0 - 3.0/duration*2.0*time[num_nodes//2:]
#y_guess = 2.0/duration*time
#initial_guess = np.ones(prob.num_free)
#initial_guess[:num_nodes] = x_guess
#initial_guess[num_nodes:2*num_nodes] = y_guess
initial_guess = 1e-10*np.ones(prob.num_free)

# %%
# Find the optimal solution.
solution, info = prob.solve(initial_guess)

# %%
# Plot the optimal state and input trajectories.
_ = prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
_ = prob.plot_constraint_violations(solution)

# %%
# Plot the objective function as a function of optimizer iteration.
_ = prob.plot_objective_value()
