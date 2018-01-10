#!/usr/bin/env python

"""This script demonstrates an attempt at identifying the controller for a n
link inverted pendulum on a cart by direct collocation. I collect "measured"
data from the system by simulating it with a known optimal controller under
the influence of random lateral force perturbations. I then form the
optimization problem such that we minimize the error in the model's
simulated outputs with respect to the measured outputs. The optimizer
searches for the best set of controller gains (which are unknown) that
reproduce the motion and ensure the dynamics are valid.

Dependencies this runs with:

    numpy 1.8+
    scipy 0.14+
    sympy 1.0+
    matplotlib 1.3+
    pydy 0.3+
    cyipopt 0.1.4

N : number of discretization points
M : number of measured time samples

n : number of states
o : number of model outputs
p : total number of model constants
q : number of free model constants
r : number of free specified inputs

"""

# standard lib
import os
import sys
import datetime
import hashlib
import time

# external
import numpy as np
from scipy.integrate import odeint
from opty import direct_collocation as dc
from opty import parameter_identification as pi
from opty import utils

# local
import data
import model
import simulate
import visualization as viz


class Identifier():

    def __init__(self, num_links, duration, sample_rate, init_type,
                 sensor_noise, do_plot, do_animate):
        """

        Parameters
        ==========
        num_links : integer
            The desired number of links in the pendulum.
        duration : float
            The desired duration of the simulation.
        sample_rate : float
            The desired sample rate for both the measurements and the
            collocation discretization in Hertz.
        init_type : string
            The type of initial guess to provide. All of the options provide
            the "measured" state trajectories as initial guesses. The
            guesses for the gains are given by these options: 'known'
            provides the known gains, 'zero' provides zero, 'ones' provides
            ones, 'close' gives random guesses that are close to the known
            gains, and 'random' give values between -100 and 100.
        sensor_noise : boolean
            If true, noise will be added to the sensor measurements.
        do_plot : boolean
            If true, plots will display showing the identification results.
        do_animate : boolean
            If true, an animation of the pendulum will be displayed.

        """

        self.num_links = num_links
        self.duration = duration
        self.sample_rate = sample_rate
        self.init_type = init_type
        self.sensor_noise = sensor_noise
        self.do_plot = do_plot
        self.do_animate = do_animate

    def compute_discretization(self):

        self.num_time_steps = int(self.duration * self.sample_rate) + 1
        self.discretization_interval = 1.0 / self.sample_rate
        self.time = np.linspace(0.0, self.duration, num=self.num_time_steps)

    def generate_eoms(self):
        # Generate the symbolic equations of motion for the two link pendulum on
        # a cart.
        print("Generating equations of motion.")
        self.system = model.n_link_pendulum_on_cart(self.num_links,
                                                    cart_force=True,
                                                    joint_torques=True,
                                                    spring_damper=True)

        self.mass_matrix = self.system[0]
        self.forcing_vector = self.system[1]
        self.constants_syms = self.system[2]
        self.coordinates_syms = self.system[3]
        self.speeds_syms = self.system[4]
        # last entry is always the lateral force
        self.specified_inputs_syms = self.system[5]

        self.states_syms = self.coordinates_syms + self.speeds_syms

        self.num_states = len(self.states_syms)

    def find_optimal_gains(self):
        # Find some optimal gains for stablizing the pendulum on the cart.
        print('Finding the optimal gains.')
        self.gains = simulate.compute_controller_gains(self.num_links)

    def simulate(self):
        # Generate some "measured" data from the simulation.
        print('Simulating the system.')

        self.lateral_force = simulate.input_force('sumsines', self.time)

        set_point = np.zeros(self.num_states)

        self.initial_conditions = np.zeros(self.num_states)
        offset = 10.0 * np.random.random((self.num_states // 2) - 1)
        self.initial_conditions[1:self.num_states // 2] = np.deg2rad(offset)

        rhs, args = simulate.closed_loop_ode_func(self.system, self.time,
                                                  set_point, self.gains,
                                                  self.lateral_force)

        start = time.clock()
        self.x = odeint(rhs, self.initial_conditions, self.time, args=args)
        msg = 'Simulation of {} real time seconds took {} CPU seconds to compute.'
        print(msg.format(self.duration, time.clock() - start))

        self.x_noise = self.x + np.deg2rad(0.25) * np.random.randn(*self.x.shape)
        self.y = pi.output_equations(self.x)
        self.y_noise = pi.output_equations(self.x_noise)
        self.u = self.lateral_force

    def generate_symbolic_closed_loop(self):
        print('Forming the closed loop equations of motion.')

        # Generate the expressions for creating the closed loop equations of
        # motion.
        control_dict, gain_syms, equil_syms = \
            model.create_symbolic_controller(self.states_syms,
                                             self.specified_inputs_syms[:-1])

        self.num_gains = len(gain_syms)

        eq_dict = dict(zip(equil_syms, self.num_states * [0]))

        # This is the symbolic closed loop continuous system.
        self.closed = model.symbolic_constraints(self.mass_matrix,
                                                 self.forcing_vector,
                                                 self.states_syms,
                                                 control_dict, eq_dict)

    def generate_objective_funcs(self):
        print('Forming the objective function.')

        self.obj_func = pi.wrap_objective(pi.objective_function,
                                          self.num_time_steps,
                                          self.num_states,
                                          self.discretization_interval,
                                          self.time,
                                          self.y_noise if self.sensor_noise else self.y)

        self.obj_grad_func = pi.wrap_objective(pi.objective_function_gradient,
                                               self.num_time_steps,
                                               self.num_states,
                                               self.discretization_interval,
                                               self.time,
                                               self.y_noise if self.sensor_noise else self.y)

    def generate_constraint_funcs(self):

        print('Forming the constraint functions.')

        self.prob = dc.Problem(self.obj_func,
                               self.obj_grad_func,
                               self.closed,
                               self.states_syms,
                               self.num_time_steps,
                               self.discretization_interval,
                               known_parameter_map=simulate.constants_dict(self.constants_syms),
                               known_trajectory_map={self.specified_inputs_syms[-1]: self.u})

        self.output_filename = 'ipopt_output.txt'
        if sys.version_info >= (3, 0):
            self.prob.addOption(b'output_file', self.output_filename.encode())
            self.prob.addOption(b'print_timing_statistics', b'yes')
            # TODO : Not available in general.
            #self.prob.addOption(b'linear_solver', b'ma57')
        else:
            self.prob.addOption('output_file', self.output_filename)
            self.prob.addOption('print_timing_statistics', 'yes')
            # TODO : Not available in general.
            #self.prob.addOption('linear_solver', 'ma57')

    def optimize(self):

        print('Solving optimization problem.')

        init_states, init_specified, init_constants = \
            utils.parse_free(self.initial_guess, self.num_states, 0,
                             self.num_time_steps)
        init_gains = init_constants.reshape(self.gains.shape)

        self.solution, info = self.prob.solve(self.initial_guess)

        self.sol_states, sol_specified, sol_constants = \
            utils.parse_free(self.solution, self.num_states, 0,
                             self.num_time_steps)
        sol_gains = sol_constants.reshape(self.gains.shape)

        print("\nInitial gain guess: {}".format(init_gains))
        print("Known gains: {}".format(self.gains))
        print("Identified gains: {}".format(sol_gains))

    def plot(self):

        viz.plot_sim_results(self.y_noise if self.sensor_noise else self.y,
                             self.u)
        viz.plot_constraints(self.prob.con(self.initial_guess),
                             self.num_states,
                             self.num_time_steps,
                             self.states_syms)
        viz.plot_constraints(self.prob.con(self.solution),
                             self.num_states,
                             self.num_time_steps,
                             self.states_syms)
        viz.plot_identified_state_trajectory(self.sol_states,
                                             self.x.T,
                                             self.states_syms)

    def animate(self, filename=None):

        viz.animate_pendulum(self.time, self.x, 1.0, filename)

    def store_results(self):

        results = data.parse_ipopt_output(self.output_filename)

        results["datetime"] = int((datetime.datetime.now() -
                                   datetime.datetime(1970, 1, 1)).total_seconds())

        results["num_links"] = self.num_links
        results["sim_duration"] = self.duration
        results["sample_rate"] = self.sample_rate
        results["sensor_noise"] = self.sensor_noise
        results["init_type"] = self.init_type

        hasher = hashlib.sha1()
        string = ''.join([str(v) for v in results.values()])
        if sys.version_info >= (3, 0):
            hasher.update(string.encode('utf-8'))
        else:
            hasher.update(string)
        results["run_id"] = hasher.hexdigest()

        known_solution = simulate.choose_initial_conditions('known', self.x,
                                                            self.gains)
        results['initial_guess'] = self.initial_guess
        results['known_solution'] = known_solution
        results['optimal_solution'] = self.solution

        results['initial_guess_constraints'] = self.prob.con(self.initial_guess)
        results['known_solution_constraints'] = self.prob.con(known_solution)
        results['optimal_solution_constraints'] = self.prob.con(self.solution)

        results['initial_conditions'] = self.initial_conditions
        results['lateral_force'] = self.lateral_force

        file_name = 'inverted_pendulum_direct_collocation_results.h5'

        data.add_results(file_name, results)

    def cleanup(self):

        pass

    def identify(self):
        msg = """Running an identification for a {} link inverted pendulum with a {} second simulation discretized at {} hz."""
        msg = msg.format(self.num_links, self.duration, self.sample_rate)
        print('+' * len(msg))
        print(msg)
        print('+' * len(msg))

        self.compute_discretization()
        self.generate_eoms()
        self.generate_symbolic_closed_loop()
        self.find_optimal_gains()
        self.simulate()
        self.generate_objective_funcs()
        self.generate_constraint_funcs()
        self.initial_guess = \
            simulate.choose_initial_conditions(self.init_type,
                                               self.x_noise if self.sensor_noise else
                                               self.x,
                                               self.gains)
        self.optimize()
        self.store_results()

        if self.do_plot:
            self.plot()

        if self.do_animate:
            self.animate()

        self.cleanup()

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run N-Line System ID")

    parser.add_argument('-n', '--numlinks', type=int,
        help="The number of links in the pendulum.", default=1)

    parser.add_argument('-d', '--duration', type=float,
        help="The duration of the simulation in seconds.", default=1.0)

    parser.add_argument('-s', '--samplerate', type=float,
        help="The sample rate of the discretization.", default=50.0)

    parser.add_argument('-i', '--initialguess', type=str,
        help="The type of initial guess.", default='random')

    parser.add_argument('-r', '--sensornoise', action="store_true",
        help="Add noise to sensor data.",)

    parser.add_argument('-a', '--animate', action="store_true",
        help="Show the pendulum animation.",)

    parser.add_argument('-p', '--plot', action="store_true",
        help="Show result plots.")

    args = parser.parse_args()

    identifier = Identifier(args.numlinks, args.duration,
                            args.samplerate, args.initialguess,
                            args.sensornoise, args.plot, args.animate)
    identifier.identify()
