#!/usr/bin/env python

import os

import numpy as np
import tables
import pandas

from utils import parse_free


def compute_gain_error(filename):
    # root mean square of gain error
    df = load_results_table(filename)
    rms = []
    for run_id, sim_dur, sample_rate in zip(df['run_id'],
                                            df['sim_duration'],
                                            df['sample_rate']):
        run_dict = load_run(filename, run_id)
        num_states = len(run_dict['initial_conditions'])
        num_time_steps = int(sim_dur * sample_rate)
        __, __, known_gains = parse_free(run_dict['known_solution'],
                                         num_states, 0, num_time_steps)
        __, __, optimal_gains = parse_free(run_dict['optimal_solution'],
                                           num_states, 0, num_time_steps)
        rms.append(np.sqrt(np.sum((known_gains - optimal_gains)**2)))
    df['RMS of Gains'] = np.asarray(rms)
    return df


def load_results_table(filename):

    handle = tables.openFile(filename, 'r')
    df = pandas.DataFrame.from_records(handle.root.results[:])
    handle.close()
    return df


def load_run(filename, run_id):
    handle = tables.openFile(filename, 'r')
    group = getattr(handle.root.arrays, run_id)
    d = {}
    for array_name in group.__members__:
        d[array_name] = getattr(group, array_name)[:]
    handle.close()
    return d


def parse_ipopt_output(file_name):
    """Returns a dictionary with the IPOPT summary results.

    Notes
    -----

    This is an example of the summary at the end of the file:

    Number of Iterations....: 1013

                                       (scaled)                 (unscaled)
    Objective...............:   2.8983286604029537e-04    2.8983286604029537e-04
    Dual infeasibility......:   4.7997817057236348e-09    4.7997817057236348e-09
    Constraint violation....:   9.4542809291867735e-09    9.8205754639479892e-09
    Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
    Overall NLP error.......:   9.4542809291867735e-09    9.8205754639479892e-09


    Number of objective function evaluations             = 6881
    Number of objective gradient evaluations             = 1014
    Number of equality constraint evaluations            = 6900
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 1014
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 0
    Total CPU secs in IPOPT (w/o function evaluations)   =     89.023
    Total CPU secs in NLP function evaluations           =    457.114

    """

    with open(file_name, 'r') as f:
        output = f.readlines()

    results = {}

    for line in output:
        if 'Number of Iterations....:' in line and 'Maximum' not in line:
            results['num_iterations'] = int(line.split(':')[1].strip())

        elif 'Number of objective function evaluations' in line:
            results['num_obj_evals'] = int(line.split('=')[1].strip())

        elif 'Number of objective gradient evaluations' in line:
            results['num_obj_grad_evals'] = int(line.split('=')[1].strip())

        elif 'Number of equality constraint evaluations' in line:
            results['num_con_evals'] = int(line.split('=')[1].strip())

        elif 'Number of equality constraint Jacobian evaluations' in line:
            results['num_con_jac_evals'] = int(line.split('=')[1].strip())

        elif 'Total CPU secs in IPOPT (w/o function evaluations)' in line:
            results['time_ipopt'] = float(line.split('=')[1].strip())

        elif 'Total CPU secs in NLP function evaluations' in line:
            results['time_func_evals'] = float(line.split('=')[1].strip())

    return results


def create_database(file_name):
    """Creates an empty optimization results database on disk if it doesn't
    exist."""

    class RunTable(tables.IsDescription):
        run_id = tables.StringCol(40)  # sha1 hashes are 40 char long
        init_type = tables.StringCol(10)
        datetime = tables.Time32Col()
        num_links = tables.Int32Col()
        sim_duration = tables.Float32Col()
        sample_rate = tables.Float32Col()
        sensor_noise = tables.BoolCol()
        num_iterations = tables.Int32Col()
        num_obj_evals = tables.Int32Col()
        num_obj_grad_evals = tables.Int32Col()
        num_con_evals = tables.Int32Col()
        num_con_jac_evals = tables.Int32Col()
        time_ipopt = tables.Float32Col()
        time_func_evals = tables.Float32Col()

    if not os.path.isfile(file_name):
        title = 'Inverted Pendulum Direct Collocation Results'
        h5file = tables.open_file(file_name,
                                  mode='w',
                                  title=title)
        h5file.create_table('/', 'results', RunTable,
                            'Optimization Results Table')
        h5file.create_group('/', 'arrays', 'Optimization Parameter Arrays')

        h5file.close()


def add_results(file_name, results):

    if not os.path.isfile(file_name):
        create_database(file_name)

    h5file = tables.open_file(file_name, mode='a')

    print('Adding run {} to the database.'.format(results['run_id']))

    group_name = 'Optimization Run #{}'.format(results['run_id'])
    run_array_dir = h5file.create_group(h5file.root.arrays,
                                        results['run_id'],
                                        group_name)
    arrays = ['initial_guess',
              'known_solution',
              'optimal_solution',
              'initial_guess_constraints',
              'known_solution_constraints',
              'optimal_solution_constraints',
              'initial_conditions',
              'lateral_force']

    for k in arrays:
        v = results.pop(k)
        h5file.create_array(run_array_dir, k, v)

    table = h5file.root.results
    opt_row = table.row

    for k, v in results.items():
        opt_row[k] = v

    opt_row.append()

    table.flush()

    h5file.close()
