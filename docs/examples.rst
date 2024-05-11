====================
Examples (Old Style)
====================

Pendulum Parameter Identification
=================================

Identifies a single constant from measured pendulum swing data. The default
initial guess for trajectories are the known continuous solution plus
artificial Gaussian noise and a random positive value for the parameter.

.. plot:: ../examples/vyasarayani2011.py
   :include-source:

Example console output::

   Using noisy measurements for the trajectory initial guess and a random positive value for the parameter.

   ******************************************************************************
   This program contains Ipopt, a library for large-scale nonlinear optimization.
    Ipopt is released as open source code under the Eclipse Public License (EPL).
            For more information visit http://projects.coin-or.org/Ipopt
   ******************************************************************************

   This is Ipopt version 3.12.8, running with linear solver mumps.
   NOTE: Other linear solvers might be more efficient (see Ipopt documentation).

   Number of nonzeros in equality constraint Jacobian...:    49990
   Number of nonzeros in inequality constraint Jacobian.:        0
   Number of nonzeros in Lagrangian Hessian.............:        0

   Total number of variables............................:    10001
                        variables with only lower bounds:        0
                   variables with lower and upper bounds:        0
                        variables with only upper bounds:        0
   Total number of equality constraints.................:     9998
   Total number of inequality constraints...............:        0
           inequality constraints with only lower bounds:        0
      inequality constraints with lower and upper bounds:        0
           inequality constraints with only upper bounds:        0

   iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      0  0.0000000e+00 8.42e+01 0.00e+00   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
      1  5.0922544e-01 2.04e+01 6.00e+02 -11.0 7.06e+01    -  1.00e+00 1.00e+00h  1
      2  5.9686839e-01 2.79e+00 5.72e+01 -11.0 1.11e+01    -  1.00e+00 1.00e+00h  1
      3  5.7926200e-01 7.31e-02 5.80e+00 -11.0 1.17e+00    -  1.00e+00 1.00e+00h  1
      4  5.7694616e-01 4.40e-04 5.54e-02 -11.0 5.54e-02    -  1.00e+00 1.00e+00h  1
      5  5.0524814e-01 9.16e-02 7.30e-01 -11.0 1.61e+00    -  1.00e+00 5.00e-01f  2
      6  1.3347194e-01 3.11e-02 3.03e-01 -11.0 3.96e-01    -  1.00e+00 1.00e+00h  1
      7  1.3053329e-01 5.72e-04 1.09e-03 -11.0 4.14e-02    -  1.00e+00 1.00e+00h  1
      8  1.3043109e-01 1.40e-05 1.82e-03 -11.0 8.64e-03    -  1.00e+00 1.00e+00h  1
      9  1.2944372e-01 1.88e-05 1.82e-03 -11.0 1.28e-02    -  1.00e+00 1.00e+00h  1
   iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
     10  1.2775048e-01 2.13e-04 1.49e-03 -11.0 3.96e-02    -  1.00e+00 1.00e+00h  1
     11  1.2747061e-01 5.17e-05 5.94e-04 -11.0 1.90e-02    -  1.00e+00 1.00e+00h  1
     12  1.2747075e-01 2.36e-09 1.33e-09 -11.0 8.97e-05    -  1.00e+00 1.00e+00h  1

   Number of Iterations....: 12

                                      (scaled)                 (unscaled)
   Objective...............:   1.2747074619675813e-01    1.2747074619675813e-01
   Dual infeasibility......:   1.3326612572804250e-09    1.3326612572804250e-09
   Constraint violation....:   2.3617712230361576e-09    2.3617712230361576e-09
   Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
   Overall NLP error.......:   2.3617712230361576e-09    2.3617712230361576e-09


   Number of objective function evaluations             = 15
   Number of objective gradient evaluations             = 13
   Number of equality constraint evaluations            = 15
   Number of inequality constraint evaluations          = 0
   Number of equality constraint Jacobian evaluations   = 13
   Number of inequality constraint Jacobian evaluations = 0
   Number of Lagrangian Hessian evaluations             = 0
   Total CPU secs in IPOPT (w/o function evaluations)   =      3.052
   Total CPU secs in NLP function evaluations           =      0.048

   EXIT: Optimal Solution Found.
   =========================================
   Known value of p = 10.0
   Initial guess for p = 92.67287860789347
   Identified value of p = 10.00138902660221
   =========================================
