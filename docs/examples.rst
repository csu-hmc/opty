========
Examples
========

Single Pendulum Swing Up
========================

.. plot:: ../examples/pendulum_swing_up.py
   :include-source:

.. raw:: html

   <video controls>
     <source src="_static/pendulum_swing_up.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

Example console output::

   ******************************************************************************
   This program contains Ipopt, a library for large-scale nonlinear optimization.
    Ipopt is released as open source code under the Eclipse Public License (EPL).
            For more information visit http://projects.coin-or.org/Ipopt
   ******************************************************************************

   This is Ipopt version 3.12.8, running with linear solver mumps.
   NOTE: Other linear solvers might be more efficient (see Ipopt documentation).

   Number of nonzeros in equality constraint Jacobian...:     4994
   Number of nonzeros in inequality constraint Jacobian.:        0
   Number of nonzeros in Lagrangian Hessian.............:        0

   Total number of variables............................:     1500
                        variables with only lower bounds:        0
                   variables with lower and upper bounds:      500
                        variables with only upper bounds:        0
   Total number of equality constraints.................:     1002
   Total number of inequality constraints...............:        0
           inequality constraints with only lower bounds:        0
      inequality constraints with lower and upper bounds:        0
           inequality constraints with only upper bounds:        0

   iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      0  1.0051250e+01 2.37e+02 8.84e-02   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
      1  3.9969186e+00 1.83e+02 2.29e+01   0.9 8.83e+00    -  6.88e-01 2.29e-01f  1
      2  8.3301435e+00 1.38e+02 4.99e+01  -1.2 8.40e+00    -  3.00e-01 2.53e-01h  1
      3  1.8540696e+01 6.01e+01 5.63e+01  -5.3 6.38e+00    -  4.07e-01 5.70e-01h  1
      4  1.4859148e+01 3.86e+01 1.07e+02   0.7 5.98e+00    -  2.46e-01 3.63e-01f  1
      5  1.2535715e+01 2.48e+01 1.26e+02   0.7 4.54e+00    -  5.28e-01 3.68e-01f  1
      6  1.0337282e+01 2.49e+01 1.53e+02   2.6 1.38e+02    -  5.69e-02 1.96e-02f  1
      7  1.0132779e+01 1.18e+01 5.70e+02   1.4 4.35e+00    -  7.49e-01 5.39e-01h  1
      8  1.0049523e+01 7.13e+00 5.63e+02   1.3 5.26e+00    -  5.52e-01 3.95e-01h  1
      9  1.2555246e+01 4.36e+00 3.29e+02   1.3 4.84e+00    -  6.03e-01 3.96e-01h  1
   ...
   iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
     80  1.5656141e+01 2.43e-10 2.75e-06 -11.0 4.95e-04    -  1.00e+00 1.00e+00H  1
     81  1.5656141e+01 2.97e-11 1.01e-05 -11.0 3.08e-04    -  1.00e+00 1.00e+00H  1
     82  1.5656141e+01 3.06e-11 2.41e-07 -11.0 2.55e-04    -  1.00e+00 1.00e+00H  1
     83  1.5656141e+01 6.22e-14 3.29e-07 -11.0 1.37e-05    -  1.00e+00 1.00e+00H  1
     84  1.5656141e+01 6.04e-14 5.00e-07 -11.0 4.27e-05    -  1.00e+00 1.00e+00H  1
     85  1.5656141e+01 5.82e-14 4.40e-06 -11.0 9.70e-05    -  1.00e+00 1.00e+00H  1
     86  1.5656141e+01 1.03e-11 3.42e-07 -11.0 9.84e-05    -  1.00e+00 1.00e+00H  1
     87  1.5656141e+01 4.80e-14 3.65e-06 -11.0 1.19e-04    -  1.00e+00 1.00e+00H  1
     88  1.5656141e+01 3.86e-12 2.60e-07 -11.0 1.15e-04    -  1.00e+00 1.00e+00H  1
     89  1.5656141e+01 7.55e-14 1.61e-06 -11.0 4.94e-05    -  1.00e+00 1.00e+00H  1
   iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
     90  1.5656141e+01 6.89e-11 1.67e-08 -11.0 4.29e-05    -  1.00e+00 1.00e+00h  1
     91  1.5656141e+01 4.09e-14 6.37e-09 -11.0 4.51e-07    -  1.00e+00 1.00e+00h  1

   Number of Iterations....: 91

                                      (scaled)                 (unscaled)
   Objective...............:   1.5656141337135018e+01    1.5656141337135018e+01
   Dual infeasibility......:   6.3721938193329774e-09    6.3721938193329774e-09
   Constraint violation....:   4.0856207306205761e-14    4.0856207306205761e-14
   Complementarity.........:   1.0000001046147741e-11    1.0000001046147741e-11
   Overall NLP error.......:   6.3721938193329774e-09    6.3721938193329774e-09


   Number of objective function evaluations             = 133
   Number of objective gradient evaluations             = 92
   Number of equality constraint evaluations            = 133
   Number of inequality constraint evaluations          = 0
   Number of equality constraint Jacobian evaluations   = 92
   Number of inequality constraint Jacobian evaluations = 0
   Number of Lagrangian Hessian evaluations             = 0
   Total CPU secs in IPOPT (w/o function evaluations)   =      2.756
   Total CPU secs in NLP function evaluations           =      0.080

   EXIT: Optimal Solution Found.

Betts 2003
==========

.. plot:: ../examples/betts2003.py
   :include-source:

Example console output::

   ******************************************************************************
   This program contains Ipopt, a library for large-scale nonlinear optimization.
    Ipopt is released as open source code under the Eclipse Public License (EPL).
            For more information visit http://projects.coin-or.org/Ipopt
   ******************************************************************************

   This is Ipopt version 3.12.8, running with linear solver mumps.
   NOTE: Other linear solvers might be more efficient (see Ipopt documentation).

   Number of nonzeros in equality constraint Jacobian...:      992
   Number of nonzeros in inequality constraint Jacobian.:        0
   Number of nonzeros in Lagrangian Hessian.............:        0

   Total number of variables............................:      201
                        variables with only lower bounds:        0
                   variables with lower and upper bounds:        0
                        variables with only upper bounds:        0
   Total number of equality constraints.................:      200
   Total number of inequality constraints...............:        0
           inequality constraints with only lower bounds:        0
      inequality constraints with lower and upper bounds:        0
           inequality constraints with only upper bounds:        0

   iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      0  1.6334109e+00 7.36e+03 8.91e-05   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
      1  1.6283214e+00 1.05e+04 6.49e+02 -11.0 5.34e+00    -  1.00e+00 1.00e+00h  1
      2  2.5566306e-03 2.88e-04 5.62e+00 -11.0 5.62e+00    -  1.00e+00 1.00e+00h  1
      3  2.5551787e-03 1.35e-12 4.99e-05 -11.0 2.07e-02    -  1.00e+00 1.00e+00h  1
      4  2.2437570e-03 9.87e-13 2.99e-11 -11.0 8.90e+00    -  1.00e+00 1.00e+00f  1

   Number of Iterations....: 4

                                      (scaled)                 (unscaled)
   Objective...............:   2.2437570323119277e-03    2.2437570323119277e-03
   Dual infeasibility......:   2.9949274697133673e-11    2.9949274697133673e-11
   Constraint violation....:   3.8899404016729276e-14    9.8676622428683913e-13
   Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
   Overall NLP error.......:   2.9949274697133673e-11    2.9949274697133673e-11


   Number of objective function evaluations             = 5
   Number of objective gradient evaluations             = 5
   Number of equality constraint evaluations            = 5
   Number of inequality constraint evaluations          = 0
   Number of equality constraint Jacobian evaluations   = 5
   Number of inequality constraint Jacobian evaluations = 0
   Number of Lagrangian Hessian evaluations             = 0
   Total CPU secs in IPOPT (w/o function evaluations)   =      0.024
   Total CPU secs in NLP function evaluations           =      0.000

   EXIT: Optimal Solution Found.
   =========================================
   Known value of p = 3.141592653589793
   Identified value of p = 3.140935326874292
   =========================================

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
