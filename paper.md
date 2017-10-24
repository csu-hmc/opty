---
title: 'opty: Software for trajectory optimization and parameter identification using direct collocation'
tags:
  - optimal control
  - trajectory optimization
  - parameter identification
  - direct collocation
  - nonlinear programming
  - symbolic computation
authors:
 - name: Jason K. Moore
   orcid: 0000-0002-8698-6143
   affiliation: 1
 - name: Antonie van den Bogert
   orcid: 0000-0002-3791-3749
   affiliation: 2
affiliations:
 - name: University of California, Davis
   index: 1
 - name: Cleveland State University
   index: 2
date: 04 June 2017
bibliography: paper.bib
---

# Summary

opty is a tool for describing and solving trajectory optimization and parameter
identification problems based on symbolic descriptions of ordinary differential
equations that describe a dynamical system. The motivation for its development
resides in the need to solve optimal control problems of biomechanical systems.
The target audience is engineers and scientists interested in solving nonlinear
optimal control and parameter identification problems with minimal
computational overhead.

A user of opty is responsible for specifying the system's dynamics (the ODEs),
the cost or fitness function, bounds on the solution, and the initial guess for
the solution. opty uses this problem specification to derive the constraints
needed to solve the optimization problem using the direct collocation method
[@Betts2010]. This method maps the problem to a non-linear programming problem
and the result is then solved numerically with an interior point optimizer,
IPOPT [@Watcher2006] which is wrapped by cyipopt [@Cyipopt2017] for use in
Python. The software allows the user to describe the dynamical system of
interest at a high level in symbolic form without needing to concern themselves
with the numerical computation details. This is made possible by utilizing
SymPy [@Meurer2017] and Cython [@Behnel2011] for code generation and
just-in-time compilation to obtain wrap optimized C functions that are
accessible in Python.

Direct collocation methods have been especially successful in the field of
human movement [@Ackermann2010, @vandenBogert2012] because those systems are
highly nonlinear, dynamically stiff, and unstable with open loop control.
Typically, closed-source tools were used for multibody dynamics (e.g. SD/Fast,
Autolev), for optimization (SNOPT), and for the programming environment
(Matlab). Recently, promising work has been done with the Opensim/Simbody
dynamics engine [@LeeUmberger2016, @LinPandy2017], but this requires that
Jacobian matrices are approximated by finite differences. In contrast, opty
provides symbolic differentiation which makes the code faster and prevents poor
convergence when the severe nonlinearity causes finite differences to be
inaccurate. Furthermore, opty allows the system to be formulated using an
implicit differential equation, which often results in far simpler equations
and better numerical conditioning. The first application of opty was in the
identification of feedback control parameters for human standing
[@MooreTGCS2015]. It should be noted that opty can use any dynamic system model
and is not limited to human movement.

Presently, opty only implements first order (backward Euler) and second order
(midpoint Euler) approximations of the dynamics. Higher accuracy and/or larger
time steps can be achieved with higher order polynomials [@PattersonRao2014],
and opty could be extended towards such capabilities. In our experience,
however, the low order discretizations provide more robust convergence to the
globally optimal trajectory in a nonlinear system when a good initial guess is
not available [@Zarei2016].

There are existing software packages that have similarities to opty. Below, is
a feature comparison to opty:

```
+========+============+================+===========================+======+========================+==========+=============================+
|        |            |                |                           |      |                        | Implicit |                             |
| Name   | Language   | License        | Derivatives               | DAEs |  Discretization        | dynamics | Solvers                     | URL |
+========+============+================+===========================+======+========================+==========+=============================+
| DIRCOL | Fortran    | Non-commercial | Finite differences        | Yes  | Piecewise linear/cubic | Yes      |                             |
+--------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| GPOPS | Matlab     | Commercial     | Automatic differentiation | No   | Pseudospectral         | No       | SNOPT                       |
+--------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| SOCS   | Fortran    | Commercial     | Finite differences        | Yes  | Euler, RK, & others    |          |                             |
+--------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| PROPT  | Matlab     | Commercial     | Analytic                  | Yes  | Pseudospectral         |          |                             |
+--------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| DIDO   | Matlab     | Commercial     | Analytic                  | No   | Pseduospectral         |          |                             |
+--------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| Casadi | C++/Python | LGPL           | Automatic differentiation | Yes  | ?                      | ?        | IPOPT, SNOPT, WORHP, KNITRO |
+--------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| PSOPT  | C++        | GPL            | Automatic differentiation |      | Pseudospectral         |          | IPOPT, SNOPT                |
|        |            |                | Sparse finite differences |      |                        |          |                             |
+--------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| GESOP  |            |                |                           |      |                        |          |                             |
+--------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| DYNOPT |            |                |                           |      |                        |          |                             |
+--------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| opty   | Python     | BSD 2-Clause   | Analytic                  | ?    | Euler, Midpoint        |          | IPOPT                       |
+--------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
```

Data for the table:

DIRCOL [@vonStryk1993]
Fortran
Free license for non-commercial use
Discretization: piecewise linear for controls, piecewise cubic for states
Analytical derivatives: no, seems to do finite difference internally
DAE (implicit dynamics): yes


GPOPS [@PattersonRao2014]
Matlab
Commercial, closed source
Discretization: pseudospectral
Analytical derivatives: generated from user's functions by automatic differentiation
DAE (implicit dynamics): no, explicit xdot only


SOCS [@Betts2010]
Fortran (Boeing)
Commercial, closed source
Discretization: Euler, RK and other ODE integration formulae
Analytical derivatives: no.  tool for sparse finite differences is provided
(Tomlab used to sell a Matlab interface, no longer available)
DAE: yes


PROPT
Matlab (TOMLAB)
Commercial, closed source
Discretization: pseudospectral
Analytical derivatives: yes, symbolic
DAE: yes


DIDO http://www.elissarglobal.com/industry/products/software-3/
Matlab
Commercial, closed source
Discretization: pseudospectral
Analytical derivatives: yes, symbolic
DAE: probably not
(tried this when it was available from TOMLAB, many years ago, and it ran out of memory for higher dimensional
problems -- TOMLAB does not have it anymore, but PROPT may be the same thing)



# References
