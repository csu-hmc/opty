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
+========+=====================+============+================+===========================+======+========================+==========+=============================+
|        |                     |            |                |                           |      |                        | Implicit |                             |
| Name   | Citation            | Language   | License        | Derivatives               | DAEs |  Discretization        | dynamics | Solvers                     |
+========+=====================+============+===========================+======+========================+==========+==============================================+
| Casadi |                     | C++/Python | LGPL           | Automatic differentiation | Yes  | ?                      | Yes      | IPOPT, SNOPT, WORHP, KNITRO |
+--------+---------------------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| DIDO   | [@Ross2002]         | Matlab     | Commercial     | Analytic                  | No   | Pseudospectral         | Yes      | built-in                    |
+--------+---------------------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| DIRCOL | [@vonStryk1993]     | Fortran    | Non-commercial | Finite differences        | Yes  | Piecewise linear/cubic | Yes      | NPSOL, SNOPT                |
+--------+---------------------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| DYNOPT |                     |            |                |                           |      |                        |          |                             |
+--------+---------------------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| GESOP  |                     |            |                |                           |      |                        |          |                             |
+--------+---------------------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| GPOPS  | [@PattersonRao2014] | Matlab     | Commercial     | Automatic differentiation | No   | Pseudospectral         | No       | SNOPT, IPOPT                |
+--------+---------------------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| opty   |                     | Python     | BSD 2-Clause   | Analytic                  | Yes  | Euler, Midpoint        | Yes      | IPOPT                       |
+--------+---------------------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| PROPT  |                     | Matlab     | Commercial     | Analytic                  | Yes  | Pseudospectral         | Yes      | SNOPT, KNITRO               |
+--------+---------------------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| PSOPT  |                     | C++        | GPL            | Automatic differentiation |      | Pseudospectral         |          | IPOPT, SNOPT                |
|        |                     |            |                | Sparse finite differences |      |                        |          |                             |
+--------+---------------------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
| SOCS   | [@Betts2010]        | Fortran    | Commercial     | Finite differences        | Yes  | Euler, RK, & others    | Yes      | built-in                    |
+--------+---------------------+------------+----------------+---------------------------+------+------------------------+----------+-----------------------------+
```

Should we add URLs and references in the Table? Let's put the urls in the bib
file and just a citation here.

DIRCOL [@vonStryk1993] http://www.sim.informatik.tu-darmstadt.de/en/res/sw/dircol/
GPOPS [@PattersonRao2014] http://www.gpops2.com/
SOCS [@Betts2010] http://www.boeing.com/assets/pdf/phantom/socs/docs/SOCS_Users_Guide.pdf
PROPT http://tomdyn.com/index.html
DIDO [@Ross2002] http://www.elissarglobal.com/industry/products/software-3/

What is unique about opty, and can we use the Table or text to show that? The
just in time compilation?  Scales well to high-dimensional dynamics?

Things I think make opty unique:

- BSD license
- High level language: Python
- SymPy is core dependency: write problem description as SymPy expressions
- Efficient sparse constraint eval (not sure how this compares to others)

Casadi does not have collocation built in, so maybe does not belong in the
table. It is basically a NLP solver interface with automatic differentiation
to generate jacobians and hessians. Direct collocation is mentioned (very)
briefly in the manual, and used in one of the examples:
https://github.com/casadi/casadi/blob/master/docs/examples/matlab/direct_collocation.m

Good catch.

# References
