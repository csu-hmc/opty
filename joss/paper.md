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
equations and differential algebriac equations that describe a dynamical
system. The motivation for its development resides in the need to solve optimal
control problems of biomechanical systems. The target audience is engineers
and scientists interested in solving nonlinear optimal control and parameter
identification problems with minimal computational overhead.

A user of opty is responsible for specifying the system's dynamics (the
ODE/DAEs), the cost or fitness function, bounds on the solution, and the
initial guess for the solution. opty uses this problem specification to derive
the constraints needed to solve the optimization problem using the direct
collocation method [@Betts2010]. This method maps the problem to a non-linear
programming problem and the result is then solved numerically with an interior
point optimizer IPOPT [@Wachter2006] which is wrapped by cyipopt [@Cyipopt2017]
for use in Python. The software allows the user to describe the dynamical
system of interest at a high level in symbolic form without needing to concern
themselves with the numerical computation details. This is made possible by
utilizing SymPy [@Meurer2017] and Cython [@Behnel2011] for code generation and
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

There are existing software packages that can solve optimal control problems
and have have similarities to opty. Below, is a feature comparison of those we
are aware of:

\tiny

+-------------+---------------------+-------------+----------------+-----------------------------+------------------------+-------------------+-----------------+------------------------+
| Name        | Citation            | Language    | License        | Derivatives                 |  Discretization        | Implicit Dynamics | Solvers         | Project Website        |
+=============+=====================+=============+================+=============================+========================+===================+=================+========================+
| Casadi [^1] | [@Andersson2013]    | C++,        | LGPL           | Automatic differentiation   | None                   | Yes               | IPOPT, WORHP,   | [Casadi Website]       |
|             |                     | Python,     |                |                             |                        |                   | SNOPT, KNITRO   |                        |
|             |                     | Octave,     |                |                             |                        |                   |                 |                        |
+-------------+---------------------+-------------+----------------+-----------------------------+------------------------+-------------------+-----------------+------------------------+
| DIDO        | [@Ross2002]         | Matlab      | Commercial     | Analytic                    | Pseudospectral         | Yes               | built-in        | [DIDO Website]         |
+-------------+---------------------+-------------+----------------+-----------------------------+------------------------+-------------------+-----------------+------------------------+
| DIRCOL      | [@vonStryk1992]     | Fortran     | Non-commercial | Finite differences          | Piecewise linear/cubic | Yes               | NPSOL, SNOPT    | [DIRCOL Website]       |
+-------------+---------------------+-------------+----------------+-----------------------------+------------------------+-------------------+-----------------+------------------------+
| DYNOPT      | [@Cizniar2005]      | Matlab      | Custom Open    | Must be supplied by user    | Pseudospectral         | Mass matrix       | fmincon         | [DYNOPT Code]          |
|             |                     |             | Source,        |                             |                        |                   |                 |                        |
|             |                     |             | Non-commercial |                             |                        |                   |                 |                        |
+-------------+---------------------+-------------+----------------+-----------------------------+------------------------+-------------------+-----------------+------------------------+
| FROST       | [@Hereid2017]       | Matlab,     | BSD 3-Clause   | Analytic                    | ?                      | ?                 | IPOPT, fmincon  | [FROST Documentation]  |
|             |                     | Mathematica |                |                             |                        |                   |                 |                        |
+-------------+---------------------+-------------+----------------+-----------------------------+------------------------+-------------------+-----------------+------------------------+
| GESOP       | [@Gath2001]         | Matlab, C,  | Commercial     | ?                           | Pseudospectral         | No                | SLLSQP, SNOPT,  | [Astos Solutions Gmbh] |
|             |                     | Fortan, Ada |                |                             |                        |                   | SOCS            |                        |
+-------------+---------------------+-------------+----------------+-----------------------------+------------------------+-------------------+-----------------+------------------------+
| GPOPS       | [@PattersonRao2014] | Matlab      | Commercial     | Automatic differentiation   | Pseudospectral         | No                | SNOPT, IPOPT    | [GPOPS Website]        |
+-------------+---------------------+-------------+----------------+-----------------------------+------------------------+-------------------+-----------------+------------------------+
| opty        | NA                  | Python      | BSD 2-Clause   | Analytic                    | Euler, Midpoint        | Yes               | IPOPT           | [opty Documentation]   |
+-------------+---------------------+-------------+----------------+-----------------------------+------------------------+-------------------+-----------------+------------------------+
| OTIS        | [@Hargraves1987]    | Fortran     | US Export      | ?                           | Gauss-Labatto,         | Yes               | SNOPT           | [OTIS Website]         |
|             |                     |             | Controlled     |                             | Pseudospectral         |                   |                 |                        |
+-------------+---------------------+-------------+----------------+-----------------------------+------------------------+-------------------+-----------------+------------------------+
| PROPT       | [@Rutquist2010]     | Matlab      | Commercial     | Analytic                    | Pseudospectral         | Yes               | SNOPT, KNITRO   | [TOMDYN Website]       |
+-------------+---------------------+-------------+----------------+-----------------------------+------------------------+-------------------+-----------------+------------------------+
| PSOPT       | [@Becerra2010]      | C++         | GPL            | Automatic differentiation,  | Pseudospectral, RK     | Yes               | IPOPT, SNOPT    | [PSOPT Website]        |
|             |                     |             |                | Sparse finite differences   |                        |                   |                 |                        |
+-------------+---------------------+-------------+----------------+-----------------------------+------------------------+-------------------+-----------------+------------------------+
| SOCS        | [@Betts2010]        | Fortran     | Commercial     | Finite differences          | Euler, RK, & others    | Yes               | built-in        | [SOCS Documentation]   |
+-------------+---------------------+-------------+----------------+-----------------------------+------------------------+-------------------+-----------------+------------------------+

[^1]: Casadi does not have a built in direct collocation transcription but includes examples which show how to do so for specific problems.

[Casadi Website]: https://github.com/casadi/casadi/wiki
[DIDO Website]: http://www.elissarglobal.com/industry/products/software-3
[DIRCOL Website]: http://www.sim.informatik.tu-darmstadt.de/en/res/sw/dircol
[DYNOPT Code]: https://bitbucket.org/dynopt
[FROST Documentation]: http://ayonga.github.io/frost-dev
[Astos Solutions Gmbh]: https://www.astos.de/products/gesop
[GPOPS Website]: http://www.gpops2.com
[opty Documentation]: http://opty.readthedocs.io
[OTIS Website]: https://otis.grc.nasa.gov
[TOMDYN Website]: http://tomdyn.com/index.html
[PSOPT Website]: http://www.psopt.org
[SOCS Documentation]: http://www.boeing.com/assets/pdf/phantom/socs/docs/SOCS_Users_Guide.pdf

\normalsize

Each of these software packages offer a different combination of attributes and
features that make it useful for different problems. opty is the only package
that has a liberal open source license for itself and its dependencies,
following precedent set by other core scientific Python packages. This allows
anyone to use and modify the code without having to share the source of their
application. opty also is the only package, open source or not, that allows (in
fact forces) the user to describe their problem via a high level symbolic
mathematical description using the API of a widely used computer algebra system
instead of a domain specific language. This relieves the user from having to
translate the much simpler continuous problem definition into a discretize NLP
problem. opty leverages the popular Scientific Python core tools like NumPy,
SymPy, Cython, and matplotlib allowing users to include opty code into Python
programs. Lastly, the numeric code generated by opty to evaluate the NLP
constraints is optimized providing extremely efficient parallel evaluation of
the contraints. This becomes very valuable for high dimensional dynamics. opty
currently does not offer a wide range of discretization methods nor support for
solvers other than IPOPT, but those could relatively easily be added based on
user need.

# References
