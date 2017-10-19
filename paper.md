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

opty operates on symbolic descriptions of ordinary differential equations to
form the constraints needed to solve trajectory optimization and parameter
identification problems using the direct collocation method [@Betts2010]. The
direct collocation problem is then solved using non-linear programming through
cyipopt [@Cyipopt2017] and IPOPT [@Watcher2006]. The software allows the user to
describe the dynamical system of interest at a high level in symbolic form
without needing to concern themselves with the numerical computation details.
To do this, SymPy [@Meurer2017] and Cython [@Behnel2011] are utilized for code
generation and just-in-time compilation to obtain wrap optimized C functions
for use in Python. The target audience are engineers and scientists interested
in solving nonlinear optimal control and parameter identification problems with
minimal overhead.

Direct collocation methods have been especially successful in the field of
human movement [@Ackermann2010, @vandenBogert2012] because those systems are
highly nonlinear, dynamically stiff, and unstable with open loop control.
Typically, closed-source tools were used for multibody dynamics (SD/Fast,
Autolev), for optimization (SNOPT), and for the programming environment
(Matlab). Recently, promising work has been done with the Opensim/Simbody
dynamics engine [@LeeUmberger2016, @LinPandy2017], but this requires that
Jacobian matrices are approximated by finite derivatives. In contrast, opty
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

Feature comparison (this should be in a table if we do it)

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

MUSCOD http://www.iwr.uni-heidelberg.de/~agbock/RESEARCH/muscod.php
(multiple shooting, not collocation)

RIOTS http://www.schwartz-home.com/riots/
(looks like single shooting, quite old)

# References
