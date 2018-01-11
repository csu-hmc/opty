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

<table border="1" class="docutils">
<colgroup>
<col width="5%" />
<col width="8%" />
<col width="5%" />
<col width="6%" />
<col width="12%" />
<col width="10%" />
<col width="8%" />
<col width="7%" />
<col width="39%" />
</colgroup>
<thead valign="bottom">
<tr><th class="head">Name</th>
<th class="head">Citation</th>
<th class="head">Language</th>
<th class="head">License</th>
<th class="head">Derivatives</th>
<th class="head">Discretization</th>
<th class="head">Implicit Dynamics</th>
<th class="head">Solvers</th>
<th class="head">Project Website</th>
</tr>
</thead>
<tbody valign="top">
<tr><td>Casadi <a class="footnote-reference" href="#id2" id="id1">[1]</a></td>
<td>[&#64;Andersson2013]</td>
<td>C++,
Python,
Octave,</td>
<td>LGPL</td>
<td>Automatic differentiation</td>
<td>None</td>
<td>Yes</td>
<td>IPOPT, WORHP,
SNOPT, KNITRO</td>
<td><a class="reference external" href="https://github.com/casadi/casadi/wiki">Casadi Website</a></td>
</tr>
<tr><td>DIDO</td>
<td>[&#64;Ross2002]</td>
<td>Matlab</td>
<td>Commercial</td>
<td>Analytic</td>
<td>Pseudospectral</td>
<td>Yes</td>
<td>built-in</td>
<td><a class="reference external" href="http://www.elissarglobal.com/industry/products/software-3/">DIDO Website</a></td>
</tr>
<tr><td>DIRCOL</td>
<td>[&#64;vonStryk1993]</td>
<td>Fortran</td>
<td>Non-commercial</td>
<td>Finite differences</td>
<td>Piecewise linear/cubic</td>
<td>Yes</td>
<td>NPSOL, SNOPT</td>
<td><a class="reference external" href="http://www.sim.informatik.tu-darmstadt.de/en/res/sw/dircol/">DIRCOL Website</a></td>
</tr>
<tr><td>DYNOPT</td>
<td>[&#64;Cizniar2005]</td>
<td>Matlab</td>
<td>Custom Open
Source,
Non-commercial</td>
<td>Must be supplied by user</td>
<td>Pseudospectral</td>
<td>Mass matrix</td>
<td>fmincon</td>
<td><a class="reference external" href="https://bitbucket.org/dynopt/">DYNOPT Code and Documentation</a></td>
</tr>
<tr><td>FROST</td>
<td>[&#64;Hereid2017]</td>
<td>Matlab,
Mathematica</td>
<td>BSD 3-Clause</td>
<td>Analytic</td>
<td>?</td>
<td>?</td>
<td>IPOPT, fmincon</td>
<td><a class="reference external" href="http://ayonga.github.io/frost-dev/">FROST Documentation</a></td>
</tr>
<tr><td>GESOP</td>
<td>[&#64;Gath2001]</td>
<td>Matlab, C,
Fortan, Ada</td>
<td>Commercial</td>
<td>?</td>
<td>Pseudospectral</td>
<td>No</td>
<td>SLLSQP, SNOPT,
SOCS</td>
<td><a class="reference external" href="https://www.astos.de/products/gesop">Astos Solutions Gmbh</a></td>
</tr>
<tr><td>GPOPS</td>
<td>[&#64;PattersonRao2014]</td>
<td>Matlab</td>
<td>Commercial</td>
<td>Automatic differentiation</td>
<td>Pseudospectral</td>
<td>No</td>
<td>SNOPT, IPOPT</td>
<td><a class="reference external" href="http://www.gpops2.com/">GPOPS Website</a></td>
</tr>
<tr><td>opty</td>
<td>NA</td>
<td>Python</td>
<td>BSD 2-Clause</td>
<td>Analytic</td>
<td>Euler, Midpoint</td>
<td>Yes</td>
<td>IPOPT</td>
<td><a class="reference external" href="http://opty.readthedocs.io">opty Documentation</a></td>
</tr>
<tr><td>OTIS</td>
<td>[&#64;Hargraves1987]</td>
<td>Fortran</td>
<td>US Export
Controlled</td>
<td>?</td>
<td>Gauss-Labatto,
Pseudospectral</td>
<td>Yes</td>
<td>SNOPT</td>
<td><a class="reference external" href="https://otis.grc.nasa.gov">OTIS Website</a></td>
</tr>
<tr><td>PROPT</td>
<td>[&#64;Rutquist2010]</td>
<td>Matlab</td>
<td>Commercial</td>
<td>Analytic</td>
<td>Pseudospectral</td>
<td>Yes</td>
<td>SNOPT, KNITRO</td>
<td><a class="reference external" href="http://tomdyn.com/index.html">TOMDYN Website</a></td>
</tr>
<tr><td>PSOPT</td>
<td>[&#64;Becerra2010]</td>
<td>C++</td>
<td>GPL</td>
<td>Automatic differentiation,
Sparse finite differences</td>
<td>Pseudospectral, RK</td>
<td>Yes</td>
<td>IPOPT, SNOPT</td>
<td><a class="reference external" href="http://www.psopt.org/">PSOPT Website</a></td>
</tr>
<tr><td>SOCS</td>
<td>[&#64;Betts2010]</td>
<td>Fortran</td>
<td>Commercial</td>
<td>Finite differences</td>
<td>Euler, RK, &amp; others</td>
<td>Yes</td>
<td>built-in</td>
<td><a class="reference external" href="http://www.boeing.com/assets/pdf/phantom/socs/docs/SOCS_Users_Guide.pdf">SOCS Documentation</a></td>
</tr>
</tbody>
</table>
<table class="docutils footnote" frame="void" id="id2" rules="none">
<colgroup><col class="label" /><col /></colgroup>
<tbody valign="top">
<tr><td class="label"><a class="fn-backref" href="#id1">[1]</a></td><td>Casadi does not have a built in direct collocation transcription but includes examples which show how to do so for specific problems.</td></tr>
</tbody>
</table>

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
