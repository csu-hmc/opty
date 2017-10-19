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
cyipopt [Cyipopt2017] and IPOPT [@Watcher2006]. The software allows the user to
describe the dynamical system of interest at a high level in symbolic form
without needing to concern themselves with the numerical computation details.
To do this, SymPy [@Meurer2017] and Cython [@Behnel2011] are utilized for code
generation and just-in-time compilation to obtain wrap optimized C functions
for use in Python. The target audience are engineers and scientists interested
in solving nonlinear optimal control and parameter identification problems with
minimal overhead.

# References
