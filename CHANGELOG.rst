Version 1.5.0.dev0
==================

- Bump dependency minimum versions to match those in Ubuntu 22.04 (Jammy).
- Support added for providing additional path constraints (equality or
  inequality).
- New Python (NumPy) backend added to allow low performance use without Cython
  and need for JIT compilation.
- ``plot_constraint_violations()`` can now optionally plot on subplots for each
  constraint.
- ``plot_trajectories()`` can optionally show the bounds.
- Method added to compute the time vector.
- Method added to check if a solution respects the bounds and flag to check the
  initial guess before solving.
- openmp parallelization now works on Mac.
- Speed of building the examples improved by using the NumPy backend and
  animation frame reduction.
- Examples Added:

  - Bicycle countersteer example showing use of inputs consisting of variables
    and their time derivatives.
  - Quarter car riding over a bumpy road showing a simultaneous control
    trajectory and parameter identification solution.
  - Brachistochrone example.
  - Light diffraction example showing use of smoothened discontinuous
    functions.
  - Car staying within a race course.
  - Approximation of a multiphase problem.

Version 1.4.0
=============

- Dropped support for Python 3.8.
- Added support for Python 3.13.
- Added explicit dependency on setuptools.
- ``ConstraintCollocator`` raises an error if all equations of motion are
  algebraic, i.e. none are differential equations.
- ``plot_constraint_violations()`` now returns a variable number of subplot
  rows depending on the number of instance constraints. This makes the plot
  readable with large numbers of instance constraints.
- Internal common sub expression replacement symbols are now assumed to be
  real.
- Improvements to docstrings of ``Problem.solve()``.
- Added ``Problem.parse_free()`` to simplify use of the ``parse_free()``
  function.
- Added attribute descriptions to ``ConstraintCollocator`` docstring.
- Support for single first order differential equation instead of limiting to
  two first order (i.e. only second order systems).
- Enabled math constants on Windows to support ``sympy.pi``, for example.
- Made SciPy an optional dependency (was required).
- Fixed bug from backwards incompatible change in Python 3.13 for docstring
  indentation.
- Fixed bug where ``Problem.plot_`` methods did not return anything.
- Fixed bug in trajectory plots so that the input trajectories are labeled with
  known, unknown, or both types of inputs.
- Made all ``ConstraintCollocator`` attributes properties with no setter
  methods, i.e. everything must be passed into the constructor to properly
  construct the object.
- Separated examples into beginner, intermediate, and advanced groups.
- Added ``MathJaxRepr`` for printing SymPy expressions in the example gallery.
- Use MathJax v2 in the documentation so that long expressions will line wrap.
- Examples added:

  - Ball rolling on spinning disc
  - Car moving around pylons
  - Car parking into a garage
  - Crane moving a load
  - Delay equation (inequality constraints example)
  - Human gait
  - Mississippi steamboat
  - Non-contiguous parameter identification
  - ODE vs DAE comparison
  - Particle moving through a helical tube
  - Single EoM & hypersensitive control
  - Sit-to-stand
  - Standing balance control identification

Version 1.3.0
=============

- Added support for Python 3.12.
- Added a function that generates a numerical objective function and its
  gradient from a symbolic objective function: ``create_objective_function()``.
- Fixed constraint violation plot on ``Problem`` by removing the assumption
  that each equation of motion was simply a function of one state.
- Added an option to display the Cython compilation output and to automatically
  display the output if compilation fails. Display does not always work on
  Windows due to limitations in Python's capturing ``STDOUT`` on Windows with
  certain system encodings.
- Added support for variable duration solutions by passing in a symbol for the
  node time interval instead of a float.
- Switched to Sphinx Gallery for displaying examples in the documentation.
- Moved the three documentation examples to the Sphinx Gallery page.
- Added new examples:

  - A variable duration pendulum swing up example.
  - A car parallel parking example.
  - A quadcopter drone flight example.
  - A cycling time trial example that uses SymPy's new muscle models.
  - A block sliding over a hill example.

- Updated the generated Cython code to use memory views.
- ``Problem`` now supports solving problems with no unknown input trajectories.
- Corrected plot ordering for the trajectories so that mismatches no longer
  occur.
- Improved default plot display for larger number of variables and support
  customizing axes to default plots.
- ``Problem`` and other primary classes and methods can now be imported
  directly from the top level opty namespace, e.g. ``from opty import
  Problem``.
- Better handling of SymPy variable names that generate invalid or clashing C
  variable names by appending an underscore to all opty generated C variable
  names.
- Switched to pytest for unit testing.

Version 1.2.0
=============

- Dropped support for Python 2.7, 3.6, & 3.7.
- Added support for Python 3.9, 3.10, & 3.11.
- Bumped minimum dependencies up to versions released around mid 2020, except
  for cyipopt which is set to 1.1.0 since that is the first version released on
  PyPi under the name cyipopt (instead of ipopt).
- Much faster symbolic Jacobian algorithm introduced. For constraints made up
  of hundreds of thousands SymPy operations there can be greater than 200X
  performance increase.
- logging.info() used for providing information to the user.
- Moved to Github Actions continous integration.

Version 1.1.0
=============

- Added support for Windows.
- Drop support for Python 3.5, add support for 3.7 and 3.8.

Version 1.0.0
=============

- Added JOSS paper.
- Added theory section to the documentation.
- Added optional parallel execution if openmp is installed.
- Fixed a bug in plot_trajectories.
- Adjusted the pendulum swing up torque bounds.
- Updated examples to work with newer dependency versions.

Version 0.2.0
=============

- Added Sphinx documentation and Read The Docs integration.
- Added plotting to the Problem class and matplotlib as an optional dependency.
- Added conda forge installation instructions and Anaconda badge to the README.
- park2004 example now works with both Python 2.7 and 3.5+.
- Bumped the min dependencies for SymPy and PyDy to 1.0.0 and 0.3.0.

Version 0.1.1
=============

- Added a MANIFEST.in file.

Version 0.1.0
=============

- Initial release.
