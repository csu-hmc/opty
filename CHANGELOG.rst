Version 1.3.0
=============

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
- Added a car parallel parking example.
- Added a quadcopter drone flight example.
- Updated the generated Cython code to use memory views.
- ``Problem`` now supports solving problems with no unknown input trajectories.
- Added a cycling time trial example that uses SymPy's new muscle models.
- Corrected plot ordering for the trajectories so that mismatches no longer
  occur.
- ``Problem`` and other primary classes and methods can now be imported
  directly from the top level opty namespace, e.g. ``from opty import
  Problem``.
- Better handling of SymPy variable names that generate invalid or clashing C
  variable names.

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
