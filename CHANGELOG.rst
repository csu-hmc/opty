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
