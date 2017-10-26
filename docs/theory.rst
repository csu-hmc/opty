======
Theory
======

Given at a set of continuous differential equations in time in implicit form:

.. math::

   \mathbf{f}(\dot{\mathbf{x}}(t), \mathbf{x}(t), \mathbf{r}(t), \mathbf{p}, t) = \mathbf{0}

where:

- :math:`\mathbf{x}` is the state vector
- :math:`\mathbf{r}` is the vector of specified inputs
- :math:`\mathbf{p}` is the vector of constant parameters
- :math:`t` is time

one can then break up :math:`\mathbf{r}` and :math:`\mathbf{p}` into known $k$
and unknown $u$ quantities:

.. math::

   \mathbf{r}(t) = \left[ \mathbf{r}_k(t) \quad \mathbf{r}_u(t) \right]^T

   \mathbf{p} = \left[ \mathbf{p}_k \quad \mathbf{p}_u \right]^T

Then there are optimal state trajectories :math:`\mathbf{x}`, optimal input
trajectories :math:`\mathbf{r}_u` and optimal parameter values
:math:`\mathbf{p}_u` if some cost function, :math:`J(\mathbf{x}, \mathbf{r}_u,
\mathbf{p})` is specified and any optional constraints on these variables in
the form of:

.. math::

   \mathbf{x}^L \leq \mathbf{x} \leq \mathbf{x}^U \\
   \mathbf{u}^L \leq \mathbf{u} \leq \mathbf{u}^U \\
   \mathbf{p}^L \leq \mathbf{p} \leq \mathbf{p}^U

Constraints are commonly:

- path constraints
- bounds on the trajectories or parameter values
- specific trajectory or parameter values at instances of time

This type of problem called an `optimal control`_ problem with the objective of
finding the open loop input trajectories and/or the optimal system parameters
such that the dynamical system evolves in time in a way that meets the
objective and constraints. This problem can be rewritten as a `nonlinear
programming`_ (NLP) problem using `direct collocation`_ methods and solved
using a variety of NLP optimization algorithms.

.. _optimal control: https://en.wikipedia.org/wiki/Optimal_control
.. _nonlinear programming: https://en.wikipedia.org/wiki/Nonlinear_programming
.. _direct collocation: https://en.wikipedia.org/wiki/Trajectory_optimization#Direct_collocation

The direct collocation formulation formulates the differential equations as
discretized constraints using a variety of discrete integration methods and
appended as additional optimality constraints to the ones shown above. This
ensures that the dynamics are satisfied at each discretized time instance and
relieves the need to integrate the differential equations as one does in
shooting optimization.

opty currently supports two integration methods: the `backward Euler method`_
and the `midpoint method`_.

.. _backward Euler method: https://en.wikipedia.org/wiki/Backward_Euler_method
.. _midpoint method: https://en.wikipedia.org/wiki/Midpoint_method

If :math:`h` is the time interval between the discretized instances of time
:math:`t_i`, using the backward Euler method of integration, the approximation
of the derivative of the state vector is:

.. math::

   \frac{d\mathbf{x}}{dt} & \approx & \frac{\mathbf{x}_i - \mathbf{x}_{i-1}}{h} \\
   \mathbf{x}(t_i) & = & \mathbf{x}_i \\
   \mathbf{r}(t_i) & = & \mathbf{r}_i

Using the midpoint method the approximation of the derivative of the state
vector is:

.. math::

   \frac{d\mathbf{x}}{dt} & \approx & \frac{\mathbf{x}_{i+1} - \mathbf{x}_{i}}{h} \\
   \mathbf{x}(t_i) & = & \frac{\mathbf{x}_i + \mathbf{x}_{i+1}}{2} \\
   \mathbf{r}(t_i) & = & \frac{\mathbf{r}_i + \mathbf{r}_{i+1}}{2}

The discretized differential equation :math:`\mathbf{f}_i` can the be written
using both of the above approximations. For the backward Euler method:

.. math::

   \mathbf{f}_i = \mathbf{f}\left(\frac{\mathbf{x}_i - \mathbf{x}_{i-1}}{h},
                                  \mathbf{x}_i, \mathbf{r}_i, \mathbf{p}, t_i\right) = 0

For the midpoint method:

.. math::

   \mathbf{f}_i = \mathbf{f}\left(\frac{\mathbf{x}_i - \mathbf{x}_{i-1}}{h},
                                  \frac{\mathbf{x}_i + \mathbf{x}_{i+1}}{2},
                                  \frac{\mathbf{r}_i + \mathbf{r}_{i+1}}{2},
                                  \mathbf{p}, t_i\right) = 0

If the number of time values considered is :math:`m` and the number of states
is :math:`n`, the above equations will create :math:`mn` constraint equations
and the optimization problem can formally be written as:

.. math::

   & \underset{\mathbf{x} \in \mathbb{R}^n,
               \mathbf{r}_u \in \mathbb{R}^p,
               \mathbf{p}_u \in \mathbb{R}^q}
              {\text{min}}
   & & J(\mathbf{x}, \mathbf{r}_u, \mathbf{p}_u) \\
   & \text{s.t.}
   & & \mathbf{0} \leq \mathbf{f} \leq \mathbf{0} \\
   & & & \mathbf{x}^L \leq \mathbf{x} \leq \mathbf{x}^U \\
   & & & \mathbf{u}^L \leq \mathbf{u} \leq \mathbf{u}^U \\
   & & & \mathbf{p}^L \leq \mathbf{p} \leq \mathbf{p}^U

opty translates the symbolic definition of :math:`\mathbf{f}` into
:math:`\mathbf{f}_i` and forms the Jacobian of :math:`\mathbf{f}_i` with
respect to :math:`[\mathbf{x},\mathbf{r}_u,\mathbf{p}_u]^T`. These two
numerical functions are highly optimized for computational speed, taking
advantage of pre-compilation common sub expression elimination, efficient
memory usage, and the sparsity of the Jacobian.
