======
Theory
======

Given at a set of continous differential equations in implicit form:

.. math::

   \mathbf{f}(\dot{\mathbf{x}}(t), \mathbf{x}(t), \mathbf{r}(t), \mathbf{p}, t) = \mathbf{0}

where:

- :math:`\mathbf{x}` is the state vector
- :math:`\mathbf{r}` is the vector of specified inputs
- :math:`\mathbf{p}` is the vector of constant parameters
- :math:`t` is time

one can then break up :math:`\mathbf{r}` and :math:`\mathbf{p}` into known and
unknown quantities:

.. math::

   \mathbf{r} = \left[ \mathbf{r}_k \quad \mathbf{r}_u \right]

   \mathbf{r} = \left[ \mathbf{p}_k \quad \mathbf{p}_u \right]

Then there are optimal trajectories :math:`\mathbf{r}_u` and optimal values of
:math:`\mathbf{p}_u` if some cost function, :math:`J(\mathbf{r}_u, \mathbf{p})`
is specified. In addition one may specify some constraints on the trajectories
or parameters in the form of:

.. math::

    \mathbf{c}^L \leq \mathbf{c} \leq \mathbf{c}^U \\

Constraints can be:

- path constraints
- bounds on the trajectories or parameter values
- specific trajectory or parameter values at instances of time

This is typically called an optimal control problem where you are seeking the
optimal open loop input trajectories and/or the optimal system parameters such
that the dynamical system does a task in an optimal way. This problem can be
rewritten as a non-linear programming problem using direct collocation methods.
In addition to the above constraints the differential equations can be
discretized using a variety of integration methods and appended as additional
optimality constraints. This ensures that the dynamics are abided by at each
discretized time instance. Using the backward Euler method of integration, the
approximation of the derivative of the states is:

.. math::

   \frac{d\mathbf{x}}{dt} & \approx & \frac{\mathbf{x}_i - \mathbf{x}_{i-1}}{h} \\
   \mathbf{x}(t) & \approx & \mathbf{x}_i \\
   \mathbf{u}(t) & \approx & \mathbf{u}_i

and using the midpoint formual:

.. math::

   \frac{d\mathbf{x}}{dt} \approx \frac{\mathbf{x}_{i+1} - \mathbf{x}_{i}}{h}

   \mathbf{x}(t) \approx \frac{\mathbf{x}_i + \mathbf{x}_{i+1}}{2}

   \mathbf{u}(t) \approx \frac{\mathbf{u}_i + \mathbf{u}_{i+1}}{2}

Using the Euler approximation :math:`f` can be discretized to become:

.. math::

   \mathbf{f}_i = f\left(\frac{\mathbf{x}_i - \mathbf{x}_{i-1}}{h},
                         \mathbf{x}_i, \mathbf{u}_i, \mathbf{p}, t_i\right) = 0

This creates :math:`m` sets of :math:`n` constraints.

.. math::

   & \underset{\mathbf{x}, \mathbf{u}_u, \mathbf{p}_u}{\text{min}}
   & & J(\mathbf{x}, \mathbf{u}_u, \mathbf{p}_u) \\
   & \text{s.t.}
   & & \mathbf{f}^L \leq \mathbf{f}_i(x) \leq \mathbf{f}^U \\
   & & & \mathbf{c}^L \leq \mathbf{c} \leq \mathbf{c}^U \\
   & & & \mathbf{x}^L \leq \mathbf{x} \leq \mathbf{x}^U \\
   & & & \mathbf{u}^L \leq \mathbf{u} \leq \mathbf{u}^U \\
   & & & \mathbf{p}^L \leq \mathbf{p} \leq \mathbf{p}^U
