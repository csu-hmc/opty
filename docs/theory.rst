======
Theory
======

Given at a set of :math:`M` first order continuous differential equations in
time in implicit form:

.. math::

   \mathbf{f}(\dot{\mathbf{y}}(t), \mathbf{y}(t), \mathbf{u}(t), \mathbf{p}, t ) = \mathbf{0}
   \in \mathbb{R}^M

where:

- :math:`t` is time
- :math:`\mathbf{y}(t) \in \mathbb{R}^n` the state vector at time
  :math:`t`
- :math:`\mathbf{u}(t) \in \mathbb{R}^m` is the vector of specified
  (exongenous) inputs at time :math:`t`
- :math:`\mathbf{p} \in \mathbb{R}^p` is the vector of constant parameters

The equations :math:`\mathbf{f}` may contain ordinary differential equations,
differential algebraic equations, additional path constraints (algebraic or
differential) and be augmented with specific trajectory values at instances of
time.

From here on out, the notation :math:`(t)` will be dropped for convenience.

In opty, you would define these equations as a SymPy column matrix that
contains SymPy expressions. For example, a single degree of freedom compound
pendulum is described by the first order ordinary differential equations:

.. math::

   \mathbf{f}\left(\begin{bmatrix}\dot{\theta} \\ \dot{\omega}\end{bmatrix},
   \begin{bmatrix}\theta \\ \omega\end{bmatrix},
   \begin{bmatrix}T\end{bmatrix},
   \begin{bmatrix}I \\ m \\ g \\ l \end{bmatrix}\right)
   =
   \begin{bmatrix}
   \dot{\theta} - \omega \\
   I \dot{\omega} + mgl\sin\theta - T
   \end{bmatrix}
   = \mathbf{0}

In SymPy, this would look like:

.. code:: pycon

   >>> I, m, g, l, t = symbols('I, m, g, l, t')
   >>> theta, omega, T = symbols('theta, omega, T', cls=Function)
   >>> f = Matrix([theta(t).diff() - omega(t),
   ...             I*omega(t).diff() + m*g*l*sin(theta(t)) - T(t)])
   >>> f
   ⎡        -ω + θ̇        ⎤
   ⎢                      ⎥
   ⎣I⋅ω̇ + g⋅l⋅m⋅sin(θ) - T⎦

One can then break up :math:`\mathbf{u}` and :math:`\mathbf{p}` into known,
:math:`k`, and unknown, :math:`u`, quantities.

.. math::

   \mathbf{u} = \left[ \mathbf{u}_k \quad \mathbf{u}_u \right]^T

   \mathbf{p} = \left[ \mathbf{p}_k \quad \mathbf{p}_u \right]^T

where the dimension of the unknown vectors are:

- :math:`\mathbf{u}_u \in \mathbb{R}^q`
- :math:`\mathbf{p}_u \in \mathbb{R}^r`

Then there are optimal state trajectories :math:`\mathbf{y}`, optimal unknown
input trajectories :math:`\mathbf{u}_u` and optimal unknown parameter values
:math:`\mathbf{p}_u` if some cost function, :math:`J(\mathbf{y}, \mathbf{u}_u,
\mathbf{p}_u)` is specified. For example, the integral of a cost rate :math:`L`
with respect to time:

.. math::

   J(\mathbf{y}, \mathbf{u}_u, \mathbf{p}_u) =
   \int L(\mathbf{y}(t), \mathbf{u}(t), \mathbf{p}) dt

Additionally, upper :math:`U` and lower :math:`L` boundary constraints on these
variables can be specified as follows:

.. math::

   \mathbf{y}^L \leq \mathbf{y} \leq \mathbf{y}^U \\
   \mathbf{u}^L \leq \mathbf{u} \leq \mathbf{u}^U \\
   \mathbf{p}^L \leq \mathbf{p} \leq \mathbf{p}^U

which are the maximal bounds on the trajectories or parameter values.


This type of problem is called an `optimal control`_ problem with the objective
of finding the open loop input trajectories and/or the optimal system
parameters such that the dynamical system evolves in time in a way that
minimizes the objective and enforces the constraints. This problem can be
rewritten as a `nonlinear programming`_ (NLP) problem using `direct
collocation`_ transcription methods and solved using a variety of optimization
algorithms suitable for NLP problems [Betts2010]_.

.. _optimal control: https://en.wikipedia.org/wiki/Optimal_control
.. _nonlinear programming: https://en.wikipedia.org/wiki/Nonlinear_programming
.. _direct collocation: https://en.wikipedia.org/wiki/Trajectory_optimization#Direct_collocation

Direct collocation transcribes the continuous differential equations into
discretized difference equations using a variety of discrete integration
methods. These difference equations are then treated as constraints and
appended to any algebraic constraints. These new constraints ensure that the
system's dynamics are satisfied at each discretized time instance and relieves
the need to sequentially integrate the differential equations as one does in
shooting optimization.

opty currently supports two first order integration methods: the `backward
Euler method`_ and the `midpoint method`_.

.. _backward Euler method: https://en.wikipedia.org/wiki/Backward_Euler_method
.. _midpoint method: https://en.wikipedia.org/wiki/Midpoint_method

If :math:`h` is the time interval between the :math:`N` discretized instances
of time :math:`t_i`, one can use the backward Euler method of integration to
approximate the derivative of the state vector as:

.. math::

   \frac{d\mathbf{y}}{dt} & \approx & \frac{\mathbf{y}_i - \mathbf{y}_{i-1}}{h} \\
   \mathbf{y}(t_i) & = & \mathbf{y}_i \\
   \mathbf{u}(t_i) & = & \mathbf{u}_i

Using the midpoint method the approximation of the derivative of the state
vector is:

.. math::

   \frac{d\mathbf{y}}{dt} & \approx & \frac{\mathbf{y}_{i+1} - \mathbf{y}_{i}}{h} \\
   \mathbf{y}(t_i) & = & \frac{\mathbf{y}_i + \mathbf{y}_{i+1}}{2} \\
   \mathbf{u}(t_i) & = & \frac{\mathbf{u}_i + \mathbf{u}_{i+1}}{2}

The discretized differential equation :math:`\mathbf{f}_i` can the be written
using both of the above approximations.

For the backward Euler method:

.. math::

   \mathbf{f}_i = \mathbf{f}\left(\frac{\mathbf{y}_i - \mathbf{y}_{i-1}}{h},
                                  \mathbf{y}_i, \mathbf{u}_i, \mathbf{p}, t_i\right) = 0

For the midpoint method:

.. math::

   \mathbf{f}_i = \mathbf{f}\left(\frac{\mathbf{y}_{i+1} - \mathbf{y}_{i}}{h},
                                  \frac{\mathbf{y}_i + \mathbf{y}_{i+1}}{2},
                                  \frac{\mathbf{u}_i + \mathbf{u}_{i+1}}{2},
                                  \mathbf{p}, t_i\right) = \mathbf{0}

Then, defining :math:`\mathbf{x}_i` to be:

.. math::

   \mathbf{x}_i = [\mathbf{y}_i \quad \mathbf{u}_{ui} \quad \mathbf{p}_{u}]^T

The above equations will create :math:`M(N-1)` constraint equations and the
optimization problem can formally be written as:

.. math::

   & \underset{\mathbf{x}_i \in \mathbb{R}^{n + q + r}}
              {\text{min}}
   & & J(\mathbf{x}_i) \\
   & \text{s.t.}
   & & \mathbf{f}_i = \mathbf{0} \in \mathbb{R}^{M(N-1)} \\
   & & & \mathbf{x}_i^L \leq \mathbf{x}_i \leq \mathbf{x}_i^U

opty translates the symbolic definition of :math:`\mathbf{f}` into
:math:`\mathbf{f}_i` and forms the highly sparse Jacobian of
:math:`\frac{\partial\mathbf{f}_i}{\partial\mathbf{x}_i}` with respect to
:math:`\mathbf{x}_i`. These two numerical functions are optimized for
computational speed, taking advantage of pre-compilation common sub expression
elimination, efficient memory usage, and the sparsity of the Jacobian. This is
especially advantageous if :math:`\mathbf{f}` is very complex. The cost
function :math:`J` and it's gradient :math:`\frac{\partial J}{\partial
\mathbf{x}_i}` must be specified by Python functions that return a scalar and
vector, respectively.
