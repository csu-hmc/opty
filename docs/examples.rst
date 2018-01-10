========
Examples
========

Single Pendulum Swing Up
========================

.. plot:: ../examples/pendulum_swing_up.py
   :include-source:

.. raw:: html

   <video controls>
     <source src="_static/pendulum_swing_up.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

Betts 2003
==========

.. plot:: ../examples/betts2003.py
   :include-source:

Pendulum Parameter Identification
=================================

Identifies a single constant from measured pendulum swing data. The default
initial guess for trajectories are the known continuous solution plus
artificial Gaussian noise and a random positive value for the parameter.

.. plot:: ../examples/vyasarayani2011.py
   :include-source:
