#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle


def plot_identified_state_trajectory(identified, measured, state_syms):
    """
    Parameters
    ----------
    id : shape(n, N)
    me : shape(n, N)
    """
    num_states, num_time_steps = identified.shape
    fig, axes = plt.subplots(num_states, 1, sharex=True)
    for m, i, ax, sym in zip(identified, measured, axes, state_syms):
        ax.plot(m, 'b-')
        ax.plot(i, 'r.')
        ax.legend((str(sym) + ' Measured', str(sym) + ' Identified'))
    plt.show()


def plot_constraints(constraints, n, N, state_syms):
    """Plots the constrain violations for each state."""
    cons = constraints.reshape(n, N - 1).T
    plt.plot(range(2, cons.shape[0] + 2), cons)
    plt.ylabel('Constraint Violation')
    plt.xlabel('Discretization Point')
    plt.legend([str(s) for s in state_syms])
    plt.show()


def plot_sim_results(y, u):

    # Plot the simulation results and animate the pendulum.
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(u)
    axes[0].set_ylabel('Lateral Force [N]')
    axes[1].plot(y[:, 0])
    axes[1].set_ylabel('Cart Displacement [M]')
    axes[2].plot(np.rad2deg(y[:, 1:]))
    axes[2].set_ylabel('Link Angles [Deg]')
    axes[2].set_xlabel('Time [s]')
    plt.tight_layout()

    plt.show()


def animate_pendulum(t, states, length, filename=None):
    """Animates the n-pendulum and optionally saves it to file.

    Parameters
    ----------
    t : ndarray, shape(m)
        Time array.
    states: ndarray, shape(m,p)
        State time history.
    length: float
        The length of the pendulum links.
    filename: string or None, optional
        If true a movie file will be saved of the animation. This may take
        some time.

    """
    # the number of pendulum bobs
    numpoints = states.shape[1] / 2

    # first set up the figure, the axis, and the plot elements we want to
    # animate
    fig = plt.figure()

    # some dimesions
    cart_width = 0.4
    cart_height = 0.2

    # set the limits based on the motion
    xmin = np.around(states[:, 0].min() - cart_width / 2.0, 1)
    xmax = np.around(states[:, 0].max() + cart_width / 2.0, 1)

    # create the axes
    ymin = -length * (numpoints - 1) - 0.1
    ymax = length * (numpoints - 1) + 0.1
    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect='equal')

    # display the current time
    time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes)

    # create a rectangular cart
    rect = Rectangle([states[0, 0] - cart_width / 2.0, -cart_height / 2],
                     cart_width, cart_height,
                     fill=True, color='red', ec='black')
    ax.add_patch(rect)

    # blank line for the pendulum
    line, = ax.plot([], [], lw=2, marker='o', markersize=6)

    # initialization function: plot the background of each frame
    def init():
        time_text.set_text('')
        rect.set_xy((states[0, 0] - cart_width / 2.0,
                     -cart_height / 2.0))
        line.set_data([], [])
        return time_text, rect, line,

    # animation function: update the objects
    def animate(i):
        time_text.set_text('time = {:2.2f}'.format(t[i]))
        rect.set_xy((states[i, 0] - cart_width / 2.0, -cart_height / 2))
        x = np.hstack((states[i, 0], np.zeros((numpoints - 1))))
        y = np.zeros((numpoints))
        for j in np.arange(1, numpoints):
            x[j] = x[j - 1] - length * np.sin(states[i, j])
            y[j] = y[j - 1] + length * np.cos(states[i, j])
        line.set_data(x, y)
        return time_text, rect, line,

    # call the animator function
    anim = animation.FuncAnimation(fig, animate, frames=len(t),
                                   init_func=init,
                                   interval=t[-1] / len(t) * 1000,
                                   blit=False, repeat=False)
    plt.show()

    # save the animation if a filename is given
    if filename is not None:
        anim.save(filename, fps=30, codec='libx264')
