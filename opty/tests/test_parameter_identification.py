#!/usr/bin/env python

import numpy as np

from ..parameter_identification import (objective_function,
                                        objective_function_gradient)


def test_objective_function():

    M = 5
    o = 2
    n = 2 * o
    q = 3
    h = 0.01

    time = np.linspace(0.0, (M - 1) * h, num=M)

    y_measured = np.random.random((M, o))  # measured coordinates

    x_model = np.hstack((y_measured, np.random.random((M, o))))

    free = np.hstack((x_model.T.flatten(), np.random.random(q)))

    cost = objective_function(free, M, n, h, time, y_measured)

    np.testing.assert_allclose(cost, 0.0, atol=1e-15)


def test_objective_function_gradient():

    M = 5
    o = 2
    n = 2 * o
    q = 3
    h = 0.01

    time = np.linspace(0.0, (M - 1) * h, num=M)
    y_measured = np.random.random((M, o))  # measured coordinates
    x_model = np.random.random((M, n))
    free = np.hstack((x_model.T.flatten(), np.random.random(q)))

    cost = objective_function(free, M, n, h, time, y_measured)
    grad = objective_function_gradient(free, M, n, h, time, y_measured)

    expected_grad = np.zeros_like(free)
    delta = 1e-8
    for i in range(len(free)):
        free_copy = free.copy()
        free_copy[i] = free_copy[i] + delta
        perturbed = objective_function(free_copy, M, n, h, time, y_measured)
        expected_grad[i] = (perturbed - cost) / delta

    np.testing.assert_allclose(grad, expected_grad, atol=1e-8)
