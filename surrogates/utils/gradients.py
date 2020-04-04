from typing import Callable, Dict, Tuple

import numpy


def finite_difference(
    function: Callable,
    parameters: Dict[str, numpy.ndarray],
    perturbation_fraction: float = 1.001,
) -> Tuple[numpy.ndarray, Dict[str, numpy.ndarray]]:
    """A function which calculates the forward finite difference
    gradient of a function with respect to each of its parameters.

    The function should be of the form:

    >>> def some_function(parameters: Dict[str, numpy.ndarray]) -> numpy.ndarray:
    >>>    ...

    where parameters is a 1-D numpy array.

    Parameters
    ----------
    function: function
        The function to compute the gradient of.
    parameters: dict of str and numpy.ndarray
        The parameters to compute the gradient with respect to.
    perturbation_fraction: float
        The fraction amount to perturb parameters by when calculating
        the finite differences.

    Returns
    -------
    numpy.ndarray
        The value of the function at the specified parameters.
    dict of str and numpy.ndarray
        The gradient with respect to each parameter.
    """

    value = function(parameters)
    gradients = {}

    for label, parameter in parameters.items():

        perturbed_parameters = parameters.copy()
        perturbed_parameters[label] = parameters[label] * perturbation_fraction

        if numpy.isclose(perturbed_parameters[label], 0.0):
            perturbed_parameters[label] += 0.001

        perturbed_value = function(perturbed_parameters)
        gradients[label] = (perturbed_value - value) / (
            perturbed_parameters[label] - parameters[label]
        )

    return value, gradients
