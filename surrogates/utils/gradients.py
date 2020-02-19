import numpy


def finite_difference(function, parameters, pertubation_fraction=1.001):
    """A function which calculates the forward finite difference
    gradient of a function with respect to each of its parameters.

    The function should be of the form:

    >>> def some_function(parameters):
    >>>    ...

    where parameters is a 1-D numpy array.

    Parameters
    ----------
    function: function
        The function to compute the gradient of.
    parameters: numpy.ndarray
        The parameters to compute the gradient at with shape=(n_parameters).
    pertubation_fraction: float
        The fraction amount to perturb parameters by when calculating
        the finite differences.

    Returns
    -------
    numpy.ndarray
        The gradient with respect to each parameter with shape=(n_parameters).
    """

    value = function(parameters)
    gradients = numpy.zeros(parameters.shape)

    for index in range(len(parameters)):

        if numpy.isclose(parameters[index], 0.0):
            continue

        perturbed_parameters = parameters.copy()
        perturbed_parameters[index] *= pertubation_fraction

        perturbed_value = function(perturbed_parameters)
        gradients[index] = (perturbed_value - value) / (
            perturbed_parameters[index] - parameters[index]
        )

    return gradients
