import numpy

from surrogates.utils.gradients import finite_difference


def test_finite_difference():

    coefficients = {"a": 1.0, "b": 2.0, "c": 3.0}

    def function(parameters):
        return sum(coefficients[x] * parameters[x] for x in parameters)

    initial_parameters = {
        "a": numpy.array([1.0]),
        "b": numpy.array([1.0]),
        "c": numpy.array([1.0])
    }

    value, gradients = finite_difference(function, initial_parameters)

    assert numpy.isclose(value, 6.0)

    assert len(gradients) == 3
    assert all(numpy.isclose(gradients[x], coefficients[x]) for x in initial_parameters)
