import numpy

from surrogates.utils.gradients import finite_difference


def test_finite_difference():

    coefficients = numpy.array([1.0, 2.0, 3.0])

    def function(parameters):
        return numpy.dot(parameters, coefficients)

    initial_parameters = numpy.ones(3)
    gradients = finite_difference(function, initial_parameters)

    assert len(gradients) == 3
    assert numpy.allclose(gradients, coefficients)
