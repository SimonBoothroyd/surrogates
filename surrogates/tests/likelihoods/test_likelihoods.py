import numpy

from surrogates.likelihoods.likelihoods import GaussianLikelihood


def test_gaussian_likelihood():

    values = numpy.array([[1.0]])
    uncertainties = numpy.array([[numpy.sqrt(0.5)]])

    likelihood = GaussianLikelihood(values, uncertainties, None)

    value, _ = likelihood.evaluate(values, uncertainties, {})
    assert numpy.isclose(value, -numpy.log(numpy.sqrt(2 * numpy.pi)))
