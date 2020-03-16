import numpy

from surrogates.likelihoods.likelihoods import GaussianLikelihood


def test_gaussian_likelihood():

    values = numpy.array([[1.0]])
    uncertainties = numpy.array([[1.0]])

    likelihood = GaussianLikelihood(values, uncertainties, None)

    assert numpy.isclose(
        likelihood.evaluate(values, uncertainties), -numpy.log(numpy.sqrt(2 * numpy.pi))
    )
