import abc

import numpy

from surrogates.utils import distributions


class Likelihood(abc.ABC):
    """A base class for a likelihood function which is conditioned upon
    a set of reference data"""

    @property
    def driver_target(self):
        """DriverTarget: The target which describes how the driver should
        evaluate this likelihood."""
        return self._driver_target

    def __init__(self, values, uncertainties, driver_target):
        """
        Parameters
        ----------
        values: numpy.ndarray
            The values to condition the likelihood on with
            shape=(n_data_points,).
        uncertainties: numpy.ndarray
            The uncertainties in the values with
            shape=(n_data_points,)
        driver_target: DriverTarget
            The target which describes how the driver should
            evaluate this likelihood.
        """

        assert len(values) == len(uncertainties)

        self._values = values.reshape(-1, 1)
        self._uncertainties = uncertainties.reshape(-1, 1)

        self._driver_target = driver_target

    @abc.abstractmethod
    def evaluate(self, values, uncertainties):
        """Evaluates the likelihood for an evaluated set of values.

        Parameters
        ----------
        values: numpy.ndarray
            The values evaluated by the model.
        uncertainties: numpy.ndarray
            The uncertainties in the evaluated values.

        Returns
        -------
        numpy.ndarray:
            The evaluated likelihoods with shape=(n_sets,).
        """
        raise NotImplementedError()


class GaussianLikelihood(Likelihood):
    """A Gaussian likelihood function which is conditioned upon
    a set of data."""

    def evaluate(self, values, uncertainties):

        if any(numpy.isnan(values)) or any(numpy.isinf(values)):
            return -numpy.inf

        # Compute likelihood based on gaussian penalty function
        log_p = numpy.sum(
            distributions.Normal(values, self._uncertainties).log_pdf(self._values)
        )

        return log_p
