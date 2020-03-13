import abc

import numpy

from surrogates.utils import distributions


class Likelihood(abc.ABC):
    """A base class for a likelihood function which is conditioned upon
    a set of reference data"""

    @abc.abstractmethod
    def evaluate_log_p(self, parameters):
        """Evaluates the likelihood for a given parameter and
        data set.

        Parameters
        ----------
        parameters: numpy.ndarray
            The parameters to evaluate the log p at with
            shape=(1, n_variable_parameters).

        Returns
        -------
        numpy.ndarray:
            The evaluated likelihoods with shape=(n_sets,).
        """
        raise NotImplementedError()


class GaussianLikelihood(abc.ABC):
    """A Gaussian likelihood function which is conditioned upon
    a set of data."""

    def __init__(self, model, property_type, values, uncertainties, parameters=None):
        """
        Parameters
        ----------
        model: Model
            The model to evaluate.
        property_type: str
            The type of property this likelihood will evaluate.
        values: dict of str and numpy.ndarray
            The values to condition the likelihood on with shape=(n_data_points,).
        uncertainties: dict of str and numpy.ndarray
            The uncertainties in the values with shape=(n_data_points,)
        parameters: dict of str and numpy.ndarray, optional
            The parameters which the values were collected at which also correspond to
            model parameters. These may include things such as the temperature or pressure
            at which a set of measurements were made at, with shape=(n_data_points,).

            Each key must correspond to a variable model parameter which is not a trainable
            parameter.
        """

        if parameters is None:
            parameters = {}

        assert len(values) == len(uncertainties)
        assert all(len(x) == len(values) for x in parameters.values())

        non_trainable_labels = {*model.variable_parameter_labels} - {
            *model.trainable_parameter_labels
        }
        assert non_trainable_labels == {*parameters}

        self._model = model
        self._property_type = [property_type]

        self._values = values.reshape(-1, 1)
        self._uncertainties = uncertainties.reshape(-1, 1)

        self._parameters = numpy.zeros((len(values), model.n_variable_parameters))

        for parameter_label, parameter in parameters.items():

            parameter_index = self._model.variable_parameter_labels.index(
                parameter_label
            )

            self._parameters[:, parameter_index] = parameter

    def evaluate_log_p(self, parameters):

        if parameters.ndim == 1:
            parameters = parameters.reshape(-1, 1)

        assert parameters.shape == (1, self._model.n_trainable_parameters)

        self._parameters[:, : self._model.n_trainable_parameters] = parameters[:]

        evaluated_values, _ = self._model.evaluate(
            self._property_type, self._parameters
        )
        evaluated_values = evaluated_values[self._property_type[0]]

        if any(numpy.isnan(evaluated_values)) or any(numpy.isinf(evaluated_values)):
            return -numpy.inf

        # Compute likelihood based on gaussian penalty function
        log_p = numpy.sum(
            distributions.Normal(evaluated_values, self._uncertainties).log_pdf(
                self._values
            )
        )

        return log_p
