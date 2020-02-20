import logging

import numpy
from matplotlib import pyplot

from surrogates.kernels import BaseKernel

logger = logging.getLogger(__name__)


class GradientDescent(BaseKernel):
    """A simple kernel for optimizing a model using the gradient
    descent algorithm.
    """

    @property
    def cost_function_trace(self):
        return self._cost_function_trace

    def __init__(self, model, reference_data_set, learning_rates):
        """
        Parameters
        ----------
        model: Model
            The model to optimize.
        reference_data_set: DataSet
            The data set to train against.
        learning_rates: numpy.ndarray
            The rate at which each parameter is allowed to change.
        """

        super(GradientDescent, self).__init__(model, reference_data_set)

        self._cost_function_trace = None
        self._learning_rates = learning_rates

    def _evaluate_cost_function(self, parameters, temperatures):
        """Evaluate the cost function for a set of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The parameters to evaluate the cost function at.
        temperatures: numpy.ndarray
            The temperatures to evaluate the cost function at.

        Returns
        -------
        float
            The value of the cost function.
        numpy.ndarray
            The gradient of the cost function with respect to
            each parameter.
        """

        values, _, gradients = self._evaluate_model(parameters, temperatures, True)

        # Normalize the observed data and gradients.
        cost_function = 0.0
        cost_gradient = numpy.zeros(parameters.shape)

        total_observables = 0.0

        for label in values:

            values[label] -= self._data_shifts[label]
            values[label] /= self._data_scales[label]

            gradients[label] /= self._data_scales[label]

            differences = values[label] - self._reference_data[label][:, 1]

            cost_function += numpy.sum(differences ** 2)
            cost_gradient -= numpy.sum(differences * gradients[label], axis=0)

            total_observables += len(values[label])

        cost_function /= 2.0 * total_observables
        cost_gradient /= total_observables

        return cost_function, cost_gradient

    def _step(self, current_iteration, current_parameters):

        # Evaluate the cost function
        cost_function, cost_gradient = self._evaluate_cost_function(
            current_parameters, temperatures=self._temperatures
        )

        self._cost_function_trace[current_iteration] = cost_function

        # Modify the parameters
        current_parameters += cost_gradient * self._learning_rates
        return current_parameters

    def run(self, initial_parameters, iterations):

        assert initial_parameters.shape == self._learning_rates.shape

        # Initialize an array to store the cost function.
        self._cost_function_trace = numpy.zeros((iterations, 1))
        # Run the optimization.
        super(GradientDescent, self).run(initial_parameters, iterations)

    def plot_traces(self):

        super(GradientDescent, self).plot_traces()

        figure, axes = pyplot.subplots(1, 1, figsize=(5, 5), dpi=200)
        axes.plot(self._cost_function_trace, color="#17becf")
        axes.set_xlabel("steps")
        figure.show()
