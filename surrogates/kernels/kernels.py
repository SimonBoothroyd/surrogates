import abc
import logging

import arviz
import numpy
from tqdm import tqdm

from surrogates.models.analytical import StollWerthSurrogate

logger = logging.getLogger(__name__)


class BaseKernel(abc.ABC):
    """The base class for any object which aims to fit a set of
    model parameters against a target data set.
    """

    @property
    def parameter_trace(self):
        return self._parameter_trace

    def __init__(self, model, reference_data_set):
        """
        Parameters
        ----------
        model: Model
            The model to optimize.
        reference_data_set: DataSet
            The data set to train against.
        """

        self._model = model

        self._temperatures = None

        self._data_scales = None
        self._data_shifts = None

        self._reference_data = None

        self._parameter_trace = None

        self._initialize_reference_data(reference_data_set)

        # Construct an analytical Stoll-Werth surrogate model. This will
        # serve as a surrogate stand-in for the openff-evaluator, being
        # used to generate 'simulation results' rapidly.
        self._analytical_model = StollWerthSurrogate(
            reference_data_set.molecular_weight
        )

    def _initialize_reference_data(self, reference_data_set):
        """Extracts and normalizes data from a target data set

        Parameters
        ----------
        reference_data_set: DataSet
            The data set being trained against.
        """

        if (
            reference_data_set.liquid_densities.shape[0] != 1
            or reference_data_set.vapor_pressures.shape[0] != 1
            or reference_data_set.surface_tensions.shape[0] != 1
        ):

            raise ValueError(
                "Currently only data measured at a single temperature is supported."
            )

        self._temperatures = numpy.array([reference_data_set.liquid_densities[0, 0]])

        if not (
            numpy.isclose(reference_data_set.vapor_pressures[0, 0], self._temperatures)
            or numpy.isclose(
                reference_data_set.surface_tensions[0, 0], self._temperatures
            )
        ):

            raise ValueError(
                "Currently only data measured at a single temperature is supported."
            )

        # Define a set of normalizing transforms so that our data is roughly
        # in the range of [-1.0, 1.0]
        self._data_scales = numpy.array(
            [
                numpy.max(reference_data_set.liquid_densities[:, 1]),
                numpy.max(reference_data_set.vapor_pressures[:, 1]),
                numpy.max(reference_data_set.surface_tensions[:, 1]),
            ]
        )

        self._data_shifts = numpy.array(
            [
                numpy.mean(reference_data_set.liquid_densities[:, 1]),
                numpy.mean(reference_data_set.vapor_pressures[:, 1]),
                numpy.mean(reference_data_set.surface_tensions[:, 1]),
            ]
        )

        # Normalize the target experimental data
        self._reference_data = [
            reference_data_set.liquid_densities.copy(),
            reference_data_set.vapor_pressures.copy(),
            reference_data_set.surface_tensions.copy(),
        ]

        for index, reference_data in enumerate(self._reference_data):
            # Normalize the observed values.
            reference_data[:, 1] -= self._data_shifts[index]
            reference_data[:, 1] /= self._data_scales[index]

            # Normalize the uncertainties.
            reference_data[:, 2] /= self._data_scales[index]

    def _simulate_properties(self, parameters, temperatures):
        """A black-box function which 'runs' simulations using the model
        parameter at the specified temperature to calculate the liquid
        density, vapor pressure and surface tension.

        Parameters
        ----------
        parameters: numpy.ndarray
            The parameters to simulate at.
        temperatures: numpy.ndarray
            The temperatures to simulate at.

        Returns
        -------
        numpy.ndarray
            The values of the 'simulated' liquid density evaluated at each temperature
            and using the specified parameters (shape=(n_temperatures, 1)).
        numpy.ndarray
            The values of the 'simulated' vapor pressure evaluated at each temperature
            and using the specified parameters (shape=(n_temperatures, 1)).
        numpy.ndarray
            The values of the 'simulated' surface tension evaluated at each temperature
            and using the specified parameters (shape=(n_temperatures, 1)).
        """
        return self._analytical_model.evaluate(parameters, temperatures)

    def _evaluate_model(self, parameters, temperatures):

        # # TODO: Once uncertainties are added perform check whether to re-simulate
        # #       and retrain the GP.
        # if retrain_model:
        #
        #     (
        #         liquid_density,
        #         vapor_pressure,
        #         surface_tensions,
        #     ) = self._simulate_properties(parameters, temperatures)
        #     self._model.add_training_point(
        #         parameters,
        #         temperatures,
        #         liquid_density[0, 0],
        #         vapor_pressure[0, 0],
        #         surface_tensions[0, 0],
        #     )

        # Evaluate the trained model.
        # if use_simulation:
        observations = [*self._simulate_properties(parameters, temperatures)]
        # else:
        #     observations = [*self._model.evaluate(parameters, temperatures)]

        return observations

    @abc.abstractmethod
    def _step(self, current_iteration, current_parameters):
        """Propogate the kernel forward by a single step.

        Parameters
        ----------
        current_iteration: int
            The index of the current iteration.
        current_parameters: numpy.ndarray
            The current values of the parameters.

        Returns
        -------
        numpy.ndarray
            The updated parameters.
        """
        raise NotImplementedError()

    def run(self, initial_parameters, iterations):
        """Run the optimization, starting from an initial set of parameters.

        Parameters
        ----------
        initial_parameters: numpy.ndarray
            The parameters to start the optimization from.
        iterations: int
            The number of iterations to perform.
        """

        assert len(initial_parameters) == 4

        current_parameters = initial_parameters.copy()

        # Set up an array to monitor the parameter traces.
        self._parameter_trace = numpy.zeros((iterations, len(current_parameters)))

        for current_iteration in tqdm(range(iterations)):

            # Modify the parameters
            current_parameters = self._step(current_iteration, current_parameters)
            self._parameter_trace[current_iteration] = current_parameters

        return current_parameters

    def plot_traces(self):

        parameter_labels = ["epsilon", "sigma"]
        trace_dict = {}

        for index, label in enumerate(parameter_labels):
            trace_dict[label] = self._parameter_trace[:, index]

        parameter_trace = arviz.convert_to_inference_data(trace_dict)

        axes = arviz.plot_trace(parameter_trace)
        figure = axes[0][0].figure
        figure.show()