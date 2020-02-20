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

    def __init__(self, model, reference_data_set, target_properties=None):
        """
        Parameters
        ----------
        model: Model
            The model to optimize.
        reference_data_set: DataSet
            The data set to train against.
        target_properties: list of str, optional
            The properties to fit against. If `None`, a default list of
            `['liquid_density', 'vapor_pressure', 'surface_tension']` will
            be used.
        """

        self._model = model

        self._temperatures = None

        self._data_scales = None
        self._data_shifts = None

        self._reference_data = None

        self._parameter_trace = None

        all_properties = ["liquid_density", "vapor_pressure", "surface_tension"]

        if target_properties is None:
            target_properties = [*all_properties]

        assert len(target_properties) > 0
        assert all(x in all_properties for x in target_properties)

        self._initialize_reference_data(reference_data_set, target_properties)

        # Construct an analytical Stoll-Werth surrogate model. This will
        # serve as a surrogate stand-in for the openff-evaluator, being
        # used to generate 'simulation results' rapidly.
        self._analytical_model = StollWerthSurrogate(
            reference_data_set.molecular_weight, reference_data_set.bond_length / 10.0
        )

    def _initialize_reference_data(self, reference_data_set, target_properties):
        """Extracts and normalizes data from a target data set

        Parameters
        ----------
        reference_data_set: DataSet
            The data set being trained against.
        target_properties: list of str, optional
            The properties to fit against. If `None`, a default list of
            `['liquid_density', 'vapor_pressure', 'surface_tension']` will
            be used.
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
        self._data_scales = {
            "liquid_density": numpy.max(reference_data_set.liquid_densities[:, 1]),
            "vapor_pressure": numpy.max(reference_data_set.vapor_pressures[:, 1]),
            "surface_tension": numpy.max(reference_data_set.surface_tensions[:, 1]),
        }

        self._data_shifts = {
            "liquid_density": numpy.mean(reference_data_set.liquid_densities[:, 1]),
            "vapor_pressure": numpy.mean(reference_data_set.vapor_pressures[:, 1]),
            "surface_tension": numpy.mean(reference_data_set.surface_tensions[:, 1]),
        }

        # Normalize the target experimental data
        self._reference_data = {
            "liquid_density": reference_data_set.liquid_densities.copy(),
            "vapor_pressure": reference_data_set.vapor_pressures.copy(),
            "surface_tension": reference_data_set.surface_tensions.copy(),
        }

        for label, reference_data in self._reference_data.items():

            # Normalize the observed values.
            reference_data[:, 1] -= self._data_shifts[label]
            reference_data[:, 1] /= self._data_scales[label]

            # Normalize the uncertainties.
            reference_data[:, 2] /= self._data_scales[label]

        # Retain only the properties of interest.
        self._data_scales = {
            x: y for x, y in self._data_scales.items() if x in target_properties
        }
        self._data_shifts = {
            x: y for x, y in self._data_shifts.items() if x in target_properties
        }
        self._reference_data = {
            x: y for x, y in self._reference_data.items() if x in target_properties
        }

    def _simulate_properties(self, parameters, temperatures, calculate_gradients):
        """A black-box function which 'runs' simulations using the model
        parameter at the specified temperature to calculate the liquid
        density, vapor pressure and surface tension.

        Parameters
        ----------
        parameters: numpy.ndarray
            The parameters to simulate at.
        temperatures: numpy.ndarray
            The temperatures to simulate at.
        calculate_gradients: bool
            Whether or not to evaluate the gradients of each
            value with respect to the parameters.

        Returns
        -------
        dict of str and numpy.ndarray
            The values of the properties evaluated by this model using the
            specified parameters. Each array has a shape=(n_temperatures, 1)).
        dict of str and numpy.ndarray
            The uncertainties in the values of the properties evaluated by this model
            using the specified parameters. Each array has a shape=(n_temperatures, 1)).
        dict of str and numpy.ndarray, optional
            The gradients of each value with respect to each of the properties evaluated by
            this model. Each array has a shape=(n_temperatures, n_parameters)). This output
            will be `None` if `calculate_gradients == False`.
        """
        return self._analytical_model.evaluate(
            parameters, temperatures, calculate_gradients
        )

    def _evaluate_model(self, parameters, temperatures, calculate_gradients):
        """Evaluates the model at a specified set of parameters and
        temperature. If the surrogate model cannot be evaluated with
        sufficient accuracy, a new 'simulation' (or set of simulations)
        is run and the model retrained.

        Parameters
        ----------
        parameters: numpy.ndarray
            The parameters to simulate at.
        temperatures: numpy.ndarray
            The temperatures to simulate at.
        calculate_gradients: bool
            Whether or not to evaluate the gradients of each
            value with respect to the parameters.

        Returns
        -------
        dict of str and numpy.ndarray
            The values of the properties evaluated by this model using the
            specified parameters. Each array has a shape=(n_temperatures, 1)).
        dict of str and numpy.ndarray
            The uncertainties in the values of the properties evaluated by this model
            using the specified parameters. Each array has a shape=(n_temperatures, 1)).
        dict of str and numpy.ndarray, optional
            The gradients of each value with respect to each of the properties evaluated by
            this model. Each array has a shape=(n_temperatures, n_parameters)). This output
            will be `None` if `calculate_gradients == False`.
        """

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
        values, uncertainties, gradients = self._simulate_properties(
            parameters, temperatures, calculate_gradients
        )
        # else:
        #     observations = [*self._model.evaluate(parameters, temperatures)]

        return values, uncertainties, gradients

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

        assert len(initial_parameters) == 2

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
