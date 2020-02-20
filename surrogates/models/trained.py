import numpy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from surrogates.models import TrainableModel
from surrogates.utils.gradients import finite_difference


class GaussianProcessModel(TrainableModel):
    """A model which evaluates a trained Gaussian Process based on a radial-basis
    function kernel. The Gaussian Process may be retrained with extra data on the
    fly.
    """

    def __init__(
        self,
        initial_variance=1.0,
        minimum_variance=1.0e-3,
        maximum_variance=1.0e3,
        initial_length_scale=1.0,
        minimum_length_scale=1.0e-3,
        maximum_length_scale=1.0e3,
    ):

        self._temperature = None
        self._parameter_dimension = None

        self._variance_inputs = (initial_variance, (minimum_variance, maximum_variance))

        self._length_scale_inputs = (
            initial_length_scale,
            (minimum_length_scale, maximum_length_scale),
        )

        self._training_parameters = numpy.zeros(0)
        self._training_observables = [numpy.zeros(0), numpy.zeros(0), numpy.zeros(0)]

        self._gaussian_processes = []

    def add_training_point(
        self, parameter, temperature, liquid_density, vapor_pressure, surface_tension
    ):

        if self._temperature is None:
            self._temperature = temperature

        if temperature != self._temperature:
            raise NotImplementedError()

        if self._parameter_dimension is None:
            self._parameter_dimension = len(parameter)

        if self._parameter_dimension != len(parameter):

            raise ValueError(
                f"The parameter must be of length {self._parameter_dimension}"
            )

        training_parameters = [*self._training_parameters, parameter]

        training_observables = [
            [*self._training_observables[0], liquid_density],
            [*self._training_observables[1], vapor_pressure],
            [*self._training_observables[2], surface_tension],
        ]

        self._training_parameters = numpy.array(training_parameters)

        self._training_observables = [
            numpy.array(training_observables[0]),
            numpy.array(training_observables[1]),
            numpy.array(training_observables[2]),
        ]

        self._retrain()

    def _retrain(self):
        """

        Notes
        -----
        This function is based in the version found in a scikit-learn tutorial:
        https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html

        Returns
        -------

        """
        self._gaussian_processes = []

        for observables in self._training_observables:

            # Instantiate a Gaussian Process model
            kernel = ConstantKernel(*self._variance_inputs) * RBF(
                *self._length_scale_inputs
            )
            gaussian_process = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=9, normalize_y=True
            )

            # Fit to data using Maximum Likelihood Estimation of the parameters
            gaussian_process.fit(self._training_parameters, observables)

            self._gaussian_processes.append(gaussian_process)

    def evaluate(self, parameters, temperatures, calculate_gradients=False):

        if len(self._gaussian_processes) == 0:
            raise ValueError("The model has not yet been trained upon any data.")

        if self._temperature not in temperatures:
            raise NotImplementedError()

        liquid_density, liquid_density_std = self._gaussian_processes[0].predict(
            parameters.reshape(1, -1), return_std=True
        )
        vapor_pressure, vapor_pressure_std = self._gaussian_processes[1].predict(
            parameters.reshape(1, -1), return_std=True
        )
        surface_tension, surface_tension_std = self._gaussian_processes[2].predict(
            parameters.reshape(1, -1), return_std=True
        )

        values = {
            "liquid_density": liquid_density.reshape(-1, 1),
            "vapor_pressure": vapor_pressure.reshape(-1, 1),
            "surface_tension": surface_tension.reshape(-1, 1),
        }

        uncertainties = {
            "liquid_density": liquid_density_std.reshape(-1, 1),
            "vapor_pressure": vapor_pressure_std.reshape(-1, 1),
            "surface_tension": surface_tension_std.reshape(-1, 1),
        }

        gradients = None

        if calculate_gradients:

            # TODO: In future this should be replaced with the actual derivatives of
            #       the Gaussian Process.
            gradients = {
                "liquid_density": finite_difference(
                    self._gaussian_processes[0].predict, parameters.reshape(1, -1)
                ).reshape(1, -1),
                "vapor_pressure": finite_difference(
                    self._gaussian_processes[1].predict, parameters.reshape(1, -1)
                ).reshape(1, -1),
                "surface_tension": finite_difference(
                    self._gaussian_processes[2].predict, parameters.reshape(1, -1)
                ).reshape(1, -1),
            }

        return values, uncertainties, gradients
