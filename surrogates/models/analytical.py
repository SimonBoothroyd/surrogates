import numpy

from surrogates.interfaces.surrogate import StollWerthInterface
from surrogates.models import Model


class StollWerthModel(Model):
    """A surrogate model for the two-center Lennard-Jones model, which can
    be rapidly evaluated from the models critical density and temperature,
    liquid and vapor density, saturation pressure and surface tension.
    """

    def __init__(
        self, fixed_parameters, molecular_weight, file_path=None,
    ):

        self._interface = StollWerthInterface(0.0, molecular_weight, file_path)

        required_parameters = {"epsilon", "sigma", "L", "Q", "temperature"}
        provided_parameters = {*fixed_parameters.keys()}

        assert all(x in required_parameters for x in provided_parameters)

        trainable_parameters = ["epsilon", "sigma", "L", "Q", "temperature"]

        for provided_parameter in provided_parameters:
            trainable_parameters.remove(provided_parameter)

        super(StollWerthModel, self).__init__(trainable_parameters, fixed_parameters)

    def evaluate(self, properties, parameters):

        assert parameters.ndim == 2
        assert parameters.shape[1] == self.n_trainable_parameters

        all_parameters = {
            x: numpy.array([y] * parameters.shape[0])
            for x, y in zip(self._fixed_labels, self._fixed_parameters)
        }
        all_parameters.update(
            {x: y for x, y in zip(self._trainable_labels, parameters.T)}
        )

        all_parameters = numpy.concatenate(
            (
                all_parameters["epsilon"].reshape(-1, 1),
                all_parameters["sigma"].reshape(-1, 1),
                all_parameters["L"].reshape(-1, 1),
                all_parameters["Q"].reshape(-1, 1),
                all_parameters["temperature"].reshape(-1, 1),
            ),
            axis=1,
        )

        return self._interface.evaluate(properties, all_parameters)
