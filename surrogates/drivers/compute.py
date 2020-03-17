from typing import Dict, List, Tuple

import numpy

from surrogates.drivers import Driver
from surrogates.drivers.targets import PropertyTarget
from surrogates.models import SurrogateModel


class SurrogateDriver(Driver):
    """The base class for drivers which will evaluate targets by evaluating
    trained surrogate models.

    This driver will be responsible for generating new training data to ensure
    the surrogate model can be accurately evaluated at the parameters of
    interest, either through running simulations or reweighting cached simulation
    data.
    """

    def __init__(
        self,
        surrogate_models: Dict[str, SurrogateModel],
        simulation_driver: Driver,
        reweighting_driver: Driver,
    ):
        """
        Parameters
        ----------
        surrogate_models: dict of str and SurrogateModel
            The surrogate models which will be evaluated / trained
            for each type of target (defined by the `property_type`).
        simulation_driver: Driver
            The driver which will be used to generate new simulation
            data to train the model on.
        reweighting_driver: Driver
            The driver which will be used to reweight cached simulation
            data to cheaply generate new data to train the model on.
        """
        assert all(isinstance(x, SurrogateModel) for x in surrogate_models)

        assert isinstance(simulation_driver, Driver)
        assert isinstance(reweighting_driver, Driver)

        self._surrogate_models = surrogate_models

        self._simulation_driver = simulation_driver
        self._reweighting_driver = reweighting_driver

    def _train_surrogates(
        self, targets: List[PropertyTarget], parameters: Dict[str, numpy.ndarray]
    ):
        """Generate new data and retrain the surrogate model
        so that it is able to accurately be evaluated at the
        specified parameters.

        Parameters
        ----------
        targets: list of PropertyTarget
            The targets to train surrogates for.
        parameters: dict of str and numpy.ndarray
            The parameters which the surrogate should be accurately
            evaluable at (*not* the parameters to train the model at).
        """

        if len(targets) == 0:
            return

        raise NotImplementedError()

    def evaluate(
        self, targets: List[PropertyTarget], parameters: Dict[str, numpy.ndarray]
    ) -> Tuple[List[numpy.ndarray], List[numpy.ndarray]]:
        """
        Parameters
        ----------
        targets: list of  PropertyTarget
        parameters: dict of str and numpy.ndarray
        """

        assert all(isinstance(x, PropertyTarget) for x in targets)
        assert all(x.property_type in self._surrogate_models for x in targets)

        values = []
        uncertainties = []

        targets_to_retrain = [
            x
            for x in targets
            if not self._surrogate_models[x.property_type].can_evaluate(parameters)
        ]

        self._train_surrogates(targets_to_retrain, parameters)

        for target in targets:

            full_parameters = parameters.copy()
            full_parameters.update(target.parameters)

            value, uncertainty = self._surrogate_models[target.property_type].evaluate(
                parameters
            )

            values.append(value)
            uncertainties.append(uncertainty)

        return values, uncertainties
