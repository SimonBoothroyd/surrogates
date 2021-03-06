import copy
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy

from surrogates.drivers import Driver
from surrogates.drivers.targets import PropertyTarget
from surrogates.models import SurrogateModel


class SurrogateDriverSnapshot:
    """Stores a snapshot of the current state of a driver, namely the
    parameters it was evaluated at and the simulation and reweighted
    training points generated during the snapshot.
    """

    def __init__(
        self, current_evaluation: int, current_parameters: Dict[str, numpy.ndarray],
    ):

        self.current_evaluation = current_evaluation
        self.current_parameters = copy.deepcopy(current_parameters)

        self.simulation_training_parameters: Dict[str, Dict[str, numpy.ndarray]] = {}
        self.simulation_training_values: Dict[str, numpy.ndarray] = {}
        self.simulation_training_stds: Dict[str, numpy.ndarray] = {}

        self.reweighted_training_parameters: Dict[str, Dict[str, numpy.ndarray]] = {}
        self.reweighted_training_values: Dict[str, numpy.ndarray] = {}
        self.reweighted_training_stds: Dict[str, numpy.ndarray] = {}


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
        output_directory: str = "surrogate_snapshots",
        output_frequency: int = 100,
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
        output_directory: str
            The directory to store any output files in.
        output_frequency: int
            The frequency (in number of evaluations) with which to store
            the output.
        """
        assert all(isinstance(x, SurrogateModel) for x in surrogate_models.values())

        assert isinstance(simulation_driver, Driver)
        assert isinstance(reweighting_driver, Driver)

        self._surrogate_models = surrogate_models

        self._simulation_driver = simulation_driver
        self._reweighting_driver = reweighting_driver

        self._output_directory = output_directory
        os.makedirs(self._output_directory, exist_ok=True)

        self._current_evaluations = 0
        self._output_frequency = output_frequency

    def _reweight_training_data(
        self,
        target: PropertyTarget,
        surrogate_model: SurrogateModel,
        parameters: Dict[str, numpy.ndarray],
    ) -> Tuple[
        Dict[str, List[numpy.ndarray]], List[numpy.ndarray], List[numpy.ndarray]
    ]:
        """Generate new training data by reweighting existing simulation
        data.

        Parameters
        ----------
        target: PropertyTarget
            The target to generate data for.
        surrogate_model: SurrogateModel
            The model the data will be used to train.
        parameters: dict of str and numpy.ndarray
            The parameters to reweight out from.
        """
        surrogate_parameters = {*surrogate_model.parameters}
        model_parameters = {*parameters}

        parameters_to_perturb = model_parameters.intersection(surrogate_parameters)

        assert len(parameters_to_perturb) > 0

        initial_scale = 0.0005
        maximum_scale = 0.005

        reweighted_parameters = {x: [] for x in surrogate_model.parameters}

        reweighted_values = []
        reweighted_stds = []

        for parameter_label in parameters_to_perturb:

            for direction in [-1.0, 1.0]:

                perturbed_parameters = parameters.copy()
                parameter_scale = initial_scale

                reweighted_parameter = None
                reweighted_value = None
                reweighted_uncertainty = None

                while True:

                    # Increase the parameter until either the model NaNs or
                    # we have increased the parameter by 10% its initial value.
                    scale_amount = 1.0 + parameter_scale * direction

                    perturbed_parameters[parameter_label] = (
                        parameters[parameter_label] * scale_amount
                    )

                    values, uncertainties, _ = self._reweighting_driver.evaluate(
                        [target], perturbed_parameters, False
                    )

                    if (
                        any(numpy.isnan(x) for x in values)
                        or parameter_scale > maximum_scale
                    ):
                        break

                    reweighted_parameter = perturbed_parameters
                    reweighted_value = values
                    reweighted_uncertainty = uncertainties

                    parameter_scale += 0.0005

                if reweighted_parameter is None:
                    continue

                n_target_parameters = len(reweighted_value)

                for label in parameters_to_perturb:

                    reweighted_parameters[label].extend(
                        [perturbed_parameters[label]] * n_target_parameters
                    )

                for label in target.parameters:
                    reweighted_parameters[label].extend([target.parameters[label]])

                reweighted_values.extend(reweighted_value)
                reweighted_stds.extend(reweighted_uncertainty)

        return reweighted_parameters, reweighted_values, reweighted_stds

    def _train_surrogates(
        self,
        targets: List[PropertyTarget],
        parameters: Dict[str, numpy.ndarray],
        snapshot: SurrogateDriverSnapshot,
    ) -> bool:
        """Generate new data and retrain the surrogate model
        so that it is able to accurately be evaluated at the
        specified parameters.

        Parameters
        ----------
        targets: list of PropertyTarget
            The targets to train surrogates for.
        parameters: dict of str and numpy.ndarray
            The parameters which the surrogate should be accurately
            evaluable at (*not* necessarily the parameters to train
            the model at).
        snapshot: SurrogateDriverSnapshot
            The current evaluation snapshot to register the new training
            data with.
        """

        if len(targets) == 0:
            return True

        # Simulate at the points that the model stopped
        simulated_values, simulated_stds, _ = self._simulation_driver.evaluate(
            targets, parameters, False
        )

        if any(numpy.isnan(simulated_values)) or any(numpy.isnan(simulated_stds)):
            return False

        for target, simulated_value, simulated_std in zip(
            targets, simulated_values, simulated_stds
        ):

            surrogate_model = self._surrogate_models[target.property_type]

            # Add the simulation training data to the snapshot
            snapshot.simulation_training_parameters[target.property_type] = {
                x: y for x, y in parameters.items() if x in surrogate_model.parameters
            }
            snapshot.simulation_training_values[target.property_type] = simulated_value
            snapshot.simulation_training_stds[target.property_type] = simulated_std

            # Reweight the data.
            (
                training_parameters,
                training_values,
                training_stds,
            ) = self._reweight_training_data(target, surrogate_model, parameters)

            # Add the reweighted data to the snapshot.
            snapshot.reweighted_training_parameters[target.property_type] = {
                x: numpy.concatenate(y)
                for x, y in training_parameters.items()
                if x in surrogate_model.parameters
            }
            snapshot.reweighted_training_values[
                target.property_type
            ] = numpy.concatenate(training_values)
            snapshot.reweighted_training_stds[target.property_type] = numpy.concatenate(
                training_stds
            )

            n_target_parameters = len(simulated_value)

            for label in training_parameters:

                if label not in parameters:
                    continue

                training_parameters[label].extend(
                    [parameters[label]] * n_target_parameters
                )

            for label in target.parameters:
                training_parameters[label].extend([target.parameters[label]])

            training_parameters = {
                x: numpy.concatenate(y) for x, y in training_parameters.items()
            }
            training_parameters = {
                x: y.reshape(-1, 1) if y.ndim == 1 else y
                for x, y in training_parameters.items()
            }

            training_values.append(simulated_value)
            training_stds.append(simulated_std)

            training_values = numpy.concatenate(training_values)
            training_stds = numpy.concatenate(training_stds)

            if any(numpy.isnan(training_values)) or any(numpy.isnan(training_stds)):
                raise ValueError("NaN")

            surrogate_model.add_training_data(
                training_parameters, training_values, training_stds
            )

        return True

    def evaluate(
        self,
        targets: List[PropertyTarget],
        parameters: Dict[str, numpy.ndarray],
        compute_gradients: bool,
    ) -> Tuple[
        List[numpy.ndarray],
        List[numpy.ndarray],
        Optional[List[Dict[str, numpy.ndarray]]],
    ]:

        assert all(isinstance(x, PropertyTarget) for x in targets)
        assert all(x.property_type in self._surrogate_models for x in targets)

        targets_to_retrain = []
        target_parameters = {}

        # Build the arrays containing all of the model and target parameters.
        for target in targets:

            full_parameters = parameters.copy()
            full_parameters.update(target.parameters)

            full_parameters = {
                x: y
                for x, y in full_parameters.items()
                if x in self._surrogate_models[target.property_type].parameters
            }

            if not self._surrogate_models[target.property_type].can_evaluate(
                full_parameters
            ):
                targets_to_retrain.append(target)

            target_parameters[target.property_type] = full_parameters

        # Create a snapshot of the current evaluation if we have met the
        # specified frequency or if we are retraining the model.
        self._current_evaluations += 1
        evaluation_snapshot = None

        if (
            self._current_evaluations % self._output_frequency == 0
            or len(targets_to_retrain) > 0
        ):

            evaluation_snapshot = SurrogateDriverSnapshot(
                current_evaluation=self._current_evaluations,
                current_parameters=parameters,
            )

        values = []
        uncertainties = []
        gradients = []

        if len(targets_to_retrain) > 0:

            print("Retraining model")

            could_train = self._train_surrogates(
                targets_to_retrain, parameters, evaluation_snapshot
            )

            if not could_train:

                values = [numpy.array([numpy.nan])] * len(targets)
                uncertainties = [numpy.array([numpy.nan])] * len(targets)
                gradients = None

                return values, uncertainties, gradients

        for target in targets:

            value, uncertainty, gradient = self._surrogate_models[
                target.property_type
            ].evaluate(target_parameters[target.property_type])

            uncertainty = numpy.nan_to_num(uncertainty)

            values.append(value)
            uncertainties.append(uncertainty)

            gradients.append(gradient)

        # Save the snapshot to disk
        if evaluation_snapshot is not None:

            with open(
                os.path.join(self._output_directory, str(self._current_evaluations)),
                "wb",
            ) as file:

                pickle.dump(evaluation_snapshot, file)

        return values, uncertainties, gradients if compute_gradients else None
