import abc
from typing import Dict, List, Tuple

import numpy
import torch
from numpy import ndarray
from scipy.spatial.qhull import Delaunay

from surrogates.drivers import Driver
from surrogates.likelihoods.likelihoods import Likelihood
from surrogates.utils.distributions import Distribution
from surrogates.utils.numpy import parameter_dict_to_array


class BayesianModel(abc.ABC):
    """A model which may be used in Bayesian inference / fitting.
    """

    @property
    def fixed_parameters(self) -> Dict[str, float]:
        return self._fixed_parameters

    @property
    def n_fixed_parameters(self) -> int:
        """int: The number of fixed parameters within this model."""
        return len(self._fixed_parameters)

    @property
    def trainable_parameters(self) -> List[str]:
        """list of str: The names of the parameters which are trainable."""
        return self._trainable_labels

    @property
    def n_trainable_parameters(self) -> int:
        """int: The number of trainable parameters within this model."""
        return len(self._trainable_labels)

    @property
    def n_total_parameters(self) -> int:
        """int: The total number of parameters within this model."""
        return self.n_trainable_parameters + self.n_fixed_parameters

    @property
    def priors(self) -> Dict[str, Distribution]:
        """dict of str and Distribution: The priors on each trainable parameter of
        this model."""
        return self._priors

    @property
    def likelihoods(self) -> List[Likelihood]:
        """list of Likelihood: The different likelihoods which this model
        is conditioned upon."""
        return self._likelihoods

    def __init__(
        self,
        priors: Dict[str, Distribution],
        likelihoods: List[Likelihood],
        driver: Driver,
        fixed_parameters: Dict[str, float],
    ):
        """

        Parameters
        ----------
        priors: dict of str and Distribution
            The priors distributions to place on each trainable parameter, whose keys
            are the friendly name of the parameter associated with the prior.
        likelihoods: list of Likelihood
            The likelihoods to condition this model upon.
        driver: Driver
            The driver to use to evaluate the likelihoods.
        fixed_parameters: dict of str and float
            The values of the fixed model parameters, whose keys of the name
            associated with the parameter.
        """

        # Validate the priors and extract the names of the trainable parameters.
        trainable_parameters = []

        for parameter_name in priors:

            distribution = priors[parameter_name]

            if isinstance(parameter_name, tuple):

                trainable_parameters.extend(parameter_name)
                assert len(parameter_name) == distribution.n_variables
            else:

                trainable_parameters.append(parameter_name)
                assert distribution.n_variables == 1

        self._priors = {
            x if isinstance(x, tuple) else tuple([x]): y for x, y in priors.items()
        }

        self._likelihoods = likelihoods
        self._driver = driver

        self._trainable_labels = [*trainable_parameters]

        self._fixed_parameters = {
            x: numpy.array([[y]]) for x, y in fixed_parameters.items()
        }

        common_parameters = set(self._fixed_parameters).intersection(
            set(self._trainable_labels)
        )

        if len(common_parameters) > 0:

            raise ValueError(
                f"The {', '.join(common_parameters)} have been flagged "
                f"as being both fixed and trainable."
            )

    def evaluate_log_prior(self, parameters: Dict[str, ndarray]) -> numpy.ndarray:
        """Evaluates the log value of the prior for a
        set of parameters.

        Parameters
        ----------
        parameters: dict of str and numpy.ndarray
            The values of the parameters to evaluate at with
            shape=(n_sets, n_variable_parameters).

        Returns
        -------
        numpy.ndarray
            The sum of the log values of priors evaluated at `parameters`
            with shape=(n_sets,).
        """
        log_prior = numpy.zeros(len(next(iter(parameters.values()))))

        for labels, prior in self._priors.items():

            parameter = numpy.concatenate([parameters[label] for label in labels])
            log_prior += numpy.ravel(prior.log_pdf(parameter))

        return log_prior

    def evaluate_log_likelihood(self, parameters: Dict[str, ndarray]) -> numpy.ndarray:
        """Evaluates the log value of the likelihood for a
        set of parameters.

        Parameters
        ----------
        parameters: dict of str and numpy.ndarray
            The values of the parameters to evaluate at with
            shape=(n_sets, n_variable_parameters).

        Returns
        -------
        numpy.ndarray
            The sum of the log values of likelihood evaluated at `parameters`
            with shape=(n_sets,).
        """
        all_parameters = {}
        all_parameters.update(parameters)
        all_parameters.update(self._fixed_parameters)

        log_prior = numpy.zeros(len(next(iter(all_parameters.values()))))

        targets = [x.driver_target for x in self._likelihoods]

        evaluated_values, evaluated_uncertainties = self._driver.evaluate(
            targets, all_parameters
        )

        for likelihood, values, uncertainties in zip(
            self._likelihoods, evaluated_values, evaluated_uncertainties
        ):
            log_prior += likelihood.evaluate(values, uncertainties)

        return log_prior

    def evaluate(self, parameters: Dict[str, ndarray]) -> numpy.ndarray:
        """Evaluate the log posterior of this model at the specified
        sets of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The parameters to evaluate the model at with
            shape=(n_sets, n_trainable_parameters).

        Returns
        -------
        numpy.ndarray
            The evaluated log p with shape=(n_sets,).
        """

        return self.evaluate_log_prior(parameters) + self.evaluate_log_likelihood(
            parameters
        )


class SurrogateModel(abc.ABC):
    """A model which can be trained upon previously generated data,
    and then be more rapidly evaluated than generating fresh data.
    """

    @property
    def parameters(self) -> List[str]:
        """list of str: The names of the parameters that this model will be trained
        upon / can be evaluated using."""
        return self._parameter_labels

    @property
    def n_parameters(self) -> int:
        """The number the parameters that this model will be trained upon /
        can be evaluated using."""
        return len(self._parameter_labels)

    @property
    def convex_hull(self) -> Delaunay:
        """scipy.spatial.qhull: A convex hull which is wrapped around the parameters
        which were used to train the model."""
        return self._convex_hull

    def __init__(
        self,
        parameter_labels: List[str],
        condition_parameters: bool,
        condition_data: bool,
    ):
        """
        Parameters
        ----------
        parameter_labels: list of str
            The names of the parameters that this model will be trained upon /
            can be evaluated using.
        condition_parameters: bool
            If true, all training parameters for this model will be shifted to
            have a zero mean, and to fall within the range [-1, 1].
        condition_data: bool
            If true, all training data for this model will be shifted to
            have a zero mean, and to fall within the range [-1, 1]. The
            uncertainties in the training values will also be scaled by the
            same amount as the training values themselves.
        """

        self._parameter_labels = parameter_labels

        # Keep a track of the data that this model was trained upon
        self._training_parameters = None
        self._training_values = None
        self._training_uncertainties = None

        self._condition_parameters = condition_parameters
        self._condition_data = condition_data

        self._parameter_scale = None
        self._parameter_shift = None

        self._value_scale = None
        self._value_shift = None

        # Define some useful torch constants
        self._zero = torch.tensor(0.0, dtype=torch.float64)
        self._one = torch.tensor(1.0, dtype=torch.float64)

        # Define the hull we will use to check whether the parameters
        # to evaluate lie within the models region of confidence.
        self._convex_hull = None

    def _parameter_dict_to_tensor(
        self, parameters: Dict[str, numpy.ndarray]
    ) -> torch.Tensor:
        """Convert a dictionary of numpy arrays to a single
        pytorch tensor (with the parameter ordering dictated by
        the ordering of the models parameter labels.

        Parameters
        ----------
        parameters: dict of str and numpy.ndarray
            The parameter dictionary to convert.

        Returns
        -------
        torch.Tensor
            The converted parameters.
        """

        array_parameters = parameter_dict_to_array(parameters, self._parameter_labels)
        return torch.from_numpy(array_parameters)

    def _validate_training_data(
        self,
        parameters: Dict[str, numpy.ndarray],
        values: numpy.ndarray,
        uncertainties: numpy.ndarray,
    ):
        """Validate the data to train this model on, checking among
        other things that all dimensions are correct, and converting
        any `numpy` arrays to `pytorch.Tensor` objects.

        Parameters
        ----------
        parameters: dict of str and numpy.ndarray
            The parameters used to generate the training data with
            shape=(n_data_points,).
        values: numpy.ndarray
            The training data with shape=(n_data_points,).
        uncertainties: numpy.ndarray
            The uncertainties in the `values` (assumed to be gaussian) with
            shape=(n_data_points,).

        Returns
        -------
        torch.Tensor
            The validated parameters with shape=(n_data_points, n_parameters).
        torch.Tensor
            The training data with shape=(n_data_points,).
        torch.Tensor
            The uncertainties in the `values` (assumed to be gaussian) Each array
            has a shape=(n_data_points, 1)).
        """

        # Make sure the parameter / values arrays are the correct shapes.
        parameters = self._parameter_dict_to_tensor(parameters)

        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if uncertainties.ndim == 1:
            uncertainties = uncertainties.reshape(-1, 1)

        assert values.ndim == 2
        assert values.shape[1] == 1
        assert len(values) == len(parameters)

        assert uncertainties.shape == values.shape

        values = torch.from_numpy(values)
        uncertainties = torch.from_numpy(uncertainties)

        return parameters, values, uncertainties

    def _retrain(self):
        """Re-train the models hyperparameters based on the currently
        available training data.
        """
        raise NotImplementedError()

    def _rebuild_hull(self):

        if self._training_parameters.shape[0] < self._training_parameters.shape[1] + 2:
            return

        self._convex_hull = Delaunay(self._training_parameters.numpy())

    def add_training_data(
        self,
        parameters: Dict[str, numpy.ndarray],
        values: numpy.ndarray,
        uncertainties: numpy.ndarray,
    ):
        """Trains the model on a new set of data.

        Parameters
        ----------
        parameters: dict of str and numpy.ndarray
            The parameters used to collect the data with
            shape=(n_data_points, 1).
        values: numpy.ndarray
            The data collected using the specified parameters. Each array has
            a shape=(n_data_points, 1)).
        uncertainties: numpy.ndarray
            The uncertainties in the `values` (assumed to be gaussian) Each array
            has a shape=(n_data_points, 1)).
        """

        (parameters, values, uncertainties) = self._validate_training_data(
            parameters, values, uncertainties
        )

        # Add the extra data the existing set.
        if self._training_parameters is None:
            self._training_parameters = parameters
        else:

            # Make sure to un-condition the existing parameters.
            self._training_parameters = (
                self._training_parameters * self._parameter_scale
                + self._parameter_shift
            )

            self._training_parameters = torch.cat(
                [self._training_parameters, parameters]
            )

        if self._training_values is None:

            self._training_values = values
            self._training_uncertainties = uncertainties

        else:

            # First make sure to un-condition the existing data.
            self._training_values = (
                self._training_values * self._value_scale + self._value_shift
            )
            self._training_uncertainties = (
                self._training_uncertainties * self._value_scale
            )

            self._training_values = torch.cat([self._training_values, values])
            self._training_uncertainties = torch.cat(
                [self._training_uncertainties, uncertainties]
            )

        # Determine any conditioning factors.
        if self._condition_parameters:

            # noinspection PyArgumentList
            self._parameter_shift = torch.mean(self._training_parameters, axis=0)
            # noinspection PyArgumentList
            self._parameter_scale = (
                self._training_parameters.max(axis=0)[0]
                - self._training_parameters.min(axis=0)[0]
            )
            self._parameter_scale = torch.where(
                torch.isclose(self._parameter_scale, self._zero),
                self._one,
                self._parameter_scale,
            )

        else:

            self._parameter_shift = torch.zeros(
                (1, parameters.shape[1]), dtype=torch.float64
            )
            self._parameter_scale = torch.ones(
                (1, parameters.shape[1]), dtype=torch.float64
            )

        if self._condition_data:

            # noinspection PyArgumentList
            self._value_shift = torch.mean(self._training_values, axis=0)
            # noinspection PyArgumentList
            self._value_scale = (
                self._training_values.max(axis=0)[0]
                - self._training_values.min(axis=0)[0]
            )
            self._value_scale = torch.where(
                torch.isclose(self._value_scale, self._zero),
                self._one,
                self._value_scale,
            )

        else:

            self._value_shift = torch.zeros((1,), dtype=torch.float64)
            self._value_scale = torch.ones((1,), dtype=torch.float64)

        # Condition the data.
        self._training_parameters = (
            self._training_parameters - self._parameter_shift
        ) / self._parameter_scale

        self._training_values = (
            self._training_values - self._value_shift
        ) / self._value_scale
        self._training_uncertainties = self._training_uncertainties / self._value_scale

        self._rebuild_hull()
        self._retrain()

    def can_evaluate(self, parameters: Dict[str, numpy.ndarray]) -> bool:
        """Checks whether this model has been trained upon sufficient
        data close to the parameters of interest to be able to be
        accurately evaluated.

        Parameters
        ----------
        parameters: dict of str and numpy.ndarray
            The parameters to evaluate the model at where
            each array has shape=(n_data_points).

        Returns
        -------
        bool
        """

        if self._convex_hull is None:
            return False

        parameters = self._parameter_dict_to_tensor(parameters).numpy()
        return self._convex_hull.find_simplex(parameters) >= 0

    @abc.abstractmethod
    def evaluate(
        self, parameters: Dict[str, numpy.ndarray]
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Evaluate the model at the specified set of parameters.

        Parameters
        ----------
        parameters: dict of str and numpy.ndarray
            The parameters to evaluate the model at where
            each array has shape=(n_data_points).

        Returns
        -------
        numpy.ndarray
            The evaluated model values with shape=(n_data_points,)
        numpy.ndarray
            The uncertainty (assumed to be Gaussian) in each evaluated value
            with shape=(n_data_points,)
        """
        raise NotImplementedError()
