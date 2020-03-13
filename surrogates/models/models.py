import abc

import numpy
import torch


class Model(abc.ABC):
    """A model which may be used in Bayesian inference / fitting.
    """

    @property
    def fixed_parameters(self):
        return self._fixed_parameters

    @property
    def n_fixed_parameters(self):
        """int: The number of fixed parameters within this model."""
        return len(self._fixed_labels)

    @property
    def fixed_parameter_labels(self):
        """list of str: The friendly names of the parameters which are fixed."""
        return self._fixed_labels

    @property
    def n_variable_parameters(self):
        """int: The number of variable parameters (both trainable and non-trainable) within
        this model."""
        return len(self._variable_labels)

    @property
    def variable_parameter_labels(self):
        """list of str: The names of the parameters (both trainable and non-trainable) which
        are allowed to vary."""
        return self._variable_labels

    @property
    def n_total_parameters(self):
        """int: The total number of parameters within this model."""
        return self.n_variable_parameters + self.n_fixed_parameters

    @property
    def n_trainable_parameters(self):
        """int: The number of trainable parameters within this model."""
        return len(self._trainable_labels)

    @property
    def trainable_parameter_labels(self):
        """list of str: The names of the parameters which are trainable."""
        return self._trainable_labels

    @property
    def all_parameter_labels(self):
        """list of str: The names of the parameters within this model."""
        return self._variable_labels + self._fixed_labels

    @property
    def priors(self):
        """dict of str and Distribution: The priors on each trainable parameter of
        this model."""
        return self._priors

    def __init__(self, priors, variable_parameters, fixed_parameters):
        """

        Parameters
        ----------
        priors: dict of str and Distribution
            The priors distributions to place on each trainable parameter, whose keys
            are the friendly name of the parameter associated with the prior.
        variable_parameters: list of str
            The names of any variable parameters which are not trainable (i.e
            those parameters which may vary but do not appear in `priors`).
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

        common_parameters = set.intersection(
            {*trainable_parameters}, {*variable_parameters}
        )

        if len(common_parameters) > 0:

            raise ValueError(
                "A parameter cannot appear both in the priors dictionary and the variable "
                "list. A parameter having a prior already implies that it is variable."
            )

        self._priors = priors

        self._variable_labels = [*trainable_parameters, *variable_parameters]
        self._trainable_labels = [*trainable_parameters]

        self._fixed_parameters = []
        self._fixed_labels = []

        for parameter_name in fixed_parameters:

            self._fixed_parameters.append(fixed_parameters[parameter_name])
            self._fixed_labels.append(parameter_name)

        common_parameters = set(self._fixed_labels).intersection(
            set(self._variable_labels)
        )

        if len(common_parameters) > 0:

            raise ValueError(
                f"The {', '.join(common_parameters)} have been flagged "
                f"as being both fixed and variable."
            )

    @abc.abstractmethod
    def evaluate(self, properties, parameters):
        """Evaluate the model at the specified (variable) parameters

        Parameters
        ----------
        properties: list of str
            The properties which this model should evaluate.
        parameters: numpy.ndarray
            The parameters to evaluate the model at with
            shape=(n_sets, n_variable_parameters).

        Returns
        -------
        dict of str and numpy.ndarray
            The values produced by the model, where each array
            as shape=(n_sets,).
        dict of str and numpy.ndarray
            The uncertainties in the values, where each array
            as shape=(n_sets,).
        """
        raise NotImplementedError()

    def sample_priors(self):
        """Generates a set of random parameters from the prior
        distributions. Those parameters without a prior will be
        assigned their fixed values.

        Returns
        -------
        numpy.ndarray:
            The sampled parameters with shape=(`n_trainable_parameters`).
        """

        initial_parameters = numpy.zeros(self.n_trainable_parameters)
        counter = 0

        for prior in self._priors.values():

            initial_parameters[counter : counter + prior.n_variables] = prior.sample()
            counter += prior.n_variables

        return initial_parameters

    def evaluate_log_prior(self, parameters):
        """Evaluates the log value of the prior for a
        set of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters (with shape=n_variable_parameters)
            to evaluate at.

        Returns
        -------
        float
            The sum of the log values of priors evaluated at `parameters`.
        """
        log_prior = 0.0
        counter = 0

        for prior in self._priors.values():

            log_prior += prior.log_pdf(
                parameters[counter : counter + prior.n_variables]
            )
            counter += prior.n_variables

        return log_prior


class SurrogateModel(Model, abc.ABC):
    """A model which can be trained upon previously generated data,
    and then be more rapidly evaluated than generating fresh data.
    """

    def __init__(
        self,
        priors,
        variable_parameters,
        fixed_parameters,
        condition_parameters,
        condition_data,
    ):
        """
        Parameters
        ----------
        condition_parameters: bool
            If true, all training parameters for this model will be shifted to
            have a zero mean, and to fall within the range [-1, 1].
        condition_data: bool
            If true, all training data for this model will be shifted to
            have a zero mean, and to fall within the range [-1, 1]. The
            uncertainties in the training values will also be scaled by the
            same amount as the training values themselves.
        """

        super(SurrogateModel, self).__init__(
            priors, variable_parameters, fixed_parameters
        )

        # Keep a track of the data that this model was trained upon
        self._training_parameters = None
        self._training_values = None
        self._training_uncertainties = None

        self._condition_parameters = condition_parameters
        self._condition_data = condition_data

        self._parameter_scale = None
        self._parameter_shift = None

        self._value_scales = {}
        self._value_shifts = {}

    def _validate_training_data(self, parameters, values, uncertainties):
        """Validate the data to train this model on, checking among
        other things that all dimensions are correct.

        Parameters
        ----------
        parameters: numpy.ndarray
            The parameters used to generate the training data with
            shape=(n_data_points, n_variable_parameters).
        values: dict of str and numpy.ndarray
            The training data, where each value should be an array
            with shape=(n_data_points,).
        uncertainties: dict of str and numpy.ndarray
            The uncertainties in the `values` (assumed to be gaussian) Each array
            has a shape=(n_data_points, 1)).
        """

        # Make sure the parameter / values arrays are the correct shapes.
        assert parameters.ndim == 2
        assert all(x.ndim == 1 for x in values.values())

        assert all(x.shape[0] == parameters.shape[0] for x in values.values())

        if self.n_variable_parameters != parameters.shape[1]:

            raise ValueError(
                f"The parameter must be of length {self.n_variable_parameters}"
            )

        assert all(x in uncertainties for x in values)
        assert all(x in values for x in uncertainties)

        assert all(uncertainties[x].shape == values[x].shape for x in values)

    def _retrain(self):
        """Re-train the models hyperparameters based on the currently
        available training data.
        """
        raise NotImplementedError()

    def add_training_data(self, parameters, values, uncertainties):
        """Trains the model on a new set of data.

        Parameters
        ----------
        parameters: numpy.ndarray
            The parameters used to collect the data with
            shape=(n_data_points, n_variable_parameters).
        values: dict of str and numpy.ndarray
            The data collected using the specified parameters. Each array has
            a shape=(n_data_points, 1)).
        uncertainties: dict of str and numpy.ndarray
            The uncertainties in the `values` (assumed to be gaussian) Each array
            has a shape=(n_data_points, 1)).
        """
        self._validate_training_data(parameters, values, uncertainties)

        parameters = torch.from_numpy(parameters)
        values = {x: torch.from_numpy(y) for x, y in values.items()}
        uncertainties = {x: torch.from_numpy(y) for x, y in uncertainties.items()}

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
            self._training_values = {
                x: y * self._value_scales[x] for x, y in self._training_values.items()
            }
            self._training_values = {
                x: y + self._value_shifts[x] for x, y in self._training_values.items()
            }
            self._training_uncertainties = {
                x: y * self._value_scales[x]
                for x, y in self._training_uncertainties.items()
            }

            self._training_values = {
                x: torch.cat([self._training_values[x], values[x]])
                for x in self._training_values
            }
            self._training_uncertainties = {
                x: torch.cat([self._training_uncertainties[x], uncertainties[x]])
                for x in self._training_uncertainties
            }

        torch_zero = torch.tensor(0.0, dtype=torch.float64)
        torch_one = torch.tensor(1.0, dtype=torch.float64)

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
                torch.isclose(self._parameter_scale, torch_zero),
                torch_one,
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
            self._value_shifts = {
                x: torch.mean(y, axis=0) for x, y in self._training_values.items()
            }
            self._value_scales = {
                x: y.max(axis=0)[0] - y.min(axis=0)[0]
                for x, y in self._training_values.items()
            }
            self._value_scales = {
                x: torch.where(torch.isclose(y, torch_zero), torch_one, y)
                for x, y in self._value_scales.items()
            }

        else:

            self._value_shifts = {
                x: torch.zeros((1,), dtype=torch.float64)
                for x, y in self._training_values.items()
            }
            self._value_scales = {
                x: torch.ones((1,), dtype=torch.float64)
                for x, y in self._training_values.items()
            }

        # Condition the data.
        self._training_parameters = (
            self._training_parameters - self._parameter_shift
        ) / self._parameter_scale

        self._training_values = {
            x: y - self._value_shifts[x] for x, y in self._training_values.items()
        }
        self._training_values = {
            x: y / self._value_scales[x] for x, y in self._training_values.items()
        }
        self._training_uncertainties = {
            x: y / self._value_scales[x]
            for x, y in self._training_uncertainties.items()
        }

        self._retrain()
