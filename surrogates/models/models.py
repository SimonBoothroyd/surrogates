import abc

import arviz
import corner
import numpy
import torch
from matplotlib import pyplot


class Model(abc.ABC):
    """A base model which can be evaluated at a set of parameters.
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
    def n_trainable_parameters(self):
        """int: The number of trainable parameters within this model."""
        return len(self._trainable_labels)

    @property
    def trainable_parameter_labels(self):
        """list of str: The friendly names of the parameters which are allowed to vary."""
        return self._trainable_labels

    @property
    def n_total_parameters(self):
        """int: The total number of parameters within this model."""
        return self.n_trainable_parameters + self.n_fixed_parameters

    @property
    def all_parameter_labels(self):
        """list of str: The friendly names of the parameters within this model."""
        return self._trainable_labels + self._fixed_labels

    def __init__(self, trainable_parameter_labels, fixed_parameters):
        """

        Parameters
        ----------
        trainable_parameter_labels: list of str
            The names associated with the trainable parameters of this model.
        fixed_parameters: dict of str and float
            The values of the fixed model parameters, whose keys of the name
            associated with the parameter.
        """
        self._trainable_labels = [*trainable_parameter_labels]

        self._fixed_parameters = []
        self._fixed_labels = []

        for parameter_name in fixed_parameters:

            self._fixed_parameters.append(fixed_parameters[parameter_name])
            self._fixed_labels.append(parameter_name)

    @abc.abstractmethod
    def evaluate(self, parameters):
        """Evaluate the model at the specified (trainable) parameters

        Parameters
        ----------
        parameters: numpy.ndarray
            The parameters to evaluate the model at with
            shape=(..., n_trainable_parameters).

        Returns
        -------
        dict of str and numpy.ndarray
            The values produced by the model.
        dict of str and numpy.ndarray
            The uncertainties in the values.
        """
        raise NotImplementedError()


class BayesianModel(Model, abc.ABC):
    """A model which may be used in Bayesian inference / fitting.
    """

    @property
    def priors(self):
        """list of Distribution: The priors on each trainable parameter of
        this model."""
        return self._priors

    def __init__(self, priors, fixed_parameters):
        """
        Parameters
        ----------
        priors: dict of str and Distribution
            The priors distributions to place on each parameter, whose keys
            are the friendly name of the parameter associated with the prior.
            There should be one entry per trainable parameter.
        fixed_parameters: dict of str and float
            The values of the fixed model parameters, whose keys of the name
            associated with the parameter.
        """
        self._priors = []
        trainable_labels = []

        for parameter_name in priors:

            distribution = priors[parameter_name]
            self._priors.append(distribution)

            if isinstance(parameter_name, tuple):

                trainable_labels.extend(parameter_name)
                assert len(parameter_name) == distribution.n_variables
            else:
                trainable_labels.append(parameter_name)
                assert distribution.n_variables == 1

        super(BayesianModel, self).__init__(trainable_labels, fixed_parameters)

        common_parameters = set(self._fixed_labels).intersection(
            set(self._trainable_labels)
        )

        if len(common_parameters) > 0:

            raise ValueError(
                f"The {', '.join(common_parameters)} have been flagged "
                f"as being both fixed and trainable."
            )

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

        for prior in self._priors:

            initial_parameters[counter : counter + prior.n_variables] = prior.sample()
            counter += prior.n_variables

        return initial_parameters

    def evaluate_log_prior(self, parameters):
        """Evaluates the log value of the prior for a
        set of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters (with shape=n_parameters)
            to evaluate at.

        Returns
        -------
        float
            The sum of the log values of priors evaluated at `parameters`.
        """
        log_prior = 0.0
        counter = 0

        for prior in self._priors:

            log_prior += prior.log_pdf(
                parameters[counter : counter + prior.n_variables]
            )
            counter += prior.n_variables

        return log_prior

    def evaluate_log_likelihood(self, parameters):
        """Evaluates the log value of the this models likelihood for
        a set of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters (with shape=n_parameters)
            to evaluate at.

        Returns
        -------
        float
            The log value of the likelihood evaluated at `parameters`.
        """
        raise NotImplementedError()

    def evaluate_log_posterior(self, parameters):
        """Evaluates the *unnormalized* log posterior for
        a set of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters (with shape=n_parameters)
            to evaluate at.

        Returns
        -------
        float
            The log value of the posterior evaluated at `parameters`.
        """
        return self.evaluate_log_prior(parameters) + self.evaluate_log_likelihood(
            parameters
        )

    def plot_trace(self, trace, show=False):
        """Use `Arviz` to plot a trace of the trainable parameters,
        alongside a histogram of their distribution.

        Parameters
        ----------
        trace: numpy.ndarray
            The parameter trace with shape=(n_steps, n_trainable_parameters+1)
        show: bool
            If true, the plot will be shown.

        Returns
        -------
        matplotlib.pyplot.Figure
            The plotted figure.
        """

        trace_dict = {}

        for index, label in enumerate(self._trainable_labels):
            trace_dict[label] = trace[:, index + 1]

        data = arviz.convert_to_inference_data(trace_dict)

        axes = arviz.plot_trace(data)
        figure = axes[0][0].figure

        if show:
            figure.show()

        return figure

    def plot_corner(self, trace, show=False):
        """Use `corner` to plot a corner plot of the parameter
        distributions.

        Parameters
        ----------
        trace: numpy.ndarray
            The parameter trace with shape=(n_steps, n_trainable_parameters+1)
        show: bool
            If true, the plot will be shown.

        Returns
        -------
        matplotlib.pyplot.Figure
            The plotted figure.
        """

        figure = corner.corner(
            trace[:, 1 : 1 + len(self._trainable_labels)],
            labels=self._trainable_labels,
            color="#17becf",
        )

        if show:
            figure.show()

        return figure

    @staticmethod
    def plot_log_p(log_p, show=False, label="$log p$"):
        """Plot the log p trace.

        Parameters
        ----------
        log_p: numpy.ndarray
            The log p trace with shape=(n_steps, 1)
        show: bool
            If true, the plot will be shown.
        label: str
            The y-axis label to use.

        Returns
        -------
        matplotlib.pyplot.Figure
            The plotted figure.
        """
        figure, axes = pyplot.subplots(1, 1, figsize=(5, 5), dpi=200)

        axes.plot(log_p, color="#17becf")
        axes.set_xlabel("steps")
        axes.set_ylabel(f"{label}")

        if show:
            figure.show()

        return figure

    def plot(self, trace, log_p, show=False):
        """Produce plots of this models traces. This is equivalent to
        calling `plot_trace`, `plot_corner`, `plot_log_p`,
        `plot_percentage_deviations`.

        Parameters
        ----------
        trace: numpy.ndarray
            The parameter trace with shape=(n_steps, n_trainable_parameters+1)
        log_p: numpy.ndarray
            The log p trace with shape=(n_steps, 1)
        show: bool
            If true, the plots will be shown.

        Returns
        -------
        tuple of matplotlib.pyplot.Figure
            The plotted figures.
        """
        return (
            self.plot_trace(trace, show),
            self.plot_corner(trace, show),
            self.plot_log_p(log_p, show),
        )


class TrainableModel(BayesianModel, abc.ABC):
    """A model which can be trained upon previously generated data,
    and then be more rapidly evaluated than generating fresh data.
    """

    def __init__(self, priors, fixed_parameters, condition_parameters, condition_data):
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

        super(TrainableModel, self).__init__(priors, fixed_parameters)

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
            shape=(n_data_points, n_trainable_parameters).
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

        if self.n_trainable_parameters != parameters.shape[1]:

            raise ValueError(
                f"The parameter must be of length {self.n_trainable_parameters}"
            )

        assert all(x in uncertainties for x in values)
        assert all(x in values for x in uncertainties)

        assert all(uncertainties[x].shape == values[x].shape for x in values)

    @abc.abstractmethod
    def add_training_data(self, parameters, values, uncertainties):
        """Trains the model on a new set of data.

        Parameters
        ----------
        parameters: numpy.ndarray
            The parameters used to collect the data with
            shape=(n_data_points, n_trainable_parameters).
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
