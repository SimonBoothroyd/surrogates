import abc


class DriverTarget(abc.ABC):
    """A particular target that a driver should attempt
    to evaluate.
    """

    @property
    def parameters(self):
        """dict of str and numpy.ndarray: The non-model parameters which the target
        will be evaluated at with shape=(n_parameters,)."""
        return self._parameters

    def __init__(self, parameters):
        """

        Parameters
        ----------
        parameters: dict of str and numpy.ndarray
            The non-model parameters which the target will be evaluated at
            with shape=(n_parameters,).
        """
        self._parameters = parameters


class Driver(abc.ABC):
    """The base class for objects which evaluate a set of optimization
    targets for a given set of parameters by calling out to another library.

    The main purpose of this class is to provide an abstraction for models
    which will require expensive simulations to be ran at each evaluation,
    enabling them to either call out to a library which will perform real
    simulations, or a 'mock' library which will evaluate some cheaper surrogate
    to enable rapid testing and experimentation.
    """

    @abc.abstractmethod
    def evaluate(self, targets, parameters):
        """Evaluates the specified properties at the provided
        parameters.

        Parameters
        ----------
        targets: list of DriverTarget
            The targets to evaluate.
        parameters: dict of str and numpy.ndarray
            The model parameters to evaluate at with shape(n_sets,)
            where n_sets is the number of parameter sets to evaluate at.

        Returns
        -------
        numpy.ndarray
            The evaluated values with shape=(n_sets,)
        numpy.ndarray
            The uncertainties in the evaluated values with shape=(n_sets,).
        """
        raise NotImplementedError()
