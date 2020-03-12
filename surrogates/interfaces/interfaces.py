import abc


class EvaluationInterface(abc.ABC):
    """The base class for objects which evaluate a set of optimization
    targets for a given set of parameters by calling out to another library.

    The main purpose of this class is to provide an abstraction for models
    which will require expensive simulations to be ran at each evaluation,
    enabling them to either call out to a library which will perform real
    simulations, or a 'mock' library which will evaluate some cheaper surrogate
    to enable rapid testing and experimentation.
    """

    @property
    @abc.abstractmethod
    def supported_properties(self):
        """list of str: The properties supported by this interface."""
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate(self, properties, parameters):
        """Evaluates the specified properties at the provided
        parameters.

        Parameters
        ----------
        properties: list of str
            The properties to evaluate.
        parameters: numpy.ndarray
            The parameters to evaluate at with shape(n_sets, n_parameters)
            where n_sets is the number of parameter sets to evaluate at and
            n_parameters is the number of parameters in each set (this typically
            corresponds to the total number of model parameters).

        Returns
        -------
        dict of str and numpy.ndarray
            The evaluated values with shape=(n_sets,)
        dict of str and numpy.ndarray
            The uncertainties in the evaluated values with shape=(n_sets,).
        """
        raise NotImplementedError()
