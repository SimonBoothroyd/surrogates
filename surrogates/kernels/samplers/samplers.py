import abc

from surrogates.models import BayesianModel


class Sampler(abc.ABC):
    """A base class for different in-model parameter samplers."""

    @property
    def log_p_function(self):
        """function: The log p function to sample over.
        """
        return self._log_p_function

    @log_p_function.setter
    def log_p_function(self, value):
        self._log_p_function = value

    @property
    def proposed_moves(self):
        """numpy.ndarray: The number of moves this sampler has
        proposed for each parameter with shape=(n_trainable_parameters,).
        """
        return self._proposed_moves

    @property
    def accepted_moves(self):
        """numpy.ndarray: The number of moves this sampler has
        accepted for each parameter with shape=(n_trainable_parameters,).
        """
        return self._accepted_moves

    def __init__(self, log_p_function, model):
        """Initializes self.

        Parameters
        ----------
        log_p_function: function, optional
            The log probability function to sample.
        model: BayesianModel
            The model whose parameters are being sampled.
        """

        assert isinstance(model, BayesianModel)

        self._log_p_function = None
        self._gradient_function = None

        self.log_p_function = log_p_function

        self._model = model

        self._proposed_moves = {x: 0 for x in model.trainable_parameters}
        self._accepted_moves = {x: 0 for x in model.trainable_parameters}

    def reset_counters(self):
        """Resets this samplers count of the number of
        proposed and accepted moves.
        """
        self._proposed_moves = {x: 0 for x in self._model.trainable_parameters}
        self._accepted_moves = {x: 0 for x in self._model.trainable_parameters}

    @abc.abstractmethod
    def step(self, parameters, log_p, adapt):
        """Propagates a set of parameters forward one step.

        Parameters
        ----------
        parameters: dict of str and numpy.ndarray
            The parameters to propagate with shape=(1,)
        log_p: float
            The value of log p evaluated at the current `parameters`.
        adapt: bool
            If True, this sampler will attempt to tune it's
            parameters for optimal sampling.

        Returns
        -------
        dict of str and numpy.ndarray
            The new parameters with shape=(1,).
        float
            The value of log p evaluated at the new parameters.
        bool
            Whether this move was accepted or not.
        """
        raise NotImplementedError()

    def get_statistics_dictionary(self):
        """Returns a dictionary containing statistics
        about this sampler.

        Returns
        -------
        dict of str and Any
        """
        return {
            "proposed_moves": self.proposed_moves,
            "accepted_moves": self.accepted_moves,
        }
