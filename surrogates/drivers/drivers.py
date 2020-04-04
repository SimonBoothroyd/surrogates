import abc
from typing import Dict, List, Tuple

import numpy
from numpy import ndarray


class DriverTarget(abc.ABC):
    """A particular target that a driver should attempt
    to evaluate.
    """

    @property
    def parameters(self):
        """dict of str and numpy.ndarray: The non-model parameters which the target
        will be evaluated at with shape=(n_parameters,)."""
        return self._parameters

    def __init__(self, parameters: Dict[str, ndarray]):
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
    def evaluate(
        self,
        targets: List[DriverTarget],
        parameters: Dict[str, numpy.ndarray],
        compute_gradients: bool,
    ) -> Tuple[
        List[numpy.ndarray], List[numpy.ndarray], List[Dict[str, numpy.ndarray]]
    ]:
        """Evaluates the specified properties at the provided
        parameters.

        Parameters
        ----------
        targets: list of DriverTarget
            The targets to evaluate.
        parameters: dict of str and numpy.ndarray
            The model parameters to evaluate at with shape(n_sets,)
            where n_sets is the number of parameter sets to evaluate at.
        compute_gradients: bool
            Whether to compute the

        Returns
        -------
        list of numpy.ndarray
            The evaluated values of each target with shape=(n_sets,)
        list of numpy.ndarray
            The uncertainties in the evaluated values of each target
            with shape=(n_sets,).
        list of dict of str and numpy.ndarray, optional
            The gradients of each evaluated value with respect to the input
            parameters. This will be `None` if `compute_gradients` is `False`.
        """
        raise NotImplementedError()
