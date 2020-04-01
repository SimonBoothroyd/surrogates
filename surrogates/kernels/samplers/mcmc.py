"""
This module implements a samplers based off of the
Metropolis-Hasting acceptance criteria. These samplers
do not require nor make use of gradient information.
"""
from typing import Any, Dict, Tuple

import numpy
import torch

from surrogates.kernels.samplers import Sampler
from surrogates.models import BayesianModel
from surrogates.utils import distributions


class Metropolis(Sampler):
    """A base class for different in-model parameter samplers."""

    @property
    def proposal_sizes(self) -> Dict[str, numpy.ndarray]:
        """dict of str and numpy.ndarray: The size of the proposals to make for each
        parameter with shape=(n_trainable_parameters,).
        """
        return self._proposal_sizes

    @proposal_sizes.setter
    def proposal_sizes(self, value: Dict[str, numpy.ndarray]):

        assert all(x in self._model.trainable_parameters for x in value)
        self._proposal_sizes = value

    def __init__(
        self,
        model: BayesianModel,
        proposal_sizes: Dict[str, numpy.ndarray],
        acceptance_target: float = 0.5,
        tune_frequency: int = 100,
    ):
        """
        Parameters
        ----------
        proposal_sizes: dict of str and numpy.ndarray
            The size of the proposals to make for each parameter
            with shape=(1,).
        acceptance_target: float
            The target acceptance rate for this sampler
        tune_frequency: int
            The number of steps to take before attempting to
            tune the parameters.
        """
        super().__init__(model)

        assert 0.0 < acceptance_target <= 1.0
        self._acceptance_target = acceptance_target

        for label in proposal_sizes:

            if not isinstance(proposal_sizes[label], float):
                continue

            proposal_sizes[label] = numpy.array([proposal_sizes[label]])

        self.proposal_sizes = proposal_sizes

        self._tune_frequency = tune_frequency

    def step(
        self, parameters: Dict[str, numpy.ndarray], log_p: numpy.ndarray, adapt: bool
    ) -> Tuple[Dict[str, numpy.ndarray], numpy.ndarray, bool]:

        # Choose a random parameter to change
        parameter_index = torch.randint(self._model.n_trainable_parameters, (1,)).item()
        parameter_label = [*parameters][parameter_index]

        # Sample the new parameters from a normal distribution.
        proposed_parameters = {x: y for x, y in parameters.items()}

        proposed_parameters[parameter_label] = numpy.array(
            [
                distributions.Normal(
                    parameters[parameter_label][0],
                    self._proposal_sizes[parameter_label][0],
                ).sample()
            ]
        )

        proposed_log_p, _ = self._log_p_function(proposed_parameters)

        alpha = proposed_log_p - log_p

        random_number = numpy.log(torch.rand((1,)).item())
        accept = random_number < alpha

        # Update the bookkeeping
        self._proposed_moves[parameter_label] += 1

        if accept:

            self._accepted_moves[parameter_label] += 1

            parameters = proposed_parameters
            log_p = proposed_log_p

        # Tune the proposals if needed
        total_proposed_moves = sum(self._proposed_moves.values())

        if (
            adapt
            and self._tune_frequency > 0
            and total_proposed_moves > 0
            and total_proposed_moves % self._tune_frequency == 0
        ):
            self._tune_proposals()

        return parameters, log_p, accept

    def _tune_proposals(self):
        """Attempt to tune the move proposals to reach the
        `acceptance_target`.
        """

        for parameter_label in self.proposed_moves:

            if self._proposed_moves[parameter_label] == 0:
                continue

            rate = (
                self._accepted_moves[parameter_label]
                / self._proposed_moves[parameter_label]
            )

            scale = 0.9 if rate < self._acceptance_target else 1.1
            scale = 1.0 if self._proposed_moves[parameter_label] == 0 else scale

            self._proposal_sizes[parameter_label] *= scale

            self.reset_counters()

    def get_statistics_dictionary(self) -> Dict[str, Any]:

        return_value = super(Metropolis, self).get_statistics_dictionary()
        return_value.update(
            {"proposal_sizes": {x: y.tolist() for x, y in self.proposal_sizes.items()}}
        )

        return return_value
