from typing import Dict, List, Tuple

import numpy

from surrogates.drivers import Driver
from surrogates.likelihoods.likelihoods import Likelihood
from surrogates.models import BayesianModel
from surrogates.utils.distributions import Distribution


class UnconditionedModel(BayesianModel):
    """A model which is not conditioned upon any data (i.e.
    a model whose only contribution to the posterior is the
    prior).
    """

    def __init__(self, priors: Dict[str, Distribution]):

        # noinspection PyTypeChecker
        super(UnconditionedModel, self).__init__(priors, None, None, {})

    def evaluate(
        self, parameters: Dict[str, numpy.ndarray]
    ) -> Tuple[numpy.ndarray, Dict[str, numpy.ndarray]]:

        return self.evaluate_log_prior(parameters)


class TwoCenterLennardJones(BayesianModel):
    """A two-center Lennard-Jones model.
    """

    def __init__(
        self,
        priors: Dict[str, Distribution],
        likelihoods: List[Likelihood],
        driver: Driver,
        fixed_parameters: Dict[str, float],
    ):

        required_parameters = {"epsilon", "sigma", "bond_length", "quadrupole"}
        provided_parameters = [*priors, *fixed_parameters.keys()]

        assert required_parameters == set(provided_parameters)
        assert len(required_parameters) == len(provided_parameters)

        super(TwoCenterLennardJones, self).__init__(
            priors, likelihoods, driver, fixed_parameters
        )
