from typing import Dict, List

from surrogates.drivers import Driver
from surrogates.likelihoods.likelihoods import Likelihood
from surrogates.models import BayesianModel
from surrogates.utils.distributions import Distribution


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
