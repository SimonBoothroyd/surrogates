from surrogates.models import BayesianModel


class TwoCenterLennardJones(BayesianModel):
    """A two-center Lennard-Jones model.
    """

    def __init__(
        self, priors, likelihoods, driver, fixed_parameters,
    ):

        required_parameters = {"epsilon", "sigma", "bond_length", "quadrupole"}
        provided_parameters = [*priors, *fixed_parameters.keys()]

        assert required_parameters == set(provided_parameters)
        assert len(required_parameters) == len(provided_parameters)

        super(TwoCenterLennardJones, self).__init__(
            priors, likelihoods, driver, fixed_parameters
        )
