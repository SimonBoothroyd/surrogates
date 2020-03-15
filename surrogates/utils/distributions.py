"""
A set of common distributions which are differentiable with
autograd.
"""
import abc

import numpy
import torch.distributions

from surrogates.utils.serializeable import Serializable


class Distribution(Serializable, abc.ABC):
    @property
    def n_variables(self):
        """int: The number of variables which this distribution is a
        function of."""
        return 1

    @abc.abstractmethod
    def log_pdf(self, x):
        raise NotImplementedError()


class MultivariateDistribution(Distribution, abc.ABC):
    """A distribution which a function of more than
    one variable.
    """

    @property
    @abc.abstractmethod
    def n_variables(self):
        raise NotImplementedError()


class MultivariateNormal(MultivariateDistribution):
    """A multivariate normal distribution.
    """

    @property
    def n_variables(self):
        return self._dimension

    def __init__(self, mean, covariance):

        self._mean = mean
        self._dimension = len(self._mean)

        assert len(covariance.shape) == 2
        assert covariance.shape[0] == covariance.shape[1] == self._dimension

        self._covariance = covariance

        self._inverse_covariance = numpy.linalg.inv(covariance)

        self._log_determinant = numpy.log(numpy.linalg.det(covariance))

    def log_pdf(self, x):

        residuals = x - self._mean

        log_p = -0.5 * (
            self._log_determinant
            + numpy.einsum(
                "...j,jk,...k", residuals, self._inverse_covariance, residuals
            )
            + self._dimension * numpy.log(2 * numpy.pi)
        )

        return log_p

    def sample(self):

        torch_mean = torch.tensor(self._mean, dtype=torch.float64)
        torch_covariance = torch.tensor(self._covariance, dtype=torch.float64)

        distribution = torch.distributions.MultivariateNormal(
            torch_mean, torch_covariance
        )
        return distribution.rsample().numpy()

    def to_dict(self):
        return {"mean": self._mean.tolist(), "covariance": self._covariance.tolist()}

    @classmethod
    def from_dict(cls, dictionary):
        super(MultivariateNormal, cls).from_dict(dictionary)

        return cls(
            numpy.asarray(dictionary["mean"]), numpy.asarray(dictionary["covariance"]),
        )

    @staticmethod
    def _validate(dictionary):
        assert "mean" in dictionary and "covariance" in dictionary


class Exponential(Distribution):
    def __init__(self, rate):
        self.rate = rate

    def log_pdf(self, x):

        if numpy.any(x < 0.0):
            return -numpy.inf

        return numpy.log(self.rate) - self.rate * x

    def sample(self):
        return torch.distributions.Exponential(self.rate).rsample().item()

    def to_dict(self):
        return {"rate": self.rate}

    @classmethod
    def from_dict(cls, dictionary):
        super(Exponential, cls).from_dict(dictionary)
        return cls(dictionary["rate"])

    @staticmethod
    def _validate(dictionary):
        assert "rate" in dictionary and dictionary["rate"] >= 0.0


class Normal(Distribution):
    def __init__(self, loc, scale):

        self.loc = loc
        self.scale = scale

    def log_pdf(self, x):

        var = self.scale ** 2
        log_scale = numpy.log(self.scale)

        return (
            -((x - self.loc) ** 2) / (2 * var)
            - log_scale
            - numpy.log(numpy.sqrt(2 * numpy.pi))
        )

    def sample(self):
        return torch.distributions.Normal(self.loc, self.scale).rsample().item()

    def to_dict(self):
        return {"loc": self.loc, "scale": self.scale}

    @classmethod
    def from_dict(cls, dictionary):
        super(Normal, cls).from_dict(dictionary)
        return cls(dictionary["loc"], dictionary["scale"])

    @staticmethod
    def _validate(dictionary):
        assert (
            "loc" in dictionary and "scale" in dictionary and dictionary["scale"] >= 0.0
        )


class Uniform(Distribution):
    def __init__(self, low=0.0, high=1.0):

        self.low = low
        self.high = high

    def log_pdf(self, x):

        if self.low <= x <= self.high:
            return -numpy.log(self.high - self.low)

        return -numpy.inf

    def sample(self):
        return torch.distributions.Uniform(self.low, self.high).rsample().item()

    def to_dict(self):
        return {"low": self.low, "high": self.high}

    @classmethod
    def from_dict(cls, dictionary):
        super(Uniform, cls).from_dict(dictionary)
        return cls(dictionary["low"], dictionary["high"])

    @staticmethod
    def _validate(dictionary):
        assert (
            "low" in dictionary
            and "high" in dictionary
            and dictionary["low"] < dictionary["high"]
        )
