"""A collection of common driver targets.
"""
import abc
from typing import Tuple

import numpy

from surrogates.drivers import DriverTarget


class PropertyTarget(DriverTarget, abc.ABC):
    @classmethod
    @abc.abstractmethod
    def supported_properties(cls) -> Tuple[str, ...]:
        """tuple of str: The properties supported by this target."""
        raise NotImplementedError()

    @property
    def property_type(self) -> str:
        """str: The target property."""
        return self._property_type

    @property
    def temperatures(self) -> numpy.ndarray:
        """numpy.ndarray: The temperatures this target should be evaluated
        at with shape=(n_temperatures, 1)."""
        return self._parameters["temperature"]

    def __init__(self, temperatures: numpy.ndarray, property_type: str):

        super(PropertyTarget, self).__init__(parameters={"temperature": temperatures})

        assert property_type in self.supported_properties()
        self._property_type = property_type
