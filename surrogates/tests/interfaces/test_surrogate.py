import numpy

from surrogates.interfaces.surrogate import StollWerthInterface


def test_critical_temperature():

    interface = StollWerthInterface(fractional_noise=0.0, molecular_weight=30.069)

    value = interface._critical_temperature(98.0, 0.37800, 0.15, 0.01)
    assert numpy.isclose(value, 310.99575)
