import numpy
import pytest

from surrogates.drivers.analytic import StollWerthDriver, StollWerthTarget


@pytest.fixture
def default_parameters():
    """Returns a set of parameters (and their reduced values) for
    which regression values for each model property are known.
    """

    epsilon = 98.0
    sigma = 0.37800
    bond_length = 0.15
    quadrupole = 0.01

    quadrupole_star_sqr = (quadrupole * 3.1623) ** 2 / (epsilon * 1.38065 * sigma ** 5)
    bond_length_star = bond_length / sigma

    return (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    )


def test_critical_temperature():

    driver = StollWerthDriver(fractional_noise=0.0, molecular_weight=30.069)

    value = driver._critical_temperature(98.0, 0.37800, 0.15, 0.01)
    assert numpy.isclose(value, 310.99575)


@pytest.mark.parametrize(
    "temperatures, property_type, expected_value",
    [
        (numpy.array([[308.0]]), "liquid_density", 285.1592692),
        (numpy.array([[308.0]]), "vapor_pressure", 5027.57796073),
        (numpy.array([[308.0]]), "surface_tension", 0.00017652),
    ],
)
def test_evaluate(default_parameters, temperatures, property_type, expected_value):

    driver = StollWerthDriver(fractional_noise=0.0, molecular_weight=30.069)
    driver_target = StollWerthTarget(temperatures, property_type)

    (epsilon, sigma, bond_length, _, quadrupole, _,) = default_parameters

    parameters = {
        "epsilon": epsilon,
        "sigma": sigma,
        "bond_length": bond_length,
        "quadrupole": quadrupole,
    }
    values, _, _ = driver.evaluate([driver_target], parameters, True)

    assert numpy.isclose(values[0], expected_value)


@pytest.mark.parametrize(
    "temperatures, property_type",
    [
        (numpy.array([[298.0], [300.0], [308.0]]), "liquid_density"),
        (numpy.array([[298.0], [300.0], [308.0]]), "vapor_pressure"),
        (numpy.array([[298.0], [300.0], [308.0]]), "surface_tension"),
    ],
)
def test_evaluate_vectorized(default_parameters, temperatures, property_type):

    driver = StollWerthDriver(fractional_noise=0.0, molecular_weight=30.069)
    driver_target = StollWerthTarget(temperatures, property_type)

    (epsilon, sigma, bond_length, _, quadrupole, _,) = default_parameters

    parameters = {
        "epsilon": epsilon,
        "sigma": sigma,
        "bond_length": bond_length,
        "quadrupole": quadrupole,
    }
    values, uncertainties, _ = driver.evaluate([driver_target], parameters, False)

    assert len(values[0]) == len(temperatures)
    assert values[0].shape == uncertainties[0].shape
