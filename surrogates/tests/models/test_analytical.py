"""
Unit and regression test for the datasets module.
"""
import numpy
import pytest

from surrogates.models.analytical import StollWerthModel


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


def test_evaluate(default_parameters):

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = default_parameters

    model = StollWerthModel(
        fixed_parameters={"L": bond_length, "Q": quadrupole, "temperature": 308.0},
        molecular_weight=30.069,
    )

    values, _ = model.evaluate(
        ["liquid_density", "vapor_pressure", "surface_tension"],
        numpy.array([[epsilon, sigma]]),
    )
    assert numpy.isclose(values["liquid_density"], 285.1592692)
    assert numpy.isclose(values["vapor_pressure"], 5027.57796073)
    assert numpy.isclose(values["surface_tension"], 0.00017652)


def test_evaluate_vectorized(default_parameters):

    epsilon, sigma, bond_length, _, quadrupole, _ = default_parameters

    model = StollWerthModel(
        fixed_parameters={"L": bond_length, "Q": quadrupole}, molecular_weight=30.069
    )

    parameters = numpy.array(
        [[epsilon, sigma, 298.0], [epsilon, sigma, 300.0], [epsilon, sigma, 308.0]]
    )

    values, uncertainties = model.evaluate(
        ["liquid_density", "vapor_pressure", "surface_tension"], parameters
    )

    assert len(values["liquid_density"]) == len(parameters)
    assert len(values["vapor_pressure"]) == len(parameters)
    assert len(values["surface_tension"]) == len(parameters)

    assert len(uncertainties["liquid_density"]) == len(parameters)
    assert len(uncertainties["vapor_pressure"]) == len(parameters)
    assert len(uncertainties["surface_tension"]) == len(parameters)
