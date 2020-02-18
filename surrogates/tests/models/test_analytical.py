"""
Unit and regression test for the datasets module.
"""
import numpy

from surrogates.models.analytical import StollWerthSurrogate


def generate_parameters():
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

    model = StollWerthSurrogate(30.069)

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    value = model.critical_temperature(epsilon, sigma, bond_length, quadrupole)
    assert numpy.isclose(value, 310.99575)


def test_liquid_density():

    model = StollWerthSurrogate(30.069)

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    temperatures = numpy.array([308.0])

    value = model.liquid_density(temperatures, epsilon, sigma, bond_length, quadrupole)
    assert numpy.isclose(value, 285.1592692)


def test_vapor_pressure():

    model = StollWerthSurrogate(30.069)  # C2H6

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    temperatures = numpy.array([308.0])

    value = model.vapor_pressure(temperatures, epsilon, sigma, bond_length, quadrupole)
    assert numpy.isclose(value, 5027.57796073)


def test_surface_tension():

    model = StollWerthSurrogate(30.069)

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    temperatures = numpy.array([308.0])

    value = model.surface_tension(temperatures, epsilon, sigma, bond_length, quadrupole)
    assert numpy.isclose(value, 0.00017652)


def test_evaluate():

    model = StollWerthSurrogate(30.069)
    epsilon, sigma, bond_length, _, quadrupole, _ = generate_parameters()

    parameters = numpy.array([epsilon, sigma, bond_length, quadrupole])
    temperatures = numpy.array([298.0, 300.0, 308.0])

    liquid_densities, vapor_pressure, surface_tensions = model.evaluate(
        parameters, temperatures
    )

    assert len(liquid_densities) == len(temperatures)
    assert len(vapor_pressure) == len(temperatures)
    assert len(surface_tensions) == len(temperatures)
