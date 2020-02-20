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

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    model = StollWerthSurrogate(30.069, bond_length)

    value = model.critical_temperature(epsilon, sigma, bond_length, quadrupole)
    assert numpy.isclose(value, 310.99575)


def test_liquid_density():

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    model = StollWerthSurrogate(30.069, bond_length)

    temperatures = numpy.array([308.0])

    value = model.liquid_density(numpy.array([epsilon, sigma]), temperatures)
    assert numpy.isclose(value, 285.1592692)


def test_vapor_pressure():

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    model = StollWerthSurrogate(30.069, bond_length)

    temperatures = numpy.array([308.0])

    value = model.vapor_pressure(numpy.array([epsilon, sigma]), temperatures)
    assert numpy.isclose(value, 5027.57796073)


def test_surface_tension():

    (
        epsilon,
        sigma,
        bond_length,
        bond_length_star,
        quadrupole,
        quadrupole_star_sqr,
    ) = generate_parameters()

    model = StollWerthSurrogate(30.069, bond_length)

    temperatures = numpy.array([308.0])

    value = model.surface_tension(numpy.array([epsilon, sigma]), temperatures)
    assert numpy.isclose(value, 0.00017652)


def test_evaluate():

    epsilon, sigma, bond_length, _, quadrupole, _ = generate_parameters()

    model = StollWerthSurrogate(30.069, bond_length)

    parameters = numpy.array([epsilon, sigma])
    temperatures = numpy.array([298.0, 300.0, 308.0])

    values, uncertainties, _ = model.evaluate(parameters, temperatures)

    assert len(values["liquid_density"]) == len(temperatures)
    assert len(values["vapor_pressure"]) == len(temperatures)
    assert len(values["surface_tension"]) == len(temperatures)

    assert len(uncertainties["liquid_density"]) == len(temperatures)
    assert len(uncertainties["vapor_pressure"]) == len(temperatures)
    assert len(uncertainties["surface_tension"]) == len(temperatures)
