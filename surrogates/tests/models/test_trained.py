import numpy

from surrogates.models.trained import GaussianProcessModel


def test_gaussian_process_no_noise():

    model = GaussianProcessModel()
    model.add_training_point(numpy.array([0.0, 0.0]), 0.0, 1.0, 2.0, 3.0)

    liquid_density, vapor_pressure, surface_tension = model.evaluate(
        numpy.array([[0.0, 0.0]]), numpy.zeros((1, 1))
    )

    assert numpy.isclose(liquid_density, 1.0)
    assert numpy.isclose(vapor_pressure, 2.0)
    assert numpy.isclose(surface_tension, 3.0)

    liquid_density, vapor_pressure, surface_tension = model.evaluate(
        numpy.array([[1.0, 1.0]]), numpy.zeros((1, 1))
    )

    assert not numpy.isclose(liquid_density, 4.0)
    assert not numpy.isclose(vapor_pressure, 5.0)
    assert not numpy.isclose(surface_tension, 6.0)

    model.add_training_point(numpy.array([1.0, 1.0]), 0.0, 4.0, 5.0, 6.0)

    liquid_density, vapor_pressure, surface_tension = model.evaluate(
        numpy.array([[0.0, 0.0]]), numpy.zeros((1, 1))
    )

    assert numpy.isclose(liquid_density, 1.0)
    assert numpy.isclose(vapor_pressure, 2.0)
    assert numpy.isclose(surface_tension, 3.0)

    liquid_density, vapor_pressure, surface_tension = model.evaluate(
        numpy.array([[1.0, 1.0]]), numpy.zeros((1, 1))
    )

    assert numpy.isclose(liquid_density, 4.0)
    assert numpy.isclose(vapor_pressure, 5.0)
    assert numpy.isclose(surface_tension, 6.0)
