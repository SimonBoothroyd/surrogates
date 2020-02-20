import numpy

from surrogates.models.trained import GaussianProcessModel


def test_gaussian_process_no_noise():

    model = GaussianProcessModel()
    model.add_training_point(numpy.array([0.0, 0.0]), 0.0, 1.0, 2.0, 3.0)

    values, uncertainties, _ = model.evaluate(
        numpy.array([[0.0, 0.0]]), numpy.zeros((1, 1))
    )

    assert numpy.isclose(values["liquid_density"], 1.0)
    assert numpy.isclose(values["vapor_pressure"], 2.0)
    assert numpy.isclose(values["surface_tension"], 3.0)

    assert numpy.isclose(uncertainties["liquid_density"], 0.0, atol=1.0e-4)
    assert numpy.isclose(uncertainties["vapor_pressure"], 0.0, atol=1.0e-4)
    assert numpy.isclose(uncertainties["surface_tension"], 0.0, atol=1.0e-4)

    values, uncertainties, _ = model.evaluate(
        numpy.array([[1.0, 1.0]]), numpy.zeros((1, 1))
    )

    assert not numpy.isclose(values["liquid_density"], 4.0)
    assert not numpy.isclose(values["vapor_pressure"], 5.0)
    assert not numpy.isclose(values["surface_tension"], 6.0)

    model.add_training_point(numpy.array([1.0, 1.0]), 0.0, 4.0, 5.0, 6.0)

    values, uncertainties, _ = model.evaluate(
        numpy.array([[0.0, 0.0]]), numpy.zeros((1, 1))
    )

    assert numpy.isclose(values["liquid_density"], 1.0)
    assert numpy.isclose(values["vapor_pressure"], 2.0)
    assert numpy.isclose(values["surface_tension"], 3.0)

    values, uncertainties, _ = model.evaluate(
        numpy.array([[1.0, 1.0]]), numpy.zeros((1, 1))
    )

    assert numpy.isclose(values["liquid_density"], 4.0)
    assert numpy.isclose(values["vapor_pressure"], 5.0)
    assert numpy.isclose(values["surface_tension"], 6.0)
