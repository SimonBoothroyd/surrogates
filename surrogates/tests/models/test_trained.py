import numpy

from surrogates.models.trained import GaussianProcessModel
from surrogates.utils.distributions import Uniform


def test_gaussian_process_no_noise():

    model = GaussianProcessModel(
        priors={"a": Uniform(-10, 10), "b": Uniform(-10, 10)},
        fixed_parameters={},
        condition_parameters=True,
        condition_data=True,
        learning_rate=0.25,
        train_iterations=25,
    )

    model.add_training_data(
        numpy.array([[0.0, 0.0]]),
        {
            "liquid_density": numpy.array([1.0]),
            "vapor_pressure": numpy.array([2.0]),
            "surface_tension": numpy.array([3.0]),
        },
        {
            "liquid_density": numpy.array([0.0]),
            "vapor_pressure": numpy.array([0.0]),
            "surface_tension": numpy.array([0.0]),
        },
    )

    values, uncertainties = model.evaluate(numpy.array([[0.0, 0.0]]))

    assert numpy.isclose(values["liquid_density"], 1.0)
    assert numpy.isclose(values["vapor_pressure"], 2.0)
    assert numpy.isclose(values["surface_tension"], 3.0)

    assert numpy.isclose(uncertainties["liquid_density"], 0.0)
    assert numpy.isclose(uncertainties["vapor_pressure"], 0.0)
    assert numpy.isclose(uncertainties["surface_tension"], 0.0)

    model.add_training_data(
        numpy.array([[1.0, 1.0]]),
        {
            "liquid_density": numpy.array([4.0]),
            "vapor_pressure": numpy.array([5.0]),
            "surface_tension": numpy.array([6.0]),
        },
        {
            "liquid_density": numpy.array([0.0]),
            "vapor_pressure": numpy.array([0.0]),
            "surface_tension": numpy.array([0.0]),
        },
    )

    values, uncertainties = model.evaluate(numpy.array([[0.0, 0.0]]))

    assert numpy.isclose(values["liquid_density"], 1.0)
    assert numpy.isclose(values["vapor_pressure"], 2.0)
    assert numpy.isclose(values["surface_tension"], 3.0)

    assert numpy.isclose(uncertainties["liquid_density"], 0.0)
    assert numpy.isclose(uncertainties["vapor_pressure"], 0.0)
    assert numpy.isclose(uncertainties["surface_tension"], 0.0)

    values, uncertainties = model.evaluate(numpy.array([[1.0, 1.0]]))

    assert numpy.isclose(values["liquid_density"], 4.0)
    assert numpy.isclose(values["vapor_pressure"], 5.0)
    assert numpy.isclose(values["surface_tension"], 6.0)

    assert numpy.isclose(uncertainties["liquid_density"], 0.0)
    assert numpy.isclose(uncertainties["vapor_pressure"], 0.0)
    assert numpy.isclose(uncertainties["surface_tension"], 0.0)
