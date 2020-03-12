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
        {"a": numpy.array([1.0]), "b": numpy.array([2.0]), "c": numpy.array([3.0])},
        {"a": numpy.array([0.0]), "b": numpy.array([0.0]), "c": numpy.array([0.0])},
    )

    values, uncertainties = model.evaluate(["a", "b", "c"], numpy.array([[0.0, 0.0]]))

    assert numpy.isclose(values["a"], 1.0)
    assert numpy.isclose(values["b"], 2.0)
    assert numpy.isclose(values["c"], 3.0)

    assert numpy.isclose(uncertainties["a"], 0.0)
    assert numpy.isclose(uncertainties["b"], 0.0)
    assert numpy.isclose(uncertainties["c"], 0.0)

    model.add_training_data(
        numpy.array([[1.0, 1.0]]),
        {"a": numpy.array([4.0]), "b": numpy.array([5.0]), "c": numpy.array([6.0])},
        {"a": numpy.array([0.0]), "b": numpy.array([0.0]), "c": numpy.array([0.0])},
    )

    values, uncertainties = model.evaluate(["a", "b", "c"], numpy.array([[0.0, 0.0]]))

    assert numpy.isclose(values["a"], 1.0)
    assert numpy.isclose(values["b"], 2.0)
    assert numpy.isclose(values["c"], 3.0)

    assert numpy.isclose(uncertainties["a"], 0.0)
    assert numpy.isclose(uncertainties["b"], 0.0)
    assert numpy.isclose(uncertainties["c"], 0.0)

    values, uncertainties = model.evaluate(["a", "b", "c"], numpy.array([[1.0, 1.0]]))

    assert numpy.isclose(values["a"], 4.0)
    assert numpy.isclose(values["b"], 5.0)
    assert numpy.isclose(values["c"], 6.0)

    assert numpy.isclose(uncertainties["a"], 0.0)
    assert numpy.isclose(uncertainties["b"], 0.0)
    assert numpy.isclose(uncertainties["c"], 0.0)
