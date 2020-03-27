import numpy

from surrogates.models.surrogate import GaussianProcess


def test_gaussian_process_no_noise():

    model = GaussianProcess(
        parameter_labels=["a", "b"],
        condition_parameters=True,
        condition_data=True,
        learning_rate=0.7,
        train_iterations=25,
        double_precision=True,
    )

    model.add_training_data(
        {"a": numpy.array([0.0]), "b": numpy.array([0.0])},
        numpy.array([1.0]),
        numpy.array([0.0]),
    )

    value, uncertainty = model.evaluate(
        {"a": numpy.array([0.0]), "b": numpy.array([0.0])}
    )

    assert numpy.isclose(value, 1.0)
    assert numpy.isclose(uncertainty, 0.0)

    model.add_training_data(
        {"a": numpy.array([1.0]), "b": numpy.array([1.0])},
        numpy.array([4.0]),
        numpy.array([0.0]),
    )

    value, uncertainty = model.evaluate(
        {"a": numpy.array([0.0]), "b": numpy.array([0.0])}
    )

    assert numpy.isclose(value, 1.0)
    assert numpy.isclose(uncertainty, 0.0)

    value, uncertainty = model.evaluate(
        {"a": numpy.array([1.0]), "b": numpy.array([1.0])}
    )

    assert numpy.isclose(value, 4.0)
    assert numpy.isclose(uncertainty, 0.0)
