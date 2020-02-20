import os

import numpy
import pytest
from pkg_resources import resource_filename

from surrogates.datasets import DataSet
from surrogates.kernels.optimization import GradientDescent
from surrogates.models.analytical import StollWerthSurrogate


@pytest.fixture
def small_data_set():
    """A data set containing a single liquid density, vapor pressure
    and surface tension measurement at 299K.

    Returns
    -------
    DataSet
    """

    data_set_path = resource_filename(
        "surrogates", os.path.join("data", "trc_data", "C2H6.json")
    )
    data_set = DataSet.parse_file(data_set_path)

    # For now strip out any data points away from 299K
    data_set.liquid_densities = data_set.liquid_densities[:, 203].reshape(1, -1)
    data_set.surface_tensions = data_set.surface_tensions[:, 82].reshape(1, -1)
    data_set.vapor_pressures = data_set.vapor_pressures[:, 246].reshape(1, -1)

    # For convenience round the temperatures to 299K
    data_set.liquid_densities[0, 0] = 299
    data_set.surface_tensions[0, 0] = 299
    data_set.vapor_pressures[0, 0] = 299

    return data_set


def test_gradient_descent(small_data_set):

    model = StollWerthSurrogate(small_data_set.molecular_weight)

    # Define the initial epsilon (kJ / mol), sigma (nm), L (nm) and Q
    # parameters of our two center Lennard-Jones model.
    initial_parameters = numpy.array(
        [100, 0.36, small_data_set.bond_length / 10.0, 0.0]
    )

    # Define a set of learning rates for each of the different parameters
    learning_rates = numpy.array([4.0, 0.0005, 0.0, 0.0])

    # Set up an array to monitor the parameter traces and the cost function
    # trace.
    iterations = 1

    # Run the optimization
    optimizer = GradientDescent(model, small_data_set, learning_rates)
    optimizer.run(initial_parameters, iterations)

    assert optimizer.parameter_trace is not None
    assert optimizer.cost_function_trace is not None
