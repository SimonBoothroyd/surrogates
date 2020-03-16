import os

import numpy
import pytest
from pkg_resources import resource_filename

from surrogates.datasets import DataSet
from surrogates.drivers.analytic import StollWerthDriver, StollWerthTarget
from surrogates.likelihoods.likelihoods import GaussianLikelihood
from surrogates.models.simple import TwoCenterLennardJones
from surrogates.utils.distributions import Uniform


@pytest.fixture
def reference_data_set():
    """Loads a data set of properties for C2H6 measured at 300K (within 0.25K).

    Returns
    -------
    DataSet
        The loaded data set.
    """

    compound_formula = "C2H6"
    temperature = 300.0

    data_set_path = resource_filename(
        "surrogates", os.path.join("data", "trc_data", f"{compound_formula}.json")
    )
    data_set = DataSet.parse_file(data_set_path)

    # For now strip out any data points away from 299K
    liquid_density_index = (
        numpy.abs(data_set.liquid_density[0, :] - temperature)
    ).argmin()
    data_set.liquid_density = data_set.liquid_density[:, liquid_density_index].reshape(
        1, -1
    )

    surface_tension_index = (
        numpy.abs(data_set.surface_tension[0, :] - temperature)
    ).argmin()
    data_set.surface_tension = data_set.surface_tension[
        :, surface_tension_index
    ].reshape(1, -1)

    vapor_pressure_index = (
        numpy.abs(data_set.vapor_pressure[0, :] - temperature)
    ).argmin()
    data_set.vapor_pressure = data_set.vapor_pressure[:, vapor_pressure_index].reshape(
        1, -1
    )

    # For convenience round the temperatures to be exactly that specified
    data_set.liquid_density[0, 0] = temperature
    data_set.surface_tension[0, 0] = temperature
    data_set.vapor_pressure[0, 0] = temperature

    return data_set


def test_2clj_likelihood(reference_data_set):

    target_properties = ["liquid_density", "surface_tension", "vapor_pressure"]

    (
        reference_temperatures,
        reference_values,
        reference_uncertainties,
    ) = reference_data_set.to_likelihood_dict(target_properties)

    likelihoods = []

    for target_property in target_properties:

        driver_target = StollWerthTarget(
            reference_temperatures[target_property], target_property
        )

        likelihood = GaussianLikelihood(
            reference_values[target_property],
            reference_uncertainties[target_property],
            driver_target,
        )

        likelihoods.append(likelihood)

    driver = StollWerthDriver(0.0, reference_data_set.molecular_weight)

    model = TwoCenterLennardJones(
        priors={"epsilon": Uniform(-10, 10), "sigma": Uniform(-10, 10)},
        likelihoods=likelihoods,
        driver=driver,
        fixed_parameters={
            "bond_length": reference_data_set.bond_length / 10.0,
            "quadrupole": 0.0,
        },
    )

    map_parameters = {
        "epsilon": numpy.array([97.6225759]),
        "sigma": numpy.array([0.37964502]),
    }

    log_p = model.evaluate_log_likelihood(map_parameters)
    assert numpy.isclose(log_p, -21.27155532219009)
