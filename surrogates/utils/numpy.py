from typing import List

import numpy


def parameter_dict_to_array(parameters, parameter_order: List[str]):
    """Convert a dictionary of numpy arrays to a single
    numpy array (with the parameter ordering dictated by
    the ordering of a set of parameter labels.

    Parameters
    ----------
    parameters: dict of str and numpy.ndarray
        The parameter dictionary to convert.
    parameter_order: list of str
        The order to concatenate the parameters in.

    Returns
    -------
    numpy.ndarray
        The converted parameters.
    """

    # Make sure the parameter / values arrays are the correct shapes.
    n_data_points = min(len(x) for x in parameters.values())
    max_data_points = max(len(x) for x in parameters.values())

    assert n_data_points == max_data_points

    assert all(x in parameter_order for x in parameters)
    assert all(x in parameters for x in parameter_order)

    array_parameters = numpy.zeros((n_data_points, len(parameters)))

    for label, parameter in parameters.items():

        if parameter.ndim == 1:
            parameter = parameter.reshape(-1, 1)

        assert parameter.ndim == 2
        assert parameter.shape[1] == 1

        parameter_index = parameter_order.index(label)
        array_parameters[:, parameter_index] = parameter[:, 0]

    return array_parameters
