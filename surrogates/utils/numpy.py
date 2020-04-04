from typing import Dict, List

import numpy


def parameter_dict_to_array(
    parameters: Dict[str, numpy.ndarray], parameter_order: List[str]
) -> numpy.ndarray:
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


def add_parameter_dicts(
    value_a: Dict[str, numpy.ndarray], value_b: Dict[str, numpy.ndarray]
) -> Dict[str, numpy.ndarray]:
    """Adds two parameter dictionaries together.

    Parameters
    ----------
    value_a: dict of str and numpy.ndarray
        The first set of parameters..
    value_b: dict of str and numpy.ndarray
        The second set of parameters..

    Returns
    -------
    dict of str and numpy.ndarray
        The summed parameters.
    """

    assert len(value_a) == len(value_b)
    return {x: y + value_b[x] for x, y in value_a.items()}
