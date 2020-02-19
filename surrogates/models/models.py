import abc


class Model(abc.ABC):
    """A model of a system of molecules, which can be evaluated to
    estimate the liquid density, vapor pressure and surface tension
    for a range of temperatures.
    """

    def evaluate(self, parameters, temperatures):
        """Evaluate this model for a set of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters to evaluate at with
            shape=(n parameters, 1).
        temperatures: numpy.ndarray
            The temperatures to evaluate the properties at with
            shape=(n_temperatures).

        Returns
        -------
        numpy.ndarray
            The values of the liquid density evaluated at each temperature
            and using the specified parameters (shape=(n_temperatures, 1)).
        numpy.ndarray
            The values of the vapor pressure evaluated at each temperature
            and using the specified parameters (shape=(n_temperatures, 1)).
        numpy.ndarray
            The values of the surface tension evaluated at each temperature
            and using the specified parameters (shape=(n_temperatures, 1)).
        """
        raise NotImplementedError()

    def evaluate_gradients(self, parameters, temperatures):
        """Evaluate the gradients of this model for a set of parameters.

        Parameters
        ----------
        parameters: numpy.ndarray
            The values of the parameters to evaluate at with
            shape=(n parameters, 1).
        temperatures: numpy.ndarray
            The temperatures to evaluate the properties at with
            shape=(n_temperatures).

        Returns
        -------
        numpy.ndarray
            The gradients of the liquid density evaluated at each temperature
            with respect to the specified parameters (shape=(n_temperatures, n_parameters)).
        numpy.ndarray
            The gradients of the vapor pressure evaluated at each temperature
            with respect to the specified parameters (shape=(n_temperatures, n_parameters)).
        numpy.ndarray
            The gradients of the surface tension evaluated at each temperature
            with respect to the specified parameters (shape=(n_temperatures, n_parameters)).
        """
        raise NotImplementedError()
