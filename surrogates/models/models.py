import abc


class Model(abc.ABC):
    """A model of a system of molecules, which can be evaluated to
    estimate the liquid density, vapor pressure and surface tension
    for a range of temperatures.
    """

    @abc.abstractmethod
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
        dict of str and numpy.ndarray
            The values of the properties evaluated by this model using the
            specified parameters. Each array has a shape=(n_temperatures, 1)).
        dict of str and numpy.ndarray
            The uncertainties in the values of the properties evaluated by this model
            using the specified parameters. Each array has a shape=(n_temperatures, 1)).
        """
        raise NotImplementedError()


class TrainableModel(Model, abc.ABC):
    """A model which can be trained upon previously generated data,
    and then be more rapidly evaluated than generating fresh data.
    """

    @abc.abstractmethod
    def add_training_point(
        self, parameter, temperature, liquid_density, vapor_pressure, surface_tension
    ):
        """Trains the model on a new set of physical properties.

        Parameters
        ----------
        parameter: numpy.ndarray
            The parameters the measurements were made at.
        temperature: float
            The temperature the measurements were recorded at.
        numpy.ndarray
            The value of the liquid density evaluated at the temperature
            and using the specified parameters (shape=(1, 1)).
        numpy.ndarray
            The values of the vapor pressure evaluated at the temperature
            and using the specified parameters (shape=(1, 1)).
        numpy.ndarray
            The values of the surface tension evaluated at the temperature
            and using the specified parameters (shape=(1, 1)).
        """
        raise NotImplementedError()
