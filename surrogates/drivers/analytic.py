import os
from typing import Dict, List, Optional, Tuple

import autograd.numpy
import numpy
import yaml
from autograd import grad
from pkg_resources import resource_filename

from surrogates.drivers import Driver
from surrogates.drivers.targets import PropertyTarget


class StollWerthTarget(PropertyTarget):
    @classmethod
    def supported_properties(cls) -> Tuple[str, ...]:
        """tuple of str: The properties supported by this interface."""
        return "liquid_density", "vapor_pressure", "surface_tension"


class StollWerthDriver(Driver):
    """A driver which 'evaluates' properties by using the analytical models
    proposed by Stoll et al and Werth et al, with added, user definable, Gaussian
    noise.
    """

    _reduced_boltzmann = 1.38065
    _reduced_D_to_sqrt_J_m3 = 3.1623

    def __init__(
        self,
        fractional_noise: float,
        molecular_weight: float,
        file_path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        fractional_noise: float
            The magnitude of the noise (relative to the evaluated property)
            to add onto any evaluated properties.
        molecular_weight: float
            The molecular weight (g / mol) of the compound this model
            represents.
        file_path: str, optional
            The path to the model parameters. If unset, the built in
            `DCLJQ.yaml` parameters will be used.
        """
        assert fractional_noise >= 0.0
        self._fractional_noise = fractional_noise

        self.molecular_weight = molecular_weight

        if file_path is None:
            file_path = resource_filename(
                "surrogates", os.path.join("data", "models", "DCLJQ.yaml")
            )

        with open(file_path) as file:
            parameters = yaml.load(file, Loader=yaml.SafeLoader)[
                "correlation_parameters"
            ]

        self.critical_temperature_star_parameters = numpy.array(
            parameters["Stoll"]["T_c_star_params"]
        )
        self.density_star_parameters = numpy.array(
            parameters["Stoll"]["rho_c_star_params"]
        )

        self._b_C1 = numpy.array(parameters["Stoll"]["rho_L_star_params"]["C1_params"])
        self._b_C2_L = numpy.array(
            parameters["Stoll"]["rho_L_star_params"]["C2_params"]
        )
        self._b_C3_L = numpy.array(
            parameters["Stoll"]["rho_L_star_params"]["C3_params"]
        )
        self._b_C2_v = numpy.array(
            parameters["Stoll"]["rho_v_star_params"]["C2_params"]
        )
        self._b_C3_v = numpy.array(
            parameters["Stoll"]["rho_v_star_params"]["C3_params"]
        )

        self._b_c1 = numpy.array(parameters["Stoll"]["P_v_star_params"]["c1_params"])
        self._b_c2 = numpy.array(parameters["Stoll"]["P_v_star_params"]["c2_params"])
        self._b_c3 = numpy.array(parameters["Stoll"]["P_v_star_params"]["c3_params"])

        self._A_a = numpy.array(parameters["Werth"]["A_star_params"]["a_params"])
        self._A_b = numpy.array(parameters["Werth"]["A_star_params"]["b_params"])
        self._A_c = numpy.array(parameters["Werth"]["A_star_params"]["c_params"])
        self._A_d = numpy.array(parameters["Werth"]["A_star_params"]["d_params"])
        self._A_e = numpy.array(parameters["Werth"]["A_star_params"]["e_params"])

        self._B = numpy.array(parameters["Werth"]["A_star_params"]["B_params"])

    @staticmethod
    def _correlation_function_1(
        quadrupole_star: numpy.ndarray,
        bond_length_star: numpy.ndarray,
        b: numpy.ndarray,
    ) -> numpy.ndarray:

        q = quadrupole_star
        l = bond_length_star

        result = (
            1 * b[0]
            + q ** 2 * b[1]
            + q ** 3 * b[2]
            + l ** 3 / (l + 0.4) ** 3 * b[3]
            + l ** 4 / (l + 0.4) ** 5 * b[4]
            + q ** 2 * l ** 2 / (l + 0.4) * b[5]
            + q ** 2 * l ** 3 / (l + 0.4) ** 7 * b[6]
            + q ** 3 * l ** 2 / (l + 0.4) * b[7]
            + q ** 3 * l ** 3 / (l + 0.4) ** 7 * b[8]
        )

        return result

    @staticmethod
    def _correlation_function_2(
        quadrupole_star: numpy.ndarray,
        bond_length_star: numpy.ndarray,
        b: numpy.ndarray,
    ) -> numpy.ndarray:

        q = quadrupole_star
        l = bond_length_star

        result = (
            1 * b[0]
            + q ** 2 * b[1]
            + q ** 3 * b[2]
            + l ** 2 * b[3]
            + l ** 3 * b[4]
            + q ** 2 * l ** 2 * b[5]
            + q ** 2 * l ** 3 * b[6]
            + q ** 3 * l ** 2 * b[7]
        )

        return result

    @staticmethod
    def _correlation_function_3(
        quadrupole_star: numpy.ndarray,
        bond_length_star: numpy.ndarray,
        b: numpy.ndarray,
    ) -> numpy.ndarray:

        q = quadrupole_star
        l = bond_length_star

        result = (
            1 * b[0]
            + q ** 2 * b[1]
            + q ** 3 * b[2]
            + l * b[3]
            + l ** 4 * b[4]
            + q ** 2 * l * b[5]
            + q ** 2 * l ** 4 * b[6]
            + q ** 3 * l ** 4 * b[7]
        )

        return result

    def _a_correlation_function(
        self, quadrupole_star: numpy.ndarray, bond_length_star: numpy.ndarray
    ) -> numpy.ndarray:

        c_a, c_b, c_c, c_d, c_e = self._A_a, self._A_b, self._A_c, self._A_d, self._A_e

        a = 1.0 * c_a
        b = (
            quadrupole_star * c_b[0]
            + quadrupole_star ** 2.0 * c_b[1]
            + quadrupole_star ** 3.0 * c_b[2]
        )
        c = 1.0 / (bond_length_star ** 2.0 + 0.1) * c_c[0]
        d = (
            quadrupole_star ** 2.0 * bond_length_star ** 2.0 * c_d[0]
            + quadrupole_star ** 2.0 * bond_length_star ** 3.0 * c_d[1]
        )
        e = (
            quadrupole_star ** 2 / (bond_length_star ** 2.0 + 0.1) * c_e[0]
            + quadrupole_star ** 2.0 / (bond_length_star ** 5.0 + 0.1) * c_e[1]
        )

        return a + b + c + d + e

    def _critical_temperature_star(
        self, quadrupole_star_sqr: numpy.ndarray, bond_length_star: numpy.ndarray
    ) -> numpy.ndarray:
        """Computes the reduced critical temperature of the two-center
        Lennard-Jones model for a given set of model parameters.

        Parameters
        ----------
        quadrupole_star_sqr: float
            The reduced quadrupole parameter squared.
        bond_length_star: float
            The reduced bond-length parameter

        Returns
        -------
        float
            The reduced critical temperature.
        """

        q = quadrupole_star_sqr
        l = bond_length_star

        b = self.critical_temperature_star_parameters

        t_c_star = (
            1 * b[0]
            + q ** 2 * b[1]
            + q ** 3 * b[2]
            + 1.0 / (0.1 + l ** 2) * b[3]
            + 1.0 / (0.1 + l ** 5) * b[4]
            + q ** 2 / (0.1 + l ** 2) * b[5]
            + q ** 2 / (0.1 + l ** 5) * b[6]
            + q ** 3 / (0.1 + l ** 2) * b[7]
            + q ** 3 / (0.1 + l ** 5) * b[8]
        )

        return t_c_star

    def _critical_temperature(
        self,
        epsilon: numpy.ndarray,
        sigma: numpy.ndarray,
        bond_length: numpy.ndarray,
        quadrupole: numpy.ndarray,
    ):
        """Computes the critical temperature of the two-center
        Lennard-Jones model for a given set of model parameters.

        Parameters
        ----------
        epsilon: numpy.numpy.ndarray
            The epsilon parameter in units of K.
        sigma: numpy.numpy.ndarray
            The sigma parameter in units of nm.
        bond_length: numpy.numpy.ndarray
            The bond-length parameter in units of nm.
        quadrupole: numpy.numpy.ndarray
            The quadrupole parameter in units of Debye * nm.

        Returns
        -------
        numpy.numpy.ndarray
            The critical temperature in units of K.
        """
        quadrupole_star_sqr = (quadrupole * self._reduced_D_to_sqrt_J_m3) ** 2 / (
            epsilon * self._reduced_boltzmann * sigma ** 5
        )
        bond_length_star = bond_length / sigma

        critical_temperature_star = self._critical_temperature_star(
            quadrupole_star_sqr, bond_length_star
        )
        critical_temperature = critical_temperature_star * epsilon

        return critical_temperature

    def _critical_density_star(
        self, quadrupole_star: numpy.ndarray, bond_length_star: numpy.ndarray
    ) -> numpy.ndarray:
        """Computes the reduced critical density of the two-center
        Lennard-Jones model for a given set of model parameters.

        Parameters
        ----------
        quadrupole_star: float
            The reduced quadrupole parameter.
        bond_length_star: float
            The reduced bond-length parameter

        Returns
        -------
        float
            The reduced critical density.
        """

        q = quadrupole_star
        l = bond_length_star

        b = self.density_star_parameters

        rho_c_star = (
            1 * b[0]
            + q ** 2 * b[1]
            + q ** 3 * b[2]
            + l ** 2 / (0.11 + l ** 2) * b[3]
            + l ** 5 / (0.11 + l ** 5) * b[4]
            + l ** 2 * q ** 2 / (0.11 + l ** 2) * b[5]
            + l ** 5 * q ** 2 / (0.11 + l ** 5) * b[6]
            + l ** 2 * q ** 3 / (0.11 + l ** 2) * b[7]
            + l ** 5 * q ** 3 / (0.11 + l ** 5) * b[8]
        )

        return rho_c_star

    def _critical_density(
        self,
        epsilon: numpy.ndarray,
        sigma: numpy.ndarray,
        bond_length: numpy.ndarray,
        quadrupole: numpy.ndarray,
    ):
        """Computes the critical density of the two-center Lennard-Jones
        model for a given set of model parameters.

        Parameters
        ----------
        epsilon: numpy.numpy.ndarray
            The epsilon parameter in units of K.
        sigma: numpy.numpy.ndarray
            The sigma parameter in units of nm.
        bond_length: numpy.numpy.ndarray
            The bond-length parameter in units of nm.
        quadrupole: numpy.numpy.ndarray
            The quadrupole parameter in units of Debye * nm.

        Returns
        -------
        numpy.numpy.ndarray
            The evaluated densities in units of kg / m3.
        """

        molecular_weight = self.molecular_weight

        quadrupole_star_sqr = (quadrupole * self._reduced_D_to_sqrt_J_m3) ** 2 / (
            epsilon * self._reduced_boltzmann * sigma ** 5
        )
        bond_length_star = bond_length / sigma

        rho_star = self._critical_density_star(quadrupole_star_sqr, bond_length_star)
        rho = rho_star * molecular_weight / sigma ** 3 / 6.02214 * 10.0
        return rho  # [kg/m3]

    def _liquid_density_star(
        self,
        temperature_star: numpy.ndarray,
        quadrupole_star: numpy.ndarray,
        bond_length_star: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the reduced liquid density of the two-center
        Lennard-Jones model for a given set of model parameters over
        a specified range of temperatures.

        Parameters
        ----------
        temperature_star: numpy.numpy.ndarray
            The reduced temperatures to evaluate the reduced density at.
        quadrupole_star: float
            The reduced quadrupole parameter.
        bond_length_star: float
            The reduced bond-length parameter

        Returns
        -------
        numpy.numpy.ndarray
            The reduced density.
        """

        _b_C1, _b_C2, _b_C3 = (
            self._b_C1,
            self._b_C2_L,
            self._b_C3_L,
        )

        t_c_star = self._critical_temperature_star(quadrupole_star, bond_length_star)
        rho_c_star = self._critical_density_star(quadrupole_star, bond_length_star)

        tau = t_c_star - temperature_star

        if autograd.numpy.all(tau > 0):

            coefficient_1 = self._correlation_function_1(
                quadrupole_star, bond_length_star, _b_C1
            )
            coefficient_2 = self._correlation_function_2(
                quadrupole_star, bond_length_star, _b_C2
            )
            coefficient_3 = self._correlation_function_3(
                quadrupole_star, bond_length_star, _b_C3
            )

            x_0 = 1.0 * rho_c_star
            x_1 = tau ** (1.0 / 3.0) * coefficient_1
            x_2 = tau * coefficient_2
            x_3 = tau ** (3.0 / 2.0) * coefficient_3

            rho_star = x_0 + x_1 + x_2 + x_3

        else:
            rho_star = numpy.empty(temperature_star.shape) * numpy.nan

        return rho_star

    def _liquid_density(
        self,
        epsilon: numpy.ndarray,
        sigma: numpy.ndarray,
        bond_length: numpy.ndarray,
        quadrupole: numpy.ndarray,
        temperature: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the liquid density of the two-center Lennard-Jones
        model for a given set of model parameters over a specified range
        of temperatures.

        Parameters
        ----------
        epsilon: numpy.numpy.ndarray
            The values of epsilon to calculate the property at in
            units of K, the sigma parameter in units of nm.
        sigma: numpy.numpy.ndarray
            The values of epsilon to calculate the property at in
            units of of nm.
        bond_length: numpy.numpy.ndarray
            The values of the bond-length to calculate the property at
            in units of of nm.
        quadrupole: numpy.numpy.ndarray
            The values of the bond-length to calculate the property at.
        temperature: numpy.numpy.ndarray
            The temperatures to evaluate the density at in units of K.

        Returns
        -------
        numpy.numpy.ndarray
            The evaluated densities in units of kg / m3.
        """

        molecular_weight = self.molecular_weight

        # Note that epsilon is defined as epsilon/kB
        temperature_star = temperature / epsilon

        quadrupole_star_sqr = (quadrupole * self._reduced_D_to_sqrt_J_m3) ** 2 / (
            epsilon * self._reduced_boltzmann * sigma ** 5
        )
        bond_length_star = bond_length / sigma

        rho_star = self._liquid_density_star(
            temperature_star, quadrupole_star_sqr, bond_length_star
        )
        rho = rho_star * molecular_weight / sigma ** 3 / 6.02214 * 10.0
        return rho  # [kg/m3]

    def _vapor_pressure_star(
        self,
        temperature_star: numpy.ndarray,
        quadrupole_star: numpy.ndarray,
        bond_length_star: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the reduced saturation pressure of the two-center
        Lennard-Jones model for a given set of model parameters over
        a specified range of temperatures.

        Parameters
        ----------
        temperature_star: numpy.numpy.ndarray
            The reduced temperatures to evaluate the reduced density at.
        quadrupole_star: float
            The reduced quadrupole parameter.
        bond_length_star: float
            The reduced bond-length parameter

        Returns
        -------
        numpy.numpy.ndarray
            The reduced saturation pressures.
        """

        _b_c1, _b_c2, _b_c3 = self._b_c1, self._b_c2, self._b_c3

        q = quadrupole_star
        l = bond_length_star

        c1 = (
            1.0 * _b_c1[0]
            + q ** 2 * _b_c1[1]
            + q ** 3 * _b_c1[2]
            + l ** 2 / (l ** 2 + 0.75) * _b_c1[3]
            + l ** 3 / (l ** 3 + 0.75) * _b_c1[4]
            + l ** 2 * q ** 2 / (l ** 2 + 0.75) * _b_c1[5]
            + l ** 3 * q ** 2 / (l ** 3 + 0.75) * _b_c1[6]
            + l ** 2 * q ** 3 / (l ** 2 + 0.75) * _b_c1[7]
            + l ** 3 * q ** 3 / (l ** 3 + 0.75) * _b_c1[8]
        )
        c2 = (
            1.0 * _b_c2[0]
            + q ** 2 * _b_c2[1]
            + q ** 3 * _b_c2[2]
            + l ** 2 / (l + 0.75) ** 2 * _b_c2[3]
            + l ** 3 / (l + 0.75) ** 3 * _b_c2[4]
            + l ** 2 * q ** 2 / (l + 0.75) ** 2 * _b_c2[5]
            + l ** 3 * q ** 2 / (l + 0.75) ** 3 * _b_c2[6]
            + l ** 2 * q ** 3 / (l + 0.75) ** 2 * _b_c2[7]
            + l ** 3 * q ** 3 / (l + 0.75) ** 3 * _b_c2[8]
        )
        c3 = q ** 2 * _b_c3[0] + q ** 5 * _b_c3[1] + l ** 0.5 * _b_c3[2]

        vapor_pressure_star = autograd.numpy.exp(
            c1 + c2 / temperature_star + c3 / (temperature_star ** 4)
        )
        return vapor_pressure_star

    def _vapor_pressure(
        self,
        epsilon: numpy.ndarray,
        sigma: numpy.ndarray,
        bond_length: numpy.ndarray,
        quadrupole: numpy.ndarray,
        temperature: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the saturation pressure of the two-center Lennard-Jones model
        for a given set of model parameters over a specified range of
        temperatures.

        Parameters
        ----------
        epsilon: numpy.numpy.ndarray
            The values of epsilon to calculate the property at in
            units of K, the sigma parameter in units of nm.
        sigma: numpy.numpy.ndarray
            The values of epsilon to calculate the property at in
            units of of nm.
        bond_length: numpy.numpy.ndarray
            The values of the bond-length to calculate the property at
            in units of of nm.
        quadrupole: numpy.numpy.ndarray
            The values of the bond-length to calculate the property at.
        temperature: numpy.numpy.ndarray
            The temperatures to evaluate the density at in units of K.
        Returns
        -------
        numpy.numpy.ndarray
            The evaluated saturation pressures in units of kPa
        """

        temperature_star = temperature / epsilon

        quadrupole_star_sqr = (quadrupole * self._reduced_D_to_sqrt_J_m3) ** 2 / (
            epsilon * self._reduced_boltzmann * sigma ** 5
        )
        bond_length_star = bond_length / sigma

        vapor_pressure_star = self._vapor_pressure_star(
            temperature_star, quadrupole_star_sqr, bond_length_star
        )
        vapor_pressure = (
            vapor_pressure_star * epsilon / sigma ** 3 * self._reduced_boltzmann * 1.0e1
        )
        return vapor_pressure  # [kPa]

    def _surface_tension_star(
        self,
        temperature_star: numpy.ndarray,
        quadrupole_star: numpy.ndarray,
        bond_length_star: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the reduced surface tension of the two-center
        Lennard-Jones model for a given set of model parameters over
        a specified range of temperatures.

        Parameters
        ----------
        temperature_star: numpy.numpy.ndarray
            The reduced temperatures to evaluate the reduced density at.
        quadrupole_star: float
            The reduced quadrupole parameter.
        bond_length_star: float
            The reduced bond-length parameter

        Returns
        -------
        numpy.numpy.ndarray
            The reduced surface tensions.
        """
        _B = self._B

        t_c_star = self._critical_temperature_star(quadrupole_star, bond_length_star)
        _a_correlation = self._a_correlation_function(quadrupole_star, bond_length_star)

        if any(temperature_star / t_c_star > 1.0):
            return numpy.empty(temperature_star.shape) * numpy.nan

        surface_tension_star = (
            _a_correlation * (1.0 - (temperature_star / t_c_star)) ** _B
        )
        return surface_tension_star

    def _surface_tension(
        self,
        epsilon: numpy.ndarray,
        sigma: numpy.ndarray,
        bond_length: numpy.ndarray,
        quadrupole: numpy.ndarray,
        temperature: numpy.ndarray,
    ) -> numpy.ndarray:
        """Computes the surface tension of the two-center Lennard-Jones model
        for a given set of model parameters over a specified range of
        temperatures.

        Parameters
        ----------
        epsilon: numpy.numpy.ndarray
            The values of epsilon to calculate the property at in
            units of K, the sigma parameter in units of nm.
        sigma: numpy.numpy.ndarray
            The values of epsilon to calculate the property at in
            units of of nm.
        bond_length: numpy.numpy.ndarray
            The values of the bond-length to calculate the property at
            in units of of nm.
        quadrupole: numpy.numpy.ndarray
            The values of the bond-length to calculate the property at.
        temperature: numpy.numpy.ndarray
            The temperatures to evaluate the density at in units of K.

        Returns
        -------
        numpy.numpy.ndarray
            The evaluated surface tensions in units of J / m^2
        """
        # Note that epsilon is defined as epsilon/kB
        temperature_star = temperature / epsilon

        quadrupole_star_sqr = (quadrupole * self._reduced_D_to_sqrt_J_m3) ** 2 / (
            epsilon * self._reduced_boltzmann * sigma ** 5
        )

        bond_length_star = bond_length / sigma

        surface_tension_star = self._surface_tension_star(
            temperature_star, quadrupole_star_sqr, bond_length_star
        )
        surface_tension = (
            surface_tension_star
            * epsilon
            / sigma ** 2
            * self._reduced_boltzmann
            * 1.0e-5
        )
        return surface_tension  # [J/m2]

    def _evaluate_analytical(
        self,
        property_type: str,
        temperatures: numpy.ndarray,
        parameters: Dict[str, numpy.ndarray],
        compute_gradients: bool,
    ) -> Tuple[numpy.ndarray, Optional[Dict[str, numpy.ndarray]]]:

        property_functions = {
            "liquid_density": self._liquid_density,
            "vapor_pressure": self._vapor_pressure,
            "surface_tension": self._surface_tension,
        }

        values = property_functions[property_type](
            **parameters, temperature=temperatures
        ).reshape(-1, 1)

        gradients = None

        if compute_gradients:

            gradient_functions = {
                x: grad(y, argnum=(0, 1, 2, 3)) for x, y in property_functions.items()
            }

            gradients = gradient_functions[property_type](
                parameters["epsilon"],
                parameters["sigma"],
                parameters["bond_length"],
                parameters["quadrupole"],
                temperature=temperatures,
            )

            gradients = {
                "epsilon": numpy.ravel(gradients[0]),
                "sigma": numpy.ravel(gradients[1]),
                "bond_length": numpy.ravel(gradients[2]),
                "quadrupole": numpy.ravel(gradients[3]),
            }

        return values, gradients

    def evaluate(
        self,
        targets: List[StollWerthTarget],
        parameters: Dict[str, numpy.ndarray],
        compute_gradients: bool,
    ) -> Tuple[
        List[numpy.ndarray],
        List[numpy.ndarray],
        Optional[List[Dict[str, numpy.ndarray]]],
    ]:

        values = []
        uncertainties = []

        gradients = []

        for target in targets:

            assert isinstance(target, StollWerthTarget)

            noiseless_values, noiseless_gradients = self._evaluate_analytical(
                target.property_type, target.temperatures, parameters, compute_gradients
            )

            property_uncertainties = noiseless_values * self._fractional_noise
            noise = numpy.random.randn(*property_uncertainties.shape)

            property_values = noiseless_values + noise * property_uncertainties

            property_gradients = {
                x: noiseless_gradients[x]
                + noise * self._fractional_noise * noiseless_gradients[x]
                for x in noiseless_gradients
            }

            values.append(property_values)
            uncertainties.append(property_uncertainties)

            gradients.append(property_gradients)

        return values, uncertainties, gradients if compute_gradients else None
