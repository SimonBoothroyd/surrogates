import numpy
from pydantic import BaseModel, Field

from surrogates.utils.serialization import NumpyNDArray


class DataSet(BaseModel):
    """A data set which contains the molecular and thermophysical properties
    of a compound which may be represented as a two-center Lennard-Jones model.
    """

    class Config:

        json_encoders = {
            numpy.ndarray: lambda x: x.tolist(),
        }

    formula: str = Field(..., description="The chemical formula of the compound.")

    molecular_weight: float = Field(
        ..., description="The molecular weight of the compound (g/mol)."
    )
    bond_length: float = Field(
        ..., description="The bond length of the compound (angstrom)."
    )

    critical_temperature: float = Field(
        ..., description="The critical temperature of the compound (K)."
    )
    critical_temperature_std: float = Field(
        ..., description="The uncertainty in the critical temperature (K)."
    )

    surface_tensions: NumpyNDArray = Field(
        ...,
        description="The surface tension of the compound as a function of temperature."
        "This is a (3, N) array with rows of temperature (K), value (N / m) and"
        "the uncertainty in the value (N / m)",
    )
    vapor_pressures: NumpyNDArray = Field(
        ...,
        description="The vapor pressure of the compound as a function of temperature."
        "This is a (3, N) array with rows of temperature (K), value (kPa) and"
        "the uncertainty in the value (kPa)",
    )
    liquid_densities: NumpyNDArray = Field(
        ...,
        description="The liquid density of the compound as a function of temperature."
        "This is a (3, N) array with rows of temperature (K), value (kg / m^3) and"
        "the uncertainty in the value (kg / m^3)",
    )
