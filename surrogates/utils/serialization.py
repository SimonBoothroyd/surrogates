import numpy


class NumpyNDArray(numpy.ndarray):
    """A stub type which wraps around a `numpy.ndarray`, and enables pydantic
    serialization / deserialization support."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):

        if isinstance(value, numpy.ndarray):
            return value

        return numpy.array(value)
