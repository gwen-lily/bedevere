# package:  bedevere
# website:  github.com/noahgill409/bedevere
# email:    noahgill409@gmail.com
###############################################################################

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto, unique
from typing import Iterator
import numpy as np
from itertools import product

from bedevere.data import ARITHMETIC_PRECISION

###############################################################################
# enums 'n such
###############################################################################


@unique
class StochasticType(Enum):
    RIGHT = auto()
    LEFT = auto()
    DOUBLY = auto()


###############################################################################
# main classes
###############################################################################


@dataclass(frozen=True)
class Distribution:
    """A distribution has values with associated weights.

    Raises:
        ValueError: Raised if values and weights have different shapes.
        TypeError: Raised if weights are not floats.
    """

    values: np.ndarray | np.ndarray[np.ndarray]
    weights: np.ndarray

    def __post_init__(self):
        # error handling

        if self.dims == 1:
            if not isinstance(self.values, np.ndarray):
                raise TypeError(self.values)

            if self.values.shape != self.weights.shape:
                raise ValueError(self.values.shape, self.weights.shape)

            if self.weights.dtype.type is not np.float64:
                raise TypeError(self.weights.dtype)

        elif self.dims == 2:
            m, n = self.shape
            if m == 1 ^ n == 1:
                raise TypeError(f"{weights=} should be 1 dimensional")

        for i in range(self.dims):
            vals_len = self.values[i].size
            axis_len = self.weights.shape[i]

            if vals_len != axis_len:
                raise ValueError(vals_len, axis_len)

    # properties

    @property
    def dtype(self) -> np.dtype:
        if self.dims == 1:
            return self.values.dtype

        data_types = np.array((val.dtype for val in self.values))
        _data_type = data_types[0]

        if any(data_type is not _data_type for data_type in data_types):
            raise ValueError(data_types)

        return _data_type

    @property
    def dtypes(self) -> np.ndarray[np.dtype]:
        if self.dims == 1:
            return np.array([self.values.dtype])

        return np.array(v.dtype for v in self.values)

    @property
    def shape(self) -> np._ShapeType:
        return self.weights.shape

    @property
    def size(self) -> int:
        return self.weights.size

    @property
    def dims(self) -> int:
        return len(self.shape)

    # operations

    def reshape(self, shape: int | tuple[int]) -> Distribution:
        new_vals = self.values.reshape(shape)
        new_weights = self.weights.reshape
        return self.__class__(new_vals, new_weights)

    def __iter__(self):
        return self

    def __next__(self) -> Iterator[tuple[tuple, np.ndarray]]:
        """Yields a tuple of values as well as their weight.

        Yields:
            Iterator[tuple[tuple, np.ndarray]]: tuple[values], weight

        """
        values_container = self.values if self.dims > 1 else np.ndarray([self.values])

        values_container = self.values if self.dims > 1 else np.ndarray([self.values])

        assert isinstance(values_container, np.ndarray)

        for indices in product(*(range(dim) for dim in self.shape)):
            values_i = tuple(
                v[i] for v, i in zip(values_container, indices, strict=True)
            )
            weights_i = self.weights[indices]
            yield values_i, weights_i

    def __len__(self) -> int:
        return self.size


@dataclass(frozen=True)
class StochasticDistribution(Distribution):
    stochastic_type: StochasticType = field(kw_only=True, default=StochasticType.RIGHT)
    precision: float = field(kw_only=True, default=ARITHMETIC_PRECISION)

    def __post_init__(self):
        super().__post_init__()

        def assert_stochastic(x: np.ndarray, /) -> bool:
            if np.absolute(x.sum(axis=0) - 1) >= self.precision:
                raise ValueError(x)

        match self.shape:
            case [_]:
                assert_stochastic(self.weights)

            case [m, n]:
                st = self.stochastic_type

                if st in (StochasticType.RIGHT, StochasticType.DOUBLY):
                    for row in self.weights:
                        assert_stochastic(row)

                elif st in (StochasticType.LEFT, StochasticType.DOUBLY):
                    for col in [self.weights[:, j] for j in range(n)]:
                        assert_stochastic(col)

            case _:
                raise NotImplementedError


###############################################################################
# helper functions
###############################################################################


def generate_stochastic_square_matrix(
    n: int, stochastic_type: StochasticType
) -> np.ndarray:

    match n:
        case [1]:
            x = np.array([1])
        case [2]:
            row = np.random.dirichlet(np.ones(2))
            x = np.array([row, row[::-1]])
        case [3]:
            # create matrix and generate first column & row
            a = np.empty(shape=(3,) * 2, dtype=np.float64)
            a[:, 0] = np.random.dirichlet(np.ones(3))
            # a[0, 1:] = (1 - a[0]) * np.random.dirichlet(np.ones(2))
            a[1, 0] = np.random.random()

            # solve system of 4 equations with 4 unknowns

            # # # # # # # # # # # # # # # # # # #
            #                                   #
            #          A  *    x  =        b    #
            #                                   #
            # # # # # # # # # # # # # # # # # # #
            #                                   #
            #    1 1 0 0    x_11    1 - a_10    #
            #    1 0 1 0    x_12    1 - a_01    #
            #    0 1 0 1    x_21    1 - a_02    #
            #    0 0 1 1    x_22    1 - a_20    #
            #                                   #
            # # # # # # # # # # # # # # # # # # #

            # easy mode
            # fill row 0:
            # fill x10 with v < (1 - max(x0_))

    return x


if __name__ == "__main__":
    test_size = 5

    values = np.arange(5)
    weights = (1 / 5) * np.ones(5)

    sd = StochasticDistribution(values, weights)

    try:
        bad_weights = np.random.random(5)
        sd = StochasticDistribution(values, bad_weights)
    except ValueError as exc:
        print(f"not stochastic: {bad_weights}")

    dim = 2
    values = np.random.random(dim**2).reshape((dim,) * 2)
    weights = np.asarray([[0.1, 0.9], [0.2, 0.8]])

    d = Distribution(values, weights)
    sd = StochasticDistribution(values, weights, stochastic_type=StochasticType.RIGHT)

    values = values.T
    weights = weights.T

    sd = StochasticDistribution(values, weights, stochastic_type=StochasticType.LEFT)

    values = np.random.random(3**2).reshape(3, 3)
    weights = np.asarray([[0.2, 0.3, 0.5], [0.4, 0.5, 0.1], [0.4, 0.2, 0.4]])

    sd = StochasticDistribution(values, weights, stochastic_type=StochasticType.DOUBLY)
