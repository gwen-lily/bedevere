"""Defines Distribution and StochasticDistribution.

###############################################################################
# package:  bedevere                                                          #
# website:  github.com/noahgill409/bedevere                                   #
# email:    noahgill409@gmail.com                                             #
###############################################################################

"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto, unique
from typing import Callable, Counter
from collections.abc import Iterator, Generator
import numpy as np
from numpy import generic, object_, float_
from numpy.typing import NDArray, DTypeLike
from itertools import product

from bedevere.math.data import ARITHMETIC_PRECISION

###############################################################################
# enums 'n such                                                               #
###############################################################################


@unique
class StochasticType(Enum):
    """Defines stochastic types for 2-D distributions (matrices).

    Right stochastic means that the row entries sum to one. Left stochastic
    means that column entries sum to one. Doubly stochastic is both left and
    right stochastic.
    """

    RIGHT = auto()
    LEFT = auto()
    DOUBLY = auto()

###############################################################################
# main classes                                                                #
###############################################################################


@dataclass(frozen=True)
class Distribution(ABC, Iterator):
    """A value-weight pairing for a N-dimensional stochastic process.

    N-dimensional stochastic processes may or may not exists, but that's not
    important right now. What is important is that markov chains are 2D, which
    is why this exists. Use it for markov chains.

    Returns
    -------
    StochasticDistribution
        A distribution has values with associated weights.

    Raises
    ------
    TypeError
        Raised if weights are not floats.
    ValueError
        Raised if values and weights have different shapes.
    """

    values: NDArray[generic] | NDArray[object_]
    weights: NDArray[float_]

    @abstractmethod
    @property
    def _is_proper_shape(self) -> bool:
        """Evaluate if the provided arguments match the use case.

        Returns
        -------
        bool
        """

    def __post_init__(self) -> None:
        """Perform validation on a new distribution.

        Raises
        ------
        TypeError

        ValueError
        """
        assert self._is_proper_shape

        if self.dims == 1:
            if not isinstance(self.values, np.ndarray):
                raise TypeError(self.values)

            if self.values.shape != self.weights.shape:
                raise ValueError(self.values.shape, self.weights.shape)

            if self.weights.dtype.type is not np.float64:
                raise TypeError(self.weights.dtype)

            return

        if self.is_pseudo_1d:
            raise TypeError(f"{self.weights=} should be 1 dimensional")

        for i in range(self.dims):
            vals_len = self.values[i].size
            axis_len = self.weights.shape[i]

            if vals_len != axis_len:
                raise ValueError(vals_len, axis_len)

    # properties

    @property
    def dtype(self) -> DTypeLike:
        """Return the dtype of the distribution's values.

        Returns
        -------
        np.dtype

        Raises
        ------
        ValueError
            Raised if the dtypes of all values axes are not homogenous.
        """
        if self.dims == 1:
            return self.values.dtype

        def val_generator() -> DTypeLike:
            for val in self.values:
                assert isinstance(val, np.ndarray)
                __dtype: DTypeLike = val.dtype
                assert isinstance(__dtype, DTypeLike)

                yield __dtype

        data_types: NDArray[object_] = np.array(val_generator, dtype=object)
        _data_type = data_types[0]

        if any(data_type is not _data_type for data_type in data_types):
            raise ValueError(data_types)

        return _data_type

    @property
    def dtypes(self) -> NDArray[object_]:
        """Return the dtype iff all value members have the same dtype.

        For distributions of greater order than 1, this property is given if
        and only if all value axes have the same dtype.

        Returns
        -------
        np.ndarray[np.dtype]
        """
        if self.dims == 1:
            return np.array([self.values.dtype])

        return np.array(v.dtype for v in self.values)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape.

        Returns
        -------
        tuple[int, ...]
        """
        return self.weights.shape

    @property
    def size(self) -> int:
        """Return the size.

        Returns
        -------
        int
        """
        return self.weights.size

    @property
    def dims(self) -> int:
        """Return the number of dimensions.

        Returns
        -------
        int
        """
        return len(self.shape)

    @property
    def is_pseudo_1d(self) -> bool:
        """Evaluate if a distribution is pseudo-1D.

        A pseudo-1D array has shape (x0, x1, x2, ...) with all xi equal to one
        except for a single i.

        Returns
        -------
        bool
        """
        c = Counter(self.shape)
        len_one_dims = c[1]
        return len_one_dims == (self.dims - 1)

    # operations

    def as_1d(self) -> "Distribution":
        """Reshape a psuedo-1D array as a proper 1D array.

        See Distribution.is_pseudo_1d for more information.

        Returns
        -------
        Distribution
        """
        assert self.is_pseudo_1d

        proper_axis = np.where(self.shape != 1)[0][0]

        _values = self.values[proper_axis]
        _weights = self.weights.reshape(self.shape[proper_axis])

        return self.__class__(_values, _weights)

    # dunder operations

    def __iter__(self):
        """Return self."""
        return self

    def __next__(self) -> Generator[
            tuple[tuple[NDArray[generic], float]], None, None]:
        """Yield a tuple of values, as well as their weight.

        Yields
        ------
        Iterator[tuple[np.ndarray], float]

        Raises
        ------
        StopIteration
        """
        if self.dims > 1:
            values_container = self.values
        else:
            values_container = np.ndarray([self.values])

        assert isinstance(values_container, np.ndarray)

        for indices in product(*(range(dim) for dim in self.shape)):
            zipped = zip(values_container, indices, strict=True)
            values_i: tuple = tuple(v[i] for v, i in zipped)
            weights_i: float = self.weights[indices]
            yield values_i, weights_i

        raise StopIteration

    def __len__(self) -> int:
        """Return self.size.

        Returns
        -------
        int
        """
        return self.size


@dataclass(frozen=True)
class StochasticDistribution(Distribution, ABC):
    """Associated values and weights which describe a stochastic process.

    A stochastic distribution may (for now) be a 1- or 2-D array whose
    elements describe a complete probability space. If the process is 2D,
    such as with markov chains, the stochastic type and precision may be
    specified.
    """

    stochastic_type: StochasticType = field(
        kw_only=True, default=StochasticType.RIGHT)
    precision: float = field(kw_only=True, default=ARITHMETIC_PRECISION)

    @abstractmethod
    @property
    def _is_stochastic(self) -> bool:
        """Evaluate if a given distribution is stochastic.

        Returns
        -------
        bool
        """

    def _is_stochastic_1d(self, __x: np.ndarray, /) -> bool:
        """Evaluate if an array is stochastic.

        This method is included in the abstract base class because all
        dimensional distributions have use of it.

        Parameters
        ----------
        __x : np.ndarray
            A one dimensional array.

        Returns
        -------
        bool
        """
        assert len(__x.shape) == 1
        return np.absolute(__x.sum(axis=0) - 1) < self.precision

    def __post_init__(self) -> None:
        """Initialize a distribution, then assert stochastic quality.

        Raises
        ------
        ValueError
            Raised if the distribution is not stochastic.
        NotImplementedError
            Raised if the distribution shape is not supported.
        """
        super().__post_init__()
        assert self._is_stochastic


class StochasticArray(StochasticDistribution):
    """A one dimensional stochastic distribution."""

    @property
    def _is_proper_shape(self) -> bool:
        assert self.dims == 1

    @property
    def _is_stochastic(self) -> bool:
        return self._is_stochastic_1d


class StochasticMatrix(StochasticDistribution):
    """A two dimensional stochastic distribution."""

    @property
    def _is_stochastic(self) -> bool:
        m, n = self.shape
        st = self.stochastic_type
        func = self._is_stochastic_1d
        right, left = None, None

        if st in (StochasticType.RIGHT, StochasticType.DOUBLY):
            right = all(func(r) for r in self.weights)

            if st is StochasticType.RIGHT:
                return right

        if st in (StochasticType.LEFT, StochasticType.DOUBLY):
            left = all(func(self.weights[:, j]) for j in range(n))

            if st is StochasticType.LEFT:
                return left

        # doubly
        assert right is not None
        assert left is not None
        return right and left

    # properties
    def _is_triangular(self, func: Callable) -> bool:
        m, n = self.shape

        if m != n:
            return False

        return self.weights == func(self.weights)

    def _main_diagonal_all_equal(self, val, /) -> bool:
        m, n = self.shape
        d = min(m, n)

        return all(self.weights[i, i] == val for i in range(d))

    @property
    def is_upper_triangular(self) -> bool:
        """Evaluate if the distribution is upper triangular.

        Returns
        -------
        bool
        """
        return self._is_triangular(np.triu)

    @property
    def is_lower_triangular(self) -> bool:
        """Evaluate if the distribution is lower triangular.

        Returns
        -------
        bool
        """
        return self._is_triangular(np.tril)


###############################################################################
# helper functions                                                            #
###############################################################################


def random_stochastic_square_matrix(n: int, stochastic_type: StochasticType) \
        -> np.ndarray:
    """Generate a random n by n matrix that obeys stochastic properties.

    Returns
    -------
    np.ndarray
    """
    match n:
        case [1]:
            x: NDArray[float_] = np.array([float(1)])
        case [2]:
            row = np.random.dirichlet(np.ones(2))
            x: NDArray[float_] = np.array([row, row[::-1]])
        case [3]:
            # # create matrix and generate first column & row
            # a = np.empty(shape=(3,) * 2, dtype=np.float64)
            # a[:, 0] = np.random.dirichlet(np.ones(3))
            # # a[0, 1:] = (1 - a[0]) * np.random.dirichlet(np.ones(2))
            # a[1, 0] = np.random.random()

            # # solve system of 4 equations with 4 unknowns

            # # # # # # # # # # # # # # # # # # # #
            # #                                   #
            # #          A  *    x  =        b    #
            # #                                   #
            # # # # # # # # # # # # # # # # # # # #
            # #                                   #
            # #    1 1 0 0    x_11    1 - a_10    #
            # #    1 0 1 0    x_12    1 - a_01    #
            # #    0 1 0 1    x_21    1 - a_02    #
            # #    0 0 1 1    x_22    1 - a_20    #
            # #                                   #
            # # # # # # # # # # # # # # # # # # # #

            # # easy mode
            # # fill row 0:
            # # fill x10 with v < (1 - max(x0_))
            raise NotImplementedError

    return x


if __name__ == "__main__":
    test_size = 5

    values = np.arange(5)
    weights = (1 / 5) * np.ones(5)

    sd = StochasticDistribution(values, weights)

    try:
        bad_weights = np.random.random(5)
        sd = StochasticDistribution(values, bad_weights)
    except ValueError:
        print(f"not stochastic: {bad_weights}")

    dim = 2
    values = np.random.random(dim**2).reshape((dim,) * 2)
    weights = np.asarray([[0.1, 0.9], [0.2, 0.8]])

    d = StochasticDistribution(values, weights)
    sd = StochasticDistribution(
        values, weights, stochastic_type=StochasticType.RIGHT)

    values = values.T
    weights = weights.T

    sd = StochasticDistribution(
        values, weights, stochastic_type=StochasticType.LEFT)

    values = np.random.random(3**2).reshape(3, 3)
    weights = np.asarray([[0.2, 0.3, 0.5], [0.4, 0.5, 0.1], [0.4, 0.2, 0.4]])

    sd = StochasticDistribution(
        values, weights, stochastic_type=StochasticType.DOUBLY)
