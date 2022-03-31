###############################################################################
# package:  bedevere
# website:  github.com/noahgill409/bedevere
# email:    noahgill409@gmail.com
###############################################################################

import math
import numpy as np

from bedevere.distribution import StochasticDistribution

###############################################################################
# exceptions
###############################################################################


class PositiveIntegerError(ValueError):
    """Raise if a value must be a positive integer."""


###############################################################################
# main functions
###############################################################################


def convolution(*args, **kwargs) -> StochasticDistribution:
    """Return convolution of one or more StochasticDistributions.

    Arguments:
        Generic convolution:
            x (StochasticDistribution):
            y (StochasticDistribution):

        Convolution of x, n times:
            x (StochasticDistribution):
            n (int):

        Convolution of discrete uniform variable X, n times:
            k (int): High bound of X with range [0, 1, ... k] (inclusive)
            n (int):

        Convolution of discrete uniform variable X (with offset), n times:
            lo (int): Low bound of X with range [lo, lo+1, ... hi] (inclusive)
            hi (int): High bound of X with range [lo, lo+1, ... hi] (inclusive)


    Raises:
        TypeError: Raised if the provided *args and **kwargs match no known
        implementation.

    Returns:
        StochasticDistribution:
    """

    match args:
        case [u, v]:
            if all(isinstance(i, StochasticDistribution) for i in (u, v)):
                return _xy_convolution(u, v, **kwargs)
            elif isinstance(u, StochasticDistribution) and isinstance(v, int):
                return _n_convolution(u, v)
            if all(isinstance(i, int) for i in (u, v)):
                return _n_convolution_discrete_uniform(u, v)
            else:
                raise TypeError(u, v)

        case [(lo, hi), v]:
            if all(isinstance(i, int) for i in [lo, hi, v]):
                return _n_convolution_discrete_uniform((lo, hi), v)


# helper functions & specific cases ###########################################


def _xy_convolution(
    x: StochasticDistribution, y: StochasticDistribution, /, *, reshape: bool = True
) -> StochasticDistribution:
    """Return a distribution representing the sum of two distributions"""

    def reshape_one_d(__a: StochasticDistribution, /) -> StochasticDistribution:
        m, n = __a.shape

        if m == 1 ^ n == 1:
            __a = __a.reshape(max(m, n))
        else:
            raise ValueError

        return __a

    match x.dims, y.dims:
        case [1, 1]:
            pass
        case [2, 2]:
            if reshape:
                x = reshape_one_d(x)
                y = reshape_one_d(y)

        case [_x, _y]:
            raise ValueError(x.dims, y.dims)

    min_val = sum(min(x.values), min(y.values))
    max_val = sum(max(x.values), max(y.values))

    possible_value_space = np.arange(min_val, max_val + 1)
    new_weights = np.zeros(shape=possible_value_space.shape)
    y_values_list = list(y.values)

    new_values = []
    new_weights = []

    for new_val in possible_value_space:

        for x_val, x_wt in x:
            y_val = new_val - x_val

            try:
                y_idx = y_values_list.index(y_val)
                y_wt = y.weights[y_idx]
                new_wt = x_wt * y_wt

                try:
                    if new_values[-1] == new_val:
                        new_weights[-1] += new_wt
                    else:
                        new_values.append(new_val)
                        new_weights.append(new_wt)

                except IndexError:
                    new_values.append(new_val)
                    new_weights.append(new_wt)

            except ValueError:
                pass

    z = StochasticDistribution(*[np.asarray(a) for a in [new_values, new_weights]])
    return z


def _n_convolution(x: StochasticDistribution, n: int) -> StochasticDistribution:
    """Return a distribution representing the sum of n distributions x"""
    if n <= 0:
        raise PositiveIntegerError(n)
    elif n == 1:
        val = x
    elif n == 2:
        val = _xy_convolution(x, x)
    else:
        val = _xy_convolution(x, _n_convolution(x, n - 1))

    return val


def _discrete_uniform_sum_term(n: int, y: int, k: int) -> float:
    """Returns a very specific term as descrbied by Caiado & Rathie

    Link: http://community.dur.ac.uk/c.c.d.s.caiado/multinomial.pdf
    the link is dead (2022-03-30)
    """

    def factorial_gamma(__x: int, /) -> int | float:
        if __x < 1:
            __y = (-1) ** p * math.gamma(__x)
        else:
            __y = (-1) ** p * math.factorial(__x - 1)

        return __y

    itersum = 0

    for p in range(0, math.floor(y / (k + 1)) + 1):
        a = n + y - p * (k + 1)
        numerator = factorial_gamma(a)

        b = n - p + 1
        c = y - p * (k + 1) + 1

        d1 = factorial_gamma(b)
        d2 = factorial_gamma(c)
        denominator = math.factorial(p + 1) * d1 * d2

        itersum += numerator / denominator

    return itersum


def _n_convolution_discrete_uniform(*args: int) -> StochasticDistribution:
    """Returns a stochastic distribution representing the sum of n discrete uniform variables.

    Given a discrete uniform variable X with values (j, j+1, ... j+k), return the distribution of
    the sum of n instances with values (nj, nj+1, ... n*(j+k)). If j = 0, this simplifies to X0
    with values (0, 1, ... k) and yield (0, 1, ... n*k).

    Arguments:
        if j == 0:
            k: int
            n: int

        if j != 0:
            lo: int
            hi: int
            n: int

    Raises:
        TypeError: If the provided args don't match any scheme.
        PositiveIntegerError: If n is not positive.

    Returns:
        StochasticDistribution:
    """

    # structural pattern matching and error handling
    offset = None

    match args:
        case [k, n]:
            k = k
            n = n
        case [(lo, hi), n]:
            k = hi - lo
            offset = n * lo
        case _:
            raise TypeError(*args)

    if n <= 0:
        raise PositiveIntegerError(n)

    values = np.empty(shape=(n * k + 1,), dtype=int)
    weights = np.empty(shape=values.shape, dtype=float)

    for idx, y in range(n * k + 1):
        wt = n * (k + 1) ** (-n) * _discrete_uniform_sum_term(n, y, k)

        values[idx] = y
        weights[idx] = wt

    if offset is not None:
        for idx, val in values[:]:
            values[idx] = val + offset

    d = StochasticDistribution(values, weights)
    return d
