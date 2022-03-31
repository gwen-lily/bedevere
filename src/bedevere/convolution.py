###############################################################################
# package:  bedevere                                                          #
# website:  github.com/noahgill409/bedevere                                   #
# email:    noahgill409@gmail.com                                             #
###############################################################################

import math
import numpy as np

from bedevere.distribution import StochasticDistribution

###############################################################################
# exceptions                                                                  #
###############################################################################


class PositiveIntegerError(ValueError):
    """Raise if a value must be a positive integer."""

###############################################################################
# main functions                                                              #
###############################################################################


def convolution(*args, **kwargs) -> StochasticDistribution:
    """Find a convolution of one or more stochastic distributions.

    This function takes a wide variety of input parameters and returns a
    stochastic distribution. Options include:

    Parameters
    ----------
    Generic convolution.
        x : StochasticDistribution
        y : StochasticDistribution

    Convolution of x, n times.
        x : StochasticDistribution
        n : int

    Convolution of discrete uniform variable X with range [0, k], n times.
        k : int
        n : int

    Convolution of discrete uniform variable X with range [lo, hi], n times.
        X : range
        n : int

    Convolution of discrete uniform variable X with range [lo, hi], n times.
        lo: int
        hi: int
        n : int

    Returns
    -------
    StochasticDistribution

    Raises
    ------
    ValueError
        Raised for higher order (>2D) calls or mismatched axes.
    ValueError
        Raised under arithmetic weirdness.
    """
    match args:
        case [u, v]:
            if all(isinstance(i, StochasticDistribution) for i in (u, v)):
                x, y = u, v
                return _xy_convolution(x, y, **kwargs)
            elif isinstance(u, StochasticDistribution) and isinstance(v, int):
                x, n = u, v
                return _n_convolution(x, n)
            elif all(isinstance(i, int) for i in (u, v)):
                k, n = u, v
                return _n_convolution_discrete_uniform(k, n)
            elif isinstance(u, range) and isinstance(v, int):
                x_range, n = u, v
                return _n_convolution_discrete_uniform(x_range, n)
            else:
                raise TypeError(u, v)

        case [lo, hi, v]:
            if all(isinstance(i, int) for i in [lo, hi, v]):
                n = v
                return _n_convolution_discrete_uniform(lo, hi, n)
        case _:
            raise TypeError(*args)


# helper functions & specific cases ###########################################

def _xy_convolution(
        x: StochasticDistribution,
        y: StochasticDistribution, /) -> StochasticDistribution:
    """Find a distribution representing the sum of two distributions.

    Parameters
    ----------
    x : Distribution
        A stochastic distribution.
    y : Distribution
        A stochastic distribution.

    Returns
    -------
    StochasticDistribution
        The distribution of the sum of x and y.

    Raises
    ------
    ValueError
        Raised for higher order (>2D) calls or mismatched axes.
    ValueError
        Raised under arithmetic weirdness.
    """
    assert all(isinstance(i, StochasticDistribution) for i in (x, y))

    if (x.dims != y.dims) or x.dims > 2:
        raise ValueError(x.dims, y.dims)

    min_val = sum(min(x.values), min(y.values))
    max_val = sum(max(x.values), max(y.values))

    z_values = np.arange(min_val, max_val + 1)
    z_weights = np.zeros(shape=z_values.shape)

    for z_idx, z_val in enumerate(z_values):
        for x_val, x_wt in x:
            y_val = z_val - x_val
            y_val_idcs = np.where(y == y_val)[0]

            if y_val_idcs.size == 0:
                continue
            elif y_val_idcs.size > 1:
                raise ValueError(
                    y_val_idcs.size,
                )

            y_idx = y_val_idcs[0]
            y_wt = y.weights[y_idx]

            new_wt = x_wt * y_wt
            z_weights[z_idx] = new_wt

    z = StochasticDistribution(z_values, z_weights)
    return z


def _n_convolution(x: StochasticDistribution, n: int) -> \
        StochasticDistribution:
    """Find the sum of distribution x, n times.

    Returns
    -------
    StochasticDistribution

    Raises
    ------
    PositiveIntegerError
        Raised if n is not positive.
    """
    if n <= 0:
        raise PositiveIntegerError(n)
    elif n == 1:
        val = x
    elif n == 2:
        val = _xy_convolution(x, x)
    else:
        val = _xy_convolution(x, _n_convolution(x, n - 1))

    return val


def wumbo(n: int, y: int, k: int) -> float:
    """Find a specific term as descrbied by Caiado & Rathie. I call it wumbo.

    Wumbo is useful for multinomial sums. I've forgotten its original name,
    or if it even had one. I would consult the documentation, however, the
    link is dead (http://community.dur.ac.uk/c.c.d.s.caiado/multinomial.pdf).

    Parameters
    ----------
    n : int
        The number of instances of the discrete uniform random variable X being
        summed.
    y : int
        The value of the discrete uniform random variable Y.
    k : int
        The high bound of X.

    Returns
    -------
    float
    """
    def factorial_gamma(__x: int, /) -> int | float:
        """Perform a different transformation on x depending on the value of x.

        Parameters
        ----------
        __x : int
            The numerator of wumbo.

        Returns
        -------
        int | float
        """
        if __x < 1:
            __y = (-1) ** p * math.gamma(__x)
        else:
            __y = (-1) ** p * math.factorial(__x - 1)

        return __y

    wumbo = 0   # though wumbo starts at just zero, look at him go!

    for p in range(0, math.floor(y / (k + 1)) + 1):
        a = n + y - p * (k + 1)
        numerator = factorial_gamma(a)

        b = n - p + 1
        c = y - p * (k + 1) + 1

        d1 = factorial_gamma(b)
        d2 = factorial_gamma(c)
        denominator = math.factorial(p + 1) * d1 * d2

        wumbo += numerator / denominator

    return wumbo


def _n_convolution_discrete_uniform(*args: int) -> StochasticDistribution:
    """Find the sum of a discrete uniform variable X, n times.

    Given a discrete uniform variable X with values (j, j+1, ... j+k), return
    the distribution of the sum of n instances with values (nj, nj+1, nj+2,
    ..., n*(j+k)). If j = 0, this simplifies to X with values (0, 1, ... k)
    and a returned distribution with range (0, 1, ... n*k).

    Returns
    -------
    StochasticDistribution

    Raises
    ------
    TypeError
        Raised if the arguments don't conform to any known scheme.
    PositiveIntegerError
        Raised if n is not positive.
    """
    offset = None

    # structural pattern matching and error handling
    match args:
        case [u, v]:
            if all(isinstance(i, int) for i in (u, v)):
                k, n = u, v
            elif isinstance(u, range) and isinstance(v, int):
                lo, hi = min(u), max(u)
                k = hi - lo
                n = v
                offset = n * lo
            else:
                raise TypeError(u, v)

        case [lo, hi, v]:
            k = hi - lo
            n = v
            offset = n * lo
        case _:
            raise TypeError(*args)

    if n <= 0:
        raise PositiveIntegerError(n)

    y_values = np.arange(n * k + 1)
    weights = np.empty(shape=y_values.shape, dtype=float)

    for y in y_values:
        weights[y] = wumbo(n, y, k) * n * (k + 1) ** (-n)

    if offset is not None:
        for idx, val in y_values[:]:
            y_values[idx] = val + offset

    d = StochasticDistribution(y_values, weights)
    return d
