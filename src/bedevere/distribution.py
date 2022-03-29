###############################################################################
# package:  bedevere
# website:  github.com/noahgill409/bedevere
# email:    noahgill409@gmail.com
###############################################################################

# damage, stances, & styles ###################################################

# damage, stances, & styles ###################################################

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class Distribution:
    values: np.ndarray
    weights: np.ndarray[float]

    dtype: np.dtype = field(init=False)
    shape: tuple[int, int] = field(init=False)
    size: int = field(init=False)

    def __post_init__(self):
        if (vs := self.values.shape) != (ws := self.weights.shape):
            raise ValueError(vs, ws)

        for w in self.weights:
            if not isinstance(w, (float, np.float64)):
                raise TypeError(w)

        self.dtype = self.values.dtype
        self.shape = tuple(vs)
        self.size = self.values.size

    # operations

    def __iter__(self):
        return self

    def __next__(self):
        yield from zip(self.values, self.weights, strict=True)

    def __len__(self) -> int:
        return self.size
