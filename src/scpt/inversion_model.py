from dataclasses import dataclass
from itertools import pairwise
from typing import Self

import numpy as np
from numpy import ndarray

from scpt.processing import strictly_increasing


@dataclass
class Layering:
    bounds: ndarray

    def __post_init__(self):
        if len(self.bounds) < 2:
            raise ValueError("bounds must have at least 2 elements")
        if not strictly_increasing(self.bounds):
            raise ValueError(f"Layer model should be strictly increasing")

    @classmethod
    def linspace(cls, start, stop, n):
        return cls(np.linspace(start, stop, n))

    def get_layer_sizes(self):
        return np.diff(self.bounds)

    @property
    def n_layers(self) -> int:
        return len(self.bounds) - 1

    @property
    def midpoints(self) -> ndarray:
        return np.asarray([np.mean(pair) for pair in pairwise(self.bounds)])

    def extend_first_layer(self, value=0) -> Self:
        """Replace first boundary"""
        if self.bounds[1] <= value:
            raise ValueError("Value should be smaller than second element")
        data = np.concatenate(([value], self.bounds[1:]))
        return self.__class__(data)

    def prepend_layer(self, value=0) -> Self:
        """Add/prepend first boundary"""
        if self.bounds[0] <= value:
            raise ValueError("Value should be smaller than first element")
        data = np.concatenate(([value], self.bounds))
        return self.__class__(data)
