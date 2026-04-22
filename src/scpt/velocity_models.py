import numpy as np
import scipy
from numpy.typing import ArrayLike

from scpt.inversion_model import Layering


def linear(layering: Layering | ArrayLike, speed_top=150, speed_bottom=300, bottom=100):
    if isinstance(layering, Layering):
        layering = layering.midpoints
    speeds = scipy.interpolate.interp1d([0, bottom], [speed_top, speed_bottom])(
        layering
    )
    return speeds


def sinus(layering: Layering | ArrayLike, amplitude=10, mean=250, waves=1.5):
    if isinstance(layering, Layering):
        layering = layering.midpoints
    speeds = (
        np.sin(layering / (layering[-1] - layering[0]) * waves * np.pi * 2) * amplitude
        + mean
    )
    return speeds


def bedded(layering: Layering | ArrayLike, low=200, high=300, size=4):
    if isinstance(layering, Layering):
        layering = layering.midpoints

    layering = np.asarray(layering)

    # Determine bed index for each layer
    bed_index = np.arange(len(layering)) // size

    # Alternate between low and high
    speeds = np.where(bed_index % 2 == 0, low, high)

    return speeds
