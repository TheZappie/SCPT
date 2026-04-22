from itertools import pairwise
from typing import Iterable

import numpy as np
from numpy import ndarray
from obspy import Trace, Stream
from scipy.interpolate import UnivariateSpline
from uncertainties import UFloat


def is_unumpy_array(x):
    return (
        isinstance(x, ndarray)
        and x.dtype == object
        and x.size > 0
        and isinstance(x.flat[0], UFloat)
    )


def strictly_increasing(array: Iterable):
    return all(x < y for x, y in pairwise(array))


def strictly_decreasing(array: Iterable):
    return all(x > y for x, y in pairwise(array))


def strictly_monotonic(array: Iterable):
    return strictly_increasing(array) or strictly_decreasing(array)


def all_positive(s) -> bool:
    return np.all(s > 0)


def cluster_1d(data: list[float], max_distance: float) -> list[list[float]]:
    """
    Cluster 1D points such that each cluster's diameter <= max_distance.
    """
    data = np.asarray(data)
    if len(data) == 0:
        return []

    data = sorted(data)
    clusters = []
    current_cluster = [data[0]]
    min_val = data[0]

    for x in data[1:]:
        # Check if adding x violates the cluster diameter
        if x - min_val <= max_distance:
            current_cluster.append(x)
        else:
            # Finish old cluster and start new
            clusters.append(current_cluster)
            current_cluster = [x]
            min_val = x

    clusters.append(current_cluster)
    return clusters


def predict_best_component(stream: Stream):
    mean_power_x = np.mean([np.mean(tr.data**2) for tr in stream.select(component="X")])
    mean_power_y = np.mean([np.mean(tr.data**2) for tr in stream.select(component="Y")])
    if mean_power_x > mean_power_y:
        return "X"
    else:
        return "Y"


def fit_axial_angle_spline(
    x,
    angle_deg,
    s=0.0,
    k=3,
    wrap_range=180.0,
) -> ndarray:
    """
    Fit a smooth spline to axial (180°-periodic) angle data.

    Parameters
    ----------
    x : array_like
        Independent variable (e.g. depth or time), shape (N,)
    angle_deg : array_like
        Angle values in degrees, axial (θ ≡ θ + 180°), shape (N,)
    s : float, optional
        Smoothing factor for UnivariateSpline.
        Larger values -> smoother results.
    k : int, optional
        Spline order (default: cubic).
    wrap_range : float, optional
        Angular periodicity in degrees (default: 180).

    Returns
    -------
    angle_smooth_deg : ndarray
        Smoothed angle in degrees, wrapped to [0, wrap_range)
    spline_sin : UnivariateSpline
        Spline fitted to sin(2θ)
    spline_cos : UnivariateSpline
        Spline fitted to cos(2θ)
    """
    x = np.asarray(x)
    angle_deg = np.asarray(angle_deg)

    # Convert to radians and axial representation
    theta = np.deg2rad(angle_deg)
    twotheta = 2.0 * theta

    # Fit splines to sine and cosine components
    spline_sin = UnivariateSpline(x, np.sin(twotheta), s=s, k=k)
    spline_cos = UnivariateSpline(x, np.cos(twotheta), s=s, k=k)

    # Reconstruct smooth angle
    twotheta_smooth = np.arctan2(
        spline_sin(x),
        spline_cos(x),
    )

    theta_smooth = 0.5 * twotheta_smooth
    angle_smooth_deg = np.rad2deg(theta_smooth)

    # Wrap to desired range
    angle_smooth_deg = np.mod(angle_smooth_deg, wrap_range)

    return angle_smooth_deg


def detect_clipping(array):
    for arr in array:
        has_nonzero_flat(arr)
        pass
    return
    #     invalid = (data == data_format.max) ^ (data == data_format.min)


#
#     # limiting the detection to a subset of the data to not have a big impact on performance
#     invalid_coherent = ndimage.binary_opening(
#         invalid[: min(200, trace_count)], structure=np.ones((10, 1))
#     )
#     if np.any(invalid_coherent):


def has_nonzero_flat(x, run_length, tol=0.0):
    x = np.asarray(x)
    if len(x) < run_length:
        return False

    nz = np.abs(x) > tol
    eq = np.abs(np.diff(x)) <= tol

    mask = nz[:-1] & nz[1:] & eq

    run = 1
    for v in mask:
        if v:
            run = run + 1
        else:
            continue
        if run >= run_length:
            return True
    return False


def spherical_divergence_correction(trace: Trace):
    """Keeps amplitude at t<=0 constant."""
    times = trace.times()
    gain = np.array(times)
    idx = np.searchsorted(times, 0)
    # reference_amplitude = times[idx]
    gain[:idx] = 0
    trace.data *= gain
