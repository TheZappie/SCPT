import bisect
import datetime
from typing import Optional

import numpy as np
from numpy import ndarray
from obspy import Stream, UTCDateTime, Trace
from scipy.signal import filtfilt, butter

from scpt.inversion_model import Layering
from scpt.organisation import (
    DelayedTrace,
    ScptSurvey,
    Station,
    DEPTH,
    DEPTH_ROUNDING,
    shot_identifier_by_attribute,
)
from scpt.ray_tracing import compute_travel_time, compute_ray_path_safe

RAY_PATH = "RayPath"


def fill_to(value: float, layer_bounds, layer_sizes=None) -> ndarray:
    """
    'Fills' an array up to given value. All filled cells are there sizes, the rest zero.
    If not matching exactly, there will be 1 element filled partly.

    Returns a vector of size len(layer_bounds) -1

    identical to: fill_between(layer_bounds[0], value, layer_bounds, layer_sizes)
    """
    # could use np.searchsorted as well
    if layer_sizes is None:
        layer_sizes = np.diff(layer_bounds)
    idx = bisect.bisect_left(layer_bounds, value)
    dat = np.zeros_like(layer_sizes)
    dat[:idx] = layer_sizes[:idx]
    remaining = value - layer_bounds[idx - 1]
    if remaining:
        dat[idx - 1] = remaining
    return dat


def arrival_times(
    depth, layering: Layering, slowness, source_offset=0, ray_traced=True
):
    try:
        iter(depth)
    except TypeError:
        result = arrival_times([depth], layering, slowness, source_offset)
        return result[0][0], result[1][0]

    depth = np.array(depth)
    sizes = layering.get_layer_sizes()
    if ray_traced:
        times = []
        paths = []
        for d in depth:
            times.append(
                compute_travel_time(source_offset, 0, 0, d, layering.bounds, slowness)
            )
            paths.append(
                compute_ray_path_safe(source_offset, 0, 0, d, layering.bounds, slowness)
            )
    else:
        path_lengths = np.hypot(depth, source_offset)
        times = [
            sum(fill_to(d, layer_bounds=layering.bounds, layer_sizes=sizes) * slowness)
            / d
            * path_length
            for d, path_length in zip(depth, path_lengths)
        ]
        paths = [0, source_offset], [depth, 0]
    return times, paths


def ricker(points, a):
    """Taken from deprecated scipy function"""
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = 1 - xsq / wsq
    gauss = np.exp(-xsq / (2 * wsq))
    total = A * mod * gauss
    return total


def ricker_at_delay(
    n_samples: int,
    *,
    delay: float,
    a: float,
    amplitude: float = 1.0,
    dt: float | None = None,
    wavelet_points: int | None = None,
) -> ndarray:
    """
    Create a 1D signal of length `n_samples` with a Ricker wavelet centered at `delay`.

    Parameters
    ----------
    n_samples : int
        Length of the output array.
    delay : float
        Center position of the wavelet.
        - If `dt is None`: interpreted in **samples** (can be float; rounded to nearest sample).
        - If `dt is not None`: interpreted in **seconds** and converted via `delay_samples = round(delay / dt)`.
    a : float
        Ricker width **in samples** (same `a` used by the provided `ricker(points, a)`).
        Larger `a` produces a broader wavelet (lower peak frequency).
    amplitude : float, default 1.0
        Amplitude multiplier applied to the Ricker wavelet.
    dt : float or None, default None
        Sampling interval in seconds. If provided, `delay` is assumed seconds.
    wavelet_points : int or None, default None
        Support (number of samples) of the Ricker wavelet kernel to be inserted.
        - If None, it's chosen automatically as the nearest odd integer to `max(21, 12*a)`.
          This gives ~±6*a support which is typically enough for negligible tails.
        - If provided and even, it will be bumped to the next odd integer to keep symmetry.

    Returns
    -------
    y : (n_samples,) ndarray
        Output array with the Ricker wavelet centered at `delay`.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if a <= 0:
        raise ValueError("a (Ricker width in samples) must be positive.")

    # Convert delay to samples (integer index for center)
    if dt is None:
        center = int(np.round(delay))
    else:
        if dt <= 0:
            raise ValueError("dt must be positive when provided.")
        center = int(np.round(delay / dt))

    # Auto-choose an odd support length if not provided
    if wavelet_points is None:
        # ensure sufficient support for tails; make odd for symmetry
        wavelet_points = int(np.round(max(21, 12 * a)))
    if wavelet_points % 2 == 0:
        wavelet_points += 1  # ensure odd

    # Build the kernel
    kernel = amplitude * ricker(points=wavelet_points, a=float(a))

    # Target output
    y = np.zeros(int(n_samples), dtype=float)

    # Placement indices
    half = wavelet_points // 2
    k_start = center - half
    k_end_excl = center + half + 1  # exclusive end (centered window)

    # Clip to output boundaries and compute overlap slices
    out_start = max(0, k_start)
    out_end = min(n_samples, k_end_excl)
    if out_start >= out_end:
        return y  # wavelet lies completely outside; return zeros

    ker_start = out_start - k_start
    ker_end = ker_start + (out_end - out_start)

    # Insert (add) the kernel segment
    y[out_start:out_end] += kernel[ker_start:ker_end]

    return y


def ricker_f0(points: int, f0: float, dt: float) -> ndarray:
    """
    Ricker (Mexican hat) wavelet specified by central/peak frequency f0 (Hz).

    Parameters
    ----------
    points : int
        Number of samples in the wavelet kernel (use an odd number for perfect symmetry).
    f0 : float
        Central (peak) frequency in Hz.
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    w : (points,) ndarray
        Ricker wavelet centered at 0 (i.e., t=0 at the middle sample).

    Notes
    -----
    Continuous-time form: w(t) = (1 - 2*(pi*f0*t)^2) * exp(-(pi*f0*t)^2).
    We discretize at t = (n - (points-1)/2) * dt, n = 0..points-1.
    """
    if points <= 0:
        raise ValueError("points must be positive.")
    if f0 <= 0:
        raise ValueError("f0 must be positive (Hz).")
    if dt <= 0:
        raise ValueError("dt must be positive (s).")

    n = np.arange(points)
    t = (n - (points - 1) / 2.0) * dt  # centered time axis
    a = np.pi * f0 * t  # dimensionless argument
    w = (1.0 - 2.0 * a**2) * np.exp(-(a**2))
    return w


def ricker_at_delay_f0(
    n_samples: int,
    *,
    delay: float,
    f0: float,
    dt: float,
    amplitude: float = 1.0,
    wavelet_points: Optional[int] = None,
) -> ndarray:
    """
    Create a 1D array with a Ricker wavelet (specified by central frequency f0) centered at `delay`.

    Parameters
    ----------
    n_samples : int
        Length of the output array.
    delay : float
        Center time (seconds) where the wavelet is placed. (Can be fractional; we round to nearest sample.)
    f0 : float
        Central (peak) frequency in Hz.
    dt : float
        Sampling interval in seconds.
    amplitude : float, default 1.0
        Scales the wavelet.
    wavelet_points : int or None, default None
        Support length of the wavelet kernel in samples (prefer an odd number for symmetry).
        If None, a reasonable default is chosen based on f0.

    Returns
    -------
    y : (n_samples,) ndarray
        Signal with the wavelet inserted at the requested delay.

    Implementation details
    ----------------------
    - The kernel is centered and then **cropped** to fit within [0, n_samples-1].
    - Default `wavelet_points` uses ~±4 cycles of the Gaussian envelope around the center,
      which is typically enough to capture the significant energy.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive (s).")
    if f0 <= 0:
        raise ValueError("f0 must be positive (Hz).")

    # Choose a default support if not provided:
    # A practical rule: ~8 cycles at f0 => duration ≈ 8/f0 seconds.
    # points ≈ duration / dt
    if wavelet_points is None:
        wavelet_points = int(np.round(max(21, 8.0 / f0 / dt)))
    # Make it odd for symmetry
    if wavelet_points % 2 == 0:
        wavelet_points += 1

    # Build kernel
    kernel = amplitude * ricker_f0(points=wavelet_points, f0=f0, dt=dt)

    # Output buffer
    y = np.zeros(n_samples, dtype=float)

    # Center index from delay (seconds -> samples)
    center = int(np.round(delay / dt))

    fractional = delay / dt - center
    kernel = delay_trace_phase_shift(kernel, fractional)

    half = wavelet_points // 2
    k_start = center - half
    k_end_excl = center + half + 1

    # Clip to output boundaries and compute overlap slices
    out_start = max(0, k_start)
    out_end = min(n_samples, k_end_excl)
    if out_start >= out_end:
        return y  # lies completely outside

    ker_start = out_start - k_start
    ker_end = ker_start + (out_end - out_start)

    y[out_start:out_end] += kernel[ker_start:ker_end]
    return y


def delay_trace_phase_shift(trace: ndarray, delay: float) -> Trace:
    """
    Apply an arbitrary (sub-sample) time delay to an ObsPy Trace using
    frequency-domain phase shifting.

    Parameters
    ----------
    delay : float
        Delay in seconds (positive = shift later in time)

    Returns
    -------
    obspy.Trace
        New Trace with delayed data
    """
    n = len(trace)

    # FFT (real-valued trace)
    spectrum = np.fft.rfft(trace)

    # Frequency axis
    freqs = np.fft.rfftfreq(n, d=1)

    # Phase shift
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay)
    spectrum_shifted = spectrum * phase_shift

    # Back to time domain
    delayed_data = np.fft.irfft(spectrum_shifted, n=n)

    return delayed_data


def ricker_trace_at_delay(*args, dt, **kwargs):
    arr = ricker_at_delay_f0(*args, dt=dt, **kwargs)
    # from obspy.core.trace import Stats
    # stats = Stats
    trace = DelayedTrace.create(arr, header={"delta": dt}, delay=0)
    return trace


def band_limited_noise(n_samples, dt, fmin, fmax, noise_std, order=4):
    fs = 1.0 / dt
    nyq = 0.5 * fs

    pad = int(2 / dt)  # 2 seconds

    noise = np.random.normal(0, noise_std, n_samples + 2 * pad)

    b, a = butter(order, [fmin / nyq, fmax / nyq], btype="band")
    noise = filtfilt(b, a, noise)

    return noise[pad:-pad]


def create_scpt_survey(
    n_samples,
    dt,
    probe_depths,
    synthetic_layering,
    slowness,
    central_frequency=50,
    source_offset=0,
    sensor_separation: float = 0.5,
    azimuth=60,
    components=("X", "Y", "Z"),
    repeats=1,
    noise_std=0.00,
) -> ScptSurvey:
    f"""
    Create synthetic seismic CPT survey. 
    
    Stores ray path in Trace.stats[{RAY_PATH}]
    
    if repeats number of repeating shots with noise at the same depth. 
    """
    stream: list[DelayedTrace] = []

    now = datetime.datetime.now()
    shot_number_attribute = "shot_n"

    for idx, probe_depth in enumerate(probe_depths):
        starttime = now + datetime.timedelta(seconds=idx)
        for station in (Station.upper, Station.lower):
            if station == Station.upper:
                sensor_depth = probe_depth
            else:
                sensor_depth = probe_depth + sensor_separation

            sensor_depth = np.round(sensor_depth, DEPTH_ROUNDING)
            arrival_time, paths = arrival_times(
                sensor_depth, synthetic_layering, slowness, source_offset=source_offset
            )
            # Amplitude scaling for spherical divergence. First meter assumed constant amplitude
            trace = ricker_trace_at_delay(
                n_samples,
                delay=arrival_time,
                dt=dt,
                f0=central_frequency,
                amplitude=min(1, 1 / np.hypot(sensor_depth, source_offset)),
            )

            # Multiple noisy realizations
            for r in range(repeats):
                # noise = np.random.normal(
                #     loc=0.0,
                #     scale=noise_std,
                #     size=n_samples,
                # )

                shot_number = idx * repeats + r
                for comp, arr in zip(components, wave_to_xyz(trace, azimuth)):
                    noise = band_limited_noise(
                        n_samples,
                        dt,
                        fmin=5.0,
                        fmax=500.0,
                        noise_std=noise_std,
                    )

                    arr += noise

                    stats = trace.stats
                    stats[RAY_PATH] = paths
                    stats.component = comp
                    stats.station = station
                    stats.source_location = source_offset
                    stats[DEPTH] = sensor_depth
                    stats[shot_number_attribute] = shot_number
                    stats.starttime = UTCDateTime(starttime)
                    stream.append(DelayedTrace(arr, header=stats))

    identifier = shot_identifier_by_attribute(shot_number_attribute)
    survey = ScptSurvey(
        Stream(stream), sensor_separation=sensor_separation, shot_identifier=identifier
    )
    return survey


def wave_to_xyz(
    wave,
    azimuth: float,
    inclination: float = 0,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Resolve a 1D wave into its 3D Cartesian components given polarization angles.

    Parameters
    ----------
    wave : (N,) array_like
        The scalar wave amplitude over time (or samples).
    azimuth : float, optional
        Azimuth in degrees, measured in the horizontal plane from +X towards +Y.
    inclination : float, optional
        Inclination in degrees, measured from the horizontal plane.
        Positive values tilt up (toward +Z). 0 => horizontal, +90 => +Z, -90 => -Z.

    Returns
    -------
    x, y, z : (N,) ndarrays
        The three components of the polarized wave.

    Notes
    -----
    Unit polarization vector:
        u = (cos(inc)*cos(az), cos(inc)*sin(az), sin(inc))
    """
    w = np.asarray(wave, dtype=float)
    if w.ndim != 1:
        raise ValueError("`wave` must be a 1D array.")

    az = np.deg2rad(float(azimuth))
    inc = np.deg2rad(float(inclination))

    # Unit polarization vector
    ux = np.cos(inc) * np.cos(az)
    uy = np.cos(inc) * np.sin(az)
    uz = np.sin(inc)

    # Broadcast the scalar wave along each component
    x = w * ux
    y = w * uy
    z = w * uz
    return x, y, z
