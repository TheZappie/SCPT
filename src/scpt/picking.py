import itertools
import logging
import warnings
from contextlib import contextmanager
from copy import deepcopy
from enum import StrEnum
from functools import reduce
from itertools import pairwise
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from obspy import Stream, Trace
from obspy.signal.trigger import aic_simple
from tqdm import tqdm
from uncertainties import Variable, ufloat, unumpy

from scpt import logger, ScptError
from scpt.organisation import (
    group_stream,
    DEPTH,
    STATION,
    Station,
    DelayedTrace,
    stack_traces,
    cc,
    ScptSurvey,
    DataMode,
    RelativeTravelTime,
    UncertainRelativeTravelTime,
    cc_delayed_traces,
    DualSensorScpt,
    SingleComponentScpt,
)

UncertainPicks = dict[tuple[float, Station], Variable]
Picks = dict[tuple[float, Station], float]
# key: (depth, sensor ID)
# values: value with uncertainty estimate

# these represent picks with a reference to source offset
# key: (depth, sensor ID, source offset)
SUncertainPicks = dict[tuple[float, Station, float], Variable]
SPicks = dict[tuple[float, Station, float], float]


class TravelTimeEstimation(StrEnum):
    argmax = "argmax"
    argmin = "argmin"
    cc = "cc"
    simple = "simple"
    aic = "aic"


def crop_around_picks(
    stream: Stream,
    picks: Picks,
    half_window,
    taper: float | None = 0.05,
):
    keys = {key for key, group in group_stream(stream, DEPTH, STATION)}
    if not keys.issubset(picks.keys()):
        raise ValueError(
            "Given pick's do not encompass all depth-station point in given stream"
        )

    for key, group in group_stream(stream, DEPTH, STATION):
        start = picks[key] - half_window
        end = picks[key] + half_window
        for trace in group:
            trace.cut(start, end)
            if taper is not None:
                trace.taper(taper)


def without_uncertainty(picks: UncertainPicks) -> Picks:
    return {
        key: value.nominal_value if isinstance(value, Variable) else value
        for key, value in picks.items()
    }


def assign_pick_uncertainty(
    picks: Picks, uncertainty: Iterable | float = 0.5e-3
) -> UncertainPicks:
    try:
        iter(uncertainty)
    except TypeError:
        result = {key: Variable(value, uncertainty) for key, value in picks.items()}
    else:
        result = {
            key: Variable(value, std_dev)
            for (key, value), std_dev in zip(picks.items(), uncertainty)
        }
    return result


def pick_stream_mean(
    stream: Stream,
    minimum_uncertainty: float = 0.0,
    filt_value=None,
    filt_threshold=None,
    std=True,
    **kwargs,
) -> Variable | np.floating:
    """
    Pick stream, take mean and compute standard deviation by picking each trace
    if std=True, then return Variable, else float

    filt_value & filt_threshold allow for filtering out anomalous picks.
    """
    if filt_value and filt_threshold is None:
        raise ValueError("If filter value is specified, threshold must be specified")
    arrivals = []
    for trace in stream:
        arrivals.append(pick_trace(trace, **kwargs))
    if filt_value == "median":
        filt_value = np.median(arrivals)
    if filt_value:
        arrivals = [
            arrival
            for arrival in arrivals
            if filt_value - filt_threshold < arrival < filt_value + filt_threshold
        ]
        if not arrivals:
            raise ValueError("No arrivals found for given filter")
    mean = np.mean(arrivals)
    if std:
        return ufloat(mean, min(minimum_uncertainty, float(np.std(arrivals))))
    else:
        return mean


def pick_trace(trace: Trace | DelayedTrace, sub_sample=True, **kwargs):
    pick = pick_array(trace, sub_sample=sub_sample, **kwargs)
    timestamps = trace.times()
    if sub_sample:
        time = np.interp(pick, np.arange(len(timestamps)), timestamps)
    else:
        time = timestamps[int(pick)]
    if time < 0:
        # raise ValueError(f"Pick {time} should not be negative")
        logging.warning(
            f"Picked a negative arrival time on \n{trace.stats}\ntime: {time:.3g}"
        )
    return time  # pick * trace.stats.delta


def _parabolic_refine(y: np.ndarray, k: int) -> float:
    """
    Parabolic (quadratic) refinement of a discrete extremum at index k.
    Returns fractional offset relative to k. Always within [-.5 and .5]
    """
    if k <= 0 or k >= len(y) - 1:
        return 0.0

    y_m1 = y[k - 1]
    y_0 = y[k]
    y_p1 = y[k + 1]

    denom = y_m1 - 2 * y_0 + y_p1
    if denom == 0:
        return 0.0

    return 0.5 * (y_m1 - y_p1) / denom


def pick_array(
    trace,
    method: TravelTimeEstimation = TravelTimeEstimation.argmax,
    *,
    sub_sample: bool = False,
) -> float | int:
    """
    Pick a characteristic index in a 1D array.

    Parameters
    ----------
    trace : array-like
        Input signal.
    method : TravelTimeEstimation
        Picking strategy.
    sub_sample : bool, optional
        If True, apply parabolic interpolation around the picked index
        (when applicable) to obtain subsample precision.

    Returns
    -------
    pick : float
        Pick location in samples. Fractional if sub_sample=True.
    """
    trace = np.asarray(trace)

    if method == TravelTimeEstimation.simple:
        fraction = 0.80
        pick = np.where(np.max(trace) * fraction < trace)[0][0]

    elif method == TravelTimeEstimation.argmax:
        pick = np.argmax(trace)

    elif method == TravelTimeEstimation.argmin:
        pick = np.argmin(trace)

    elif method == TravelTimeEstimation.aic:
        pick = np.argmin(aic_simple(trace))

    else:
        raise ValueError(f"Unsupported method: {method}")

    # Optional sub-sample refinement
    if sub_sample and method in (
        TravelTimeEstimation.argmax,
        TravelTimeEstimation.argmin,
    ):
        delta = _parabolic_refine(trace, pick)
        return float(pick) + delta
    else:
        return int(pick)


def pick_stream(stream: Stream, **kwargs) -> Picks | UncertainPicks:
    """
    Pick each group of data in stream
    """
    result = {
        key: pick_stream_mean(group, **kwargs)
        for key, group in group_stream(stream, DEPTH, STATION)
    }
    return result


def pick_stack_guided(
    stream: Stream, filt_threshold, minimum_uncertainty
) -> UncertainPicks:
    """Pick on stack, then use individual pick on traces for uncertainty estimation"""
    groups = list(group_stream(stream, DEPTH, STATION))
    guide = {key: pick_trace(stack_traces(group)) for key, group in groups}
    uncertainties = {
        key: pick_stream_mean(
            group,
            filt_value=guide[key],
            filt_threshold=filt_threshold,
            minimum_uncertainty=minimum_uncertainty,
        )
        for key, group in groups
    }
    result = {
        key: Variable(guide[key], uncertainties[key].std_dev) for key, _ in groups
    }
    return result


def pick_by_cc_stacks_multisource(
    survey: ScptSurvey,
) -> SPicks:
    results = [
        add_offset(pick_by_cc_stacks(s), offset)
        for offset, s in survey.split_on_source()
    ]
    return add_dicts(results)


def add_offset(picks: Picks, offset) -> SPicks:
    return {
        (depth, sensor_ID, offset): value for (depth, sensor_ID), value in picks.items()
    }


def add_dicts(results: Iterable[dict]) -> dict:
    return reduce(lambda a, b: a | b, results, {})


def pick_by_cc_stacks(
    survey: DualSensorScpt,
    reference: float = None,
) -> Picks:
    """
    Pick using cross-correlation of stacks.

    Cross-correlation gies arrival differences, an absolute arrival time is given to the data,
    by using a single reference arrival based on thresholding all stacks

    Consider muting traces around first arrival before using.

    Positive travel time corresponds with a positive velocity and traces paired (shallow -> deep)
    """
    sequential_depth_data = list(survey.iter_over_depth())
    logging.info(f"Cross-correlating stacks (N={len(sequential_depth_data)})")
    tt_increments = [0.0]

    paired_data = [
        (traces1, traces2)
        for (key1, traces1), (key2, traces2) in pairwise(sequential_depth_data)
    ]

    for group1, group2 in paired_data:
        travel_time_dif = -cc_delayed_traces(stack_traces(group1), stack_traces(group2))
        tt_increments.append(travel_time_dif)
        # if tt < 0:
        #     cc_qc(stack_traces(group1), stack_traces(group2))

    cc_arrivals = np.cumsum(tt_increments)
    if reference is None:
        logger.info(f"Picking stacks by thresholding for reference value")
        threshold_pick = np.array(
            [pick_trace(stack_traces(traces)) for key, traces in sequential_depth_data]
        )
        reference = np.median(threshold_pick - cc_arrivals)
    logger.info(f"Reference pick: {reference:.3g} ms (pick at first depth position)")

    result = {
        key: t + reference for (key, data), t in zip(sequential_depth_data, cc_arrivals)
    }
    return result


def observations_cc_based(
    survey: SingleComponentScpt,
    pairing_threshold: float = 2,
    threshold: Optional[float] = None,
    mode=DataMode.sequential_only,
    reference_picks: Optional[Picks] = None,
) -> list[RelativeTravelTime | UncertainRelativeTravelTime]:
    """
    Create relative travel time observation from a SCPT survey

    reference_picks:
    This increases reliability provided the travel time is significantly
    less than the dominant period, which is almost always the case
    for adjacent geophones at a reasonable spacing.

    pairing_threshold: only used when using DataMode.nearby
    """
    if threshold is None:
        dominant_freq = survey.dominant_frequency()
        threshold = 1 / dominant_freq / 4
        logger.info(
            f"Cross-correlation threshold set to quarter of wavelength: Dominant frequency: {dominant_freq:.3g}"
        )

    if reference_picks is None:
        logger.info(f"Creating reference picks using correlation of stacks")
        reference_picks = pick_by_cc_stacks(survey)

    sequential_depth_data = list(survey.iter_over_depth())

    if mode == DataMode.sequential_only:
        paired_data = pairwise(sequential_depth_data)
        logging.info(
            f"Pairing traces from each sounding point.  "
            f"Original number of depth locations {len(sequential_depth_data)}. "
            f"Amount of pairings: {len(sequential_depth_data) - 1}"
        )
    elif mode == DataMode.all:
        paired_data = [
            ((key1, traces1), (key2, traces2))
            for (key1, traces1), (key2, traces2) in itertools.combinations(
                sequential_depth_data, 2
            )
        ]
        logging.info(
            f"Pairing traces from each sounding point.  "
            f"Original number of depth locations {len(sequential_depth_data)}. "
            f"Amount of pairings: {len(paired_data)}"
        )
    elif mode == DataMode.nearby:
        paired_data = [
            ((key1, traces1), (key2, traces2))
            for (key1, traces1), (key2, traces2) in itertools.combinations(
                sequential_depth_data, 2
            )
            if (abs(key2[0] - key1[0]) < pairing_threshold)
        ]
        logging.info(
            f"Pairing traces based on distance from each other. Distance {pairing_threshold} m. "
            f"Original number of depth locations {len(sequential_depth_data)}. "
            f"Amount of pairings: {len(paired_data)}"
        )
    else:
        raise ValueError(f"{mode} is not a valid mode")

    logging.info(f"Cross-correlating individual traces")

    observations = []
    for ((depth1, pos1), group1), ((depth2, pos2), group2) in tqdm(
        paired_data, desc="Cross correlating traces "
    ):
        # tt_product = [
        #     cc(*trace_pair) * sample_interval
        #     for trace_pair in itertools.product(group2, group1)
        # ]
        tt_product = [
            cc_delayed_traces(*trace_pair)
            for trace_pair in itertools.product(group2, group1)
        ]
        reference_time = (
            reference_picks[(depth2, pos2)] - reference_picks[(depth1, pos1)]
        )
        tt_filtered = [
            candidate
            for candidate in tt_product
            if (reference_time - threshold) < candidate < (reference_time + threshold)
        ]
        logging.debug(
            f"Filtered out {100 - len(tt_filtered) * 100 / len(tt_product):.1f}% cross-correlations using reference picks (cycle skipping)"
        )
        if not tt_filtered:
            raise ScptError(
                f"Can't create travel time difference observations. "
                f"All ΔT results from cross-correlations are too far away from reference ΔT. "
                f"Threshold: {threshold:.4g} s. Crosscorrelation between traces of depth {depth1}, "
                f"station {pos1} and depth {depth2}, station {pos2}"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tt = ufloat(np.mean(tt_filtered), np.std(tt_filtered))
        observation = RelativeTravelTime(
            relative_travel_time=tt,
            depth=(depth1, depth2),
            station=(pos1, pos2),
            offset=(group1[0].stats.source_location, group2[0].stats.source_location),
        )
        observations.append(observation)
    return observations


def travel_time_diff(data1, data2, sample_interval, method=TravelTimeEstimation.simple):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    if method == TravelTimeEstimation.simple:
        fraction = 0.8
        delay = (
            np.where(np.max(data1) * fraction < data1)[0][0]
            - np.where(np.max(data2) * fraction < data2)[0][0]
        )
        return delay * sample_interval
    elif method == TravelTimeEstimation.argmax:
        delay = (np.argmax(data1) - np.argmax(data2)) * sample_interval
        return delay
    elif method == TravelTimeEstimation.cc:
        return cc(data1, data2) * sample_interval
    else:
        raise ValueError(
            f"Travel time estimation method not supported ({method}), choose from {set(i.name for i in TravelTimeEstimation)}"
        )


def relative_observations_from_picks(
    picks: UncertainPicks | Picks | SUncertainPicks | SPicks,
    mode=DataMode.sequential_only,
    constant_offset: Optional[float] = None,
    include_non_relative: bool = False,
) -> list[UncertainRelativeTravelTime | RelativeTravelTime]:
    """
    From a set of travel time picks, pair data to give a set of relative observations.
    if constant_offset: input should be  UncertainPicks | Picks

    """
    # data_sorted: pairwise (key_upper, key_lower)
    if mode == DataMode.sequential_only:
        data_sorted = pairwise(sorted(picks, key=lambda x: x[0]))
    elif (mode == DataMode.top) or (mode == DataMode.bottom):
        dic = {DataMode.top: Station.upper, DataMode.bottom: Station.lower}
        data_sorted = pairwise(
            sorted((pick for pick in picks if pick[1] == dic[mode]), key=lambda x: x[0])
        )
    elif mode == DataMode.true_interval:
        # Find the corresponding lower pick for each upper pick.
        # separation = 0.5
        # upper = [pick for pick in sorted(picks, key=lambda x: x[0]) if pick[1] == Position.upper]
        #
        # data_sorted = zip(upper, [(depth + separation, Position.lower) for depth, _ in upper])

        dic = {Station.upper: 0, Station.lower: 1}
        data_sorted = [
            (a, b)
            for a, b in pairwise(sorted(picks, key=lambda x: (x[0], dic[x[1]])))
            if ((a[1] == Station.upper) and (b[1] == Station.lower))
        ]
    elif mode == DataMode.all:
        warnings.warn(
            f"DataMode {DataMode.all} does not contain more information than {DataMode.sequential_only}"
        )
        data_sorted = itertools.combinations(picks, 2)
    elif mode == DataMode.pseudo_interval:
        data_sorted = [
            (a, b)
            for a, b in pairwise(sorted(picks, key=lambda x: x[0]))
            if ((a[1] == Station.lower) and (b[1] == Station.upper))
        ]
    else:
        raise
    observations = []
    if constant_offset is not None:
        for (depth1, position1), (depth2, position2) in data_sorted:
            # noinspection PyUnresolvedReferences
            tt = picks[depth2, position2] - picks[depth1, position1]
            obs = RelativeTravelTime(
                relative_travel_time=tt,
                depth=(depth1, depth2),
                station=(position1, position2),
                offset=(constant_offset, constant_offset),
            )
            observations.append(obs)
    else:
        for (depth1, position1, offset1), (depth2, position2, offset2) in data_sorted:
            tt = picks[depth2, position2, offset2] - picks[depth1, position1, offset1]
            obs = RelativeTravelTime(
                relative_travel_time=tt,
                depth=(depth1, depth2),
                station=(position1, position2),
                offset=(offset1, offset2),
            )
            observations.append(obs)

    if include_non_relative:
        if constant_offset:
            for (depth, station), value in picks.items():
                obs = RelativeTravelTime(
                    relative_travel_time=value,
                    depth=(0, depth),
                    station=(None, station),
                    offset=(0, constant_offset),
                )
                observations.append(obs)
        else:
            for (depth, station, offset), value in picks.items():
                obs = RelativeTravelTime(
                    relative_travel_time=value,
                    depth=(0, depth),
                    station=(None, station),
                    offset=(0, offset),
                )
                observations.append(obs)
    return observations


def filter_observations(observations):
    """Remove observations with zero relative travel time, or negative travel time with a positively incrementing depth"""
    results = []
    for obs in observations:
        if obs.relative_travel_time == 0:
            continue
        elif (obs.depth[1] > obs.depth[0]) and obs.relative_travel_time < 0:
            continue
        elif (obs.depth[1] < obs.depth[0]) and obs.relative_travel_time > 0:
            continue
        results.append(obs)
    return results


def to_csv(picks: UncertainPicks, filename: str = "SCPT_travel_time.csv"):
    array = 1000 * np.asarray(list(picks.values()))
    df = pd.DataFrame(
        {"mean [ms]": unumpy.nominal_values(array), "std [ms]": unumpy.std_devs(array)},
        index=pd.Index(picks.keys()),
    )
    df.index.names = ["Depth [m]", "Sensor"]
    df.to_csv(filename, index=True, float_format="%.3f")


def validate_picks(survey: ScptSurvey, picks, threshold=0.5 / 50):
    survey_points = [key for key, _ in survey.iter_over_depth()]
    if not set(survey_points).issuperset(picks.keys()):
        raise ValueError(f"Picks does not contain all survey points")
    sequential_picks = [picks[key] for key in survey_points]
    if any(abs(np.diff(sequential_picks)) > threshold):
        raise ValueError(
            f"Travel times contain anomalous picks. One or more times, 2 sequential picks are more than {threshold * 1000:.3g} ms apart. "
        )


def shift_to_zero_offset(
    survey: DualSensorScpt, picks: Picks, inplace=True
) -> None | DualSensorScpt:
    """
    Reduces data in-place to zero source offset, by changing sample interval and delay

    Slope method or 'vertical travel time slope-based method' Hallal & Cox 2019
    """
    if not inplace:
        survey = deepcopy(survey)
    if not {key for key, _ in survey.iter_over_depth()}.issubset(picks.keys()):
        raise ValueError(f"Picks does not contain all survey points")
    offset = survey.get_source_offset()
    for (depth, station), trace_group in survey.iter_over_depth():
        correction = depth / np.hypot(depth, offset)
        for trace in trace_group:
            trace.stats.delta *= correction
            trace.source_location = 0
            if isinstance(trace, DelayedTrace):
                trace.delay *= correction
    return survey


def shift_picks_to_zero_offset(picks: Picks | SPicks, constant_offset=None):
    result = {}
    if constant_offset is not None:
        picks: Picks
        for (depth, station), value in picks.items():
            correction = depth / np.hypot(depth, constant_offset)
            result[(depth, station)] = value * correction
    else:
        picks: SPicks
        for (depth, station, offset), value in picks.items():
            correction = depth / np.hypot(depth, offset)
            result[(depth, station, offset)] = value * correction
    return result
