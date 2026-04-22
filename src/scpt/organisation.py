import itertools
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from itertools import groupby
from pathlib import Path
from typing import Optional, Iterable, Sequence, Self, Any, Iterator, cast, Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import ndarray
from obspy import Stream, Trace
from obspy.core import Stats
from uncertainties import Variable

from scpt import logger
from scpt.processing import (
    cluster_1d,
    has_nonzero_flat,
    spherical_divergence_correction,
)
from scpt.ray_tracing import solve_ray_parameter_between_points, compute_ray_lengths
from scpt.typical_values import LOWER_SPEED, UPPER_SPEED, WAVE_SIZE

DEPTH_ROUNDING = 4

FILEPATH = "Filepath"
DEPTH = "depth"
STATION = "station"


PLOT_FILL = "fill"
PLOT_BUTTERFLY = "butterfly"


class Station(StrEnum):
    """Denotes whether something is in the top or bottom of probe"""

    upper = "1"
    lower = "2"


class DelayedTrace(Trace):
    """
    An Obspy seismic Trace with explicit possibility to contain a non-zero recording start time.

    Delay is stored in self.stats.seg2["DELAY"]
    """

    @property
    def delay(self):
        delay = float(self.stats.seg2["DELAY"])
        return delay

    @delay.setter
    def delay(self, value):
        self.stats.seg2["DELAY"] = value

    def times(self, *args, **kwargs):
        return super().times(*args, **kwargs) + self.delay

    @classmethod
    def create(cls, *args, delay, header=None, **kwargs):
        # should this be a __init__?
        if header is None:
            header = {}
        if "seg2" not in header:
            header["seg2"] = {}
        trace = cls(*args, header=header, **kwargs)
        trace.delay = delay
        return trace

    def cut(self, start=None, end=None):
        """Trimming relative to start time"""
        d = self.stats.delta
        if start is not None:
            idx = max(0, int(np.round((start - self.delay) / d)))
        else:
            idx = 0
        if end is not None:
            jdx = min(len(self), int(np.round((end - self.delay) / d)))
        else:
            jdx = len(self)
        # noinspection PyAttributeOutsideInit
        self.data = self.data[idx:jdx]
        self.stats.npts = len(self.data)
        self.delay = start


def vstack_stream(stream: Stream) -> ndarray:
    return np.array([trace.data for trace in stream])


def cc(array_1, array_2, *, sub_sample: bool = False) -> float:
    """
    Cross-correlate two arrays.

    Parameters
    ----------
    array_1, array_2 : np.ndarray
        Input signals.
    sub_sample : bool, optional
        If True, apply parabolic interpolation around the correlation peak
        to obtain a sub-sample lag estimate. Default is False.

    Returns
    -------
    delay : float
        Lag in number of samples. Fractional if sub_sample=True.
    """
    corr = scipy.signal.correlate(array_1, array_2, mode="same")
    lags = scipy.signal.correlation_lags(len(array_1), len(array_2), mode="same")

    k = np.argmax(corr)

    if sub_sample and 0 < k < len(corr) - 1:
        y_m1 = corr[k - 1]
        y_0 = corr[k]
        y_p1 = corr[k + 1]

        denom = y_m1 - 2 * y_0 + y_p1
        if denom != 0:
            delta = 0.5 * (y_m1 - y_p1) / denom
        else:
            delta = 0.0
    else:
        delta = 0.0

    return lags[k] + delta


def cc_delayed_traces(
    trace_1: DelayedTrace, trace_2: DelayedTrace, sub_sample=True
) -> float:
    """
    Cross-correlate 2 traces. Takes their relative starting into account.
    Returns lag time in seconds
    """
    assert trace_1.stats.delta == trace_2.stats.delta
    sample_interval = trace_1.stats.delta
    starting_lag = trace_1.delay - trace_2.delay
    delay = cc(trace_1, trace_2, sub_sample=sub_sample)
    return delay * sample_interval + starting_lag


def group_stream(stream: Stream, *keys, reverse=False) -> Iterator[tuple[Any, Stream]]:
    def make_key_func(key):
        if isinstance(key, str):
            return lambda stats, k=key: stats[k]
        return key

    keys_functions = [make_key_func(key) for key in keys]

    keys_aggregated = lambda trace: tuple(
        [func(trace.stats) for func in keys_functions]
    )
    for k, g in groupby(
        sorted(stream, key=keys_aggregated, reverse=reverse), key=keys_aggregated
    ):
        yield k, Stream(g)


def amplitude_spectrum(trace: Trace):
    freqs = np.fft.rfftfreq(len(trace), trace.stats.delta)
    fft = np.abs(np.fft.rfft(trace))
    return freqs, fft


def split_on_source(st: Iterable[Trace]):
    sources = np.array([trace.stats.source_location for trace in st])
    sources_offsets = np.unique(sources)
    return [
        (
            offset,
            Stream([trace for trace in st if trace.stats.source_location == offset]),
        )
        for offset in sources_offsets
    ]


def select_sensor(stream, station: Station):
    # Identical to obspy.Stream.select(station=station)
    return Stream([trace for trace in stream if trace.stats.station == station])


def shot_starttime(stats: Stats | Trace):
    """Returns a value that is identical for all traces related to the same shot"""
    # an alternative:
    # trace.stats.shot_id = (
    #     trace.stats.seg2["ACQUISITION_DATE"]
    #     + " "
    #     + trace.stats.seg2["ACQUISITION_TIME"]
    # )
    if isinstance(stats, Stats):
        return np.datetime64(stats.starttime)
    elif isinstance(stats, Trace):
        return np.datetime64(stats.stats.starttime)
    else:
        raise ValueError()


def shot_identifier_by_attribute(attr):
    """Returns a value that is identical for all traces related to the same shot"""

    def shot_identifier(stats: Stats | Trace):
        if isinstance(stats, Stats):
            return getattr(stats, attr)
        elif isinstance(stats, Trace):
            return getattr(stats.stats, attr)
        else:
            raise ValueError()

    return shot_identifier


def iter_shots(
    stream: Stream, identifier=shot_starttime
) -> Iterable[tuple[Any, Stream]]:
    shot = np.array([identifier(trace.stats) for trace in stream])
    sorted_shots = np.unique(shot)
    for timestamp in sorted_shots:
        yield (
            timestamp,
            Stream([stream[i] for i in np.where(shot == timestamp)[0]]),
        )


def guess_sensor_separation(stream: Stream, identifier=shot_starttime):
    all_sets = itertools.chain.from_iterable(
        [
            iter_shots(stream.select(component=component), identifier=identifier)
            for component in ("X", "Y", "Z")
        ]
    )
    first_shot_id, (first_a, first_b) = next(all_sets)
    separation = np.round(
        abs(first_a.stats.depth - first_b.stats.depth), DEPTH_ROUNDING
    )

    for shot_id, (trace_a, trace_b) in all_sets:
        if not np.isclose(abs(trace_a.stats.depth - trace_b.stats.depth), separation):
            raise ValueError(
                f"At trace with shot_ID: {shot_id}, the station separation does match the "
                f"separation from the first set at {first_shot_id}. "
                f"Separation first shot: {separation}. "
                f"Anomalous {trace_a.stats.depth - trace_b.stats.depth}, "
                f"depth values: {trace_a.stats.depth} and {trace_b.stats.depth}"
            )
    return separation


def get_trace_positions_from_stream(stream: Stream) -> ndarray:
    return np.array([trace.stats.depth for trace in stream])


def get_stat_from_stream(stream: Stream, statistic) -> ndarray:
    return np.array([trace.stats[statistic] for trace in stream])


@dataclass
class DualSensorScpt:
    """
    Encapsulated all data related Seismic CPT setup with a single source.

    The 1-component can be artificial, the result of getting the best component of a 3-component survey.

    Contained data in the underlying obspy.Stream has several guarantees to it related to SCPT survey setup.
    See method validate.
    - Data has exactly 3 components (orthogonal directions).
    - Each shot has exactly 2 sensors.
    - The distance between these 2 sensors is constant (sensor_separation)

    Data can have variable length and starting time, contained in obspy Trace objects.
    """

    stream: Stream
    sensor_separation: float
    shot_identifier: Callable[[Trace | Stats], Any] = shot_starttime

    def __str__(self):
        probe_positions = self.get_unique_depths(Station.upper)

        string = (
            f"Seismic CPT with 2 sensors, each 3 components\n"
            f"Sensor separation: {self.sensor_separation} m\n"
            f"Source offset: {self.get_source_offsets()} m\n"
            f"Number of shots: {len(self.stream) // 6} (traces: {len(self.stream)})\n"
            f"Top sensor range: {probe_positions[0]} - {probe_positions[-1]} m, {len(probe_positions)} positions\n"
            f"Dominant frequency: {self.dominant_frequency():.2f} Hz\n"
            f"Samples per wavelength: ≈{int(self.samples_per_wavelength())}\n"
        )
        return string

    @classmethod
    def from_stream(cls, stream: Stream, identifier=shot_starttime):
        sensor_separation = guess_sensor_separation(stream, identifier)

        obj = cls(
            stream=stream,
            sensor_separation=sensor_separation,
            shot_identifier=identifier,
        )
        return obj

    def __post_init__(self):
        source_locations = np.unique(
            [trace.stats.source_location for trace in self.stream]
        )
        if len(source_locations) > 1:
            raise ValueError(
                f"Multiple source locations, not allowed for {self.__class__.__name__}, values:: {source_locations}"
            )

    def plot_experiment_repetition(self):
        experiment_repetition = self.experiment_repetition()
        groups_sizes = list(experiment_repetition.values())
        depths = list(experiment_repetition.keys())
        print(
            f"Mean amount of shots at each probe position: {np.mean(groups_sizes):.2f}"
        )

        fig, ax = plt.subplots()
        ax.set_xlabel("Probe depth (m)")
        ax.set_ylabel("Number of experiments (-)")
        ax.stem(depths, groups_sizes)
        return ax

    def find_clipped_trace(self, threshold=10):
        """Return a trace that is clipped, if any"""
        for trace in self.stream:
            if has_nonzero_flat(trace, threshold):
                return trace
        return None

    def warn_for_clipping(self, **kwargs):
        if trace := self.find_clipped_trace(**kwargs):
            identity = self.shot_identifier(trace)
            print(f"In shot {identity}, there is a trace that is clipped")

    def shot_info(self) -> dict[Any, dict[str, Any]]:
        info_keys = [
            "starttime",
            "probe_depth",
            "upper_sensor_depth",
            "lower_sensor_depth",
        ]

        def func(stats: Stats):
            dic = {}
            dic["starttime"] = stats.starttime
            dic["probe_depth"] = stats.depth
            dic["upper_sensor_depth"] = stats.depth
            dic["lower_sensor_depth"] = np.round(
                stats.depth + self.sensor_separation, 4
            )
            return dic

        return {
            identifier: func(shot.select(station=Station.upper)[0].stats)
            for identifier, shot in self.iter_shots()
        }

    def get_shot(self, identifier) -> Stream:
        """Each shot has identifier, depth, time"""
        return next(
            shot for identity, shot in self.iter_shots() if identity == identifier
        )

    def cut_measurements(
        self, top: Optional[float] = None, bottom: Optional[float] = None
    ):
        """
        Filters data in-place
        if only one of the sensors is in the range, it will not be cut.
        """
        # trace.stats.depth is positive direction into the earth
        if top is None and bottom is None:
            return
        elif top is None:
            stream = [
                trace
                for trace in self.stream
                if (
                    (trace.stats.depth > (bottom - self.sensor_separation))
                    and (trace.stats.station == Station.upper)
                )
                or (
                    (trace.stats.depth > bottom)
                    and (trace.stats.station == Station.lower)
                )
            ]
        elif bottom is None:
            stream = [
                trace
                for trace in self.stream
                if (
                    (trace.stats.depth < top) and (trace.stats.station == Station.upper)
                )
                or (
                    (trace.stats.depth < (top - self.sensor_separation))
                    and (trace.stats.station == Station.lower)
                )
            ]
        else:
            stream = [
                trace for trace in self.stream if not (bottom > trace.stats.depth > top)
            ]
            # self.stream.data = [trace for trace in self.stream if
            #                  (((trace.stats.depth > (bottom - self.sensor_separation)) and (trace.stats.station == Station.upper) )
            #                  or ((trace.stats.depth > bottom) and (trace.stats.station == Station.lower)))
            #                  and
            #                  (((trace.stats.depth < top) and (trace.stats.station == Station.upper))
            #                     or ((trace.stats.depth < (top - self.sensor_separation)) and (
            #                              trace.stats.station == Station.lower)))
            #                  ]
        self.stream = Stream(stream)

    def sensor_map(self):
        return {trace.stats.depth: trace.stats.station for trace in self.stream}

    def experiment_repetition(self):
        """How often a measurement is taken at a certain probe depth"""
        obj = self.select(component="X")
        groups_sizes = {
            depth: sum(obj.get_trace_positions() == depth)
            for depth in obj.get_unique_depths(sensor=Station.upper)
        }
        return groups_sizes

    def filter_low_repetitions(self, minimum_repetition=0):
        if not minimum_repetition:
            return self
        experiment_repetition = self.experiment_repetition()
        bad_depths = {
            depth
            for depth, reps in experiment_repetition.items()
            if reps < minimum_repetition
        }
        if not bad_depths:
            print(
                f"All probe depth measurement repeated more than {minimum_repetition} times"
            )
            return self
        print(f"Filtering all measurements taken at: {bad_depths}")

        bad_shots = set()
        for shot, group in self.iter_shots():
            if group.select(station=Station.upper)[0].stats.depth in bad_depths:
                print(f"Filtering shot {shot}")
                bad_shots.add(shot)
        stream = Stream(
            [
                trace
                for trace in self.stream
                if self.shot_identifier(trace.stats) not in bad_shots
            ]
        )
        # noinspection PyArgumentList
        return self.__class__(
            stream=stream,
            sensor_separation=self.sensor_separation,
        )

    # def select_sensor(self, position: Station):
    #     self.stream = select_sensor(self.stream, position)

    def validate_stream(self):
        if not all_equal(trace.stats.delta for trace in self.stream):
            raise ValueError("Sample interval not identical for all traces")

    def select(self, *args, station=None, **kwargs) -> Self:
        """
        Example:

        survey.select(component='X')
        survey.select(station=Position.upper)
        """
        # noinspection PyArgumentList
        if station:
            raise ValueError(
                f"This class has guaranteed two stations. Use {self.__class__.__name__}.stream.select directly"
            )
        return self.__class__(
            stream=self.stream.select(*args, **kwargs),
            sensor_separation=self.sensor_separation,
        )

    @classmethod
    def dual_survey_by_polarizing(cls, stream: Stream, sensor_separation):
        """
        Flip the signal of the second

        :param stream:
        :param sensor_separation:
        :return:
        """
        sources = np.array([trace.stats.source_location for trace in stream])
        sources_offsets = np.unique(sources)
        if len(sources_offsets) != 2:
            raise ValueError()
        for trace in stream:
            if trace.stats.source_location == sources_offsets[1]:
                trace.data *= -1

        obj = cls(
            stream=stream,
            sensor_separation=sensor_separation,
        )
        return obj

    def get_probe_positions(self):
        shot_probe_position_map = {
            timestamp: st.select(station=Station.upper)[0].stats.depth
            for timestamp, st in self.iter_shots()
        }
        probe_position = np.array(
            [
                shot_probe_position_map[self.shot_identifier(trace.stats)]
                for trace in self.stream
            ]
        )
        return probe_position

    def get_trace_positions(self) -> ndarray:
        return get_trace_positions_from_stream(self.stream)

    def get_stat(self, statistic):
        return get_stat_from_stream(self.stream, statistic)

    def get_unique_depths(self, sensor=None):
        # np.unique is guaranteed to be sorted
        if sensor is None:
            traces = self.get_trace_positions()
        elif sensor.upper:
            traces = get_trace_positions_from_stream(
                self.stream.select(station=Station.upper)
            )
        elif sensor.lower:
            traces = get_trace_positions_from_stream(
                self.stream.select(station=Station.lower)
            )
        else:
            raise
        return np.unique(traces)

    def iter_shots(self) -> Iterable[tuple[Any, Stream]]:
        yield from iter_shots(self.stream, identifier=self.shot_identifier)

    def iter_over_depth(self) -> Iterator[tuple[tuple[float, Station], Stream]]:
        """Iterate data over data depth, shallow to deeper"""
        return group_stream(self.stream, DEPTH, STATION)

    def iter_true_interval_data(self):
        """
        Iterate stacked data of one component, for each probe position
        """
        dic = defaultdict(list)
        for _, st in self.iter_shots():
            upper, lower = (
                st.select(station=Station.upper)[0],
                st.select(station=Station.lower)[0],
            )
            dic[upper.stats.depth].append((upper, lower))
        for depth, data in sorted(dic.items()):
            upper = stack([d[0] for d in data])
            lower = stack([d[1] for d in data])
            yield depth, (upper, lower)

    def get_source_offsets(self) -> list[float]:
        return np.unique(
            [trace.stats.source_location for trace in self.stream]
        ).tolist()

    def get_source_offset(self) -> float:
        """Check if there is only one source"""
        offsets = self.get_source_offsets()
        if len(offsets) > 1:
            raise ValueError("Survey has multiple sources. ")
        return offsets[0]

    def split_on_source(self):
        streams = split_on_source(self.stream)
        # noinspection PyArgumentList
        return [
            (offset, self.__class__(stream, self.sensor_separation))
            for offset, stream in streams
        ]

    def cut(self, start=None, end=None):
        for trace in self.stream:
            trace.cut(start, end)

    def cut_around_velocities(
        self,
        lower_velocity: float = LOWER_SPEED,
        upper_velocity: float = UPPER_SPEED,
        wave_size: float = WAVE_SIZE,
        taper: float | None = 0.05,
    ):
        """
        Crops traces around arrival time, using straight line assumption.
        If taper is not None, the trace is tapered at edges
        """
        for trace in self.stream.traces:
            source_offset = trace.stats.source_location
            depth = trace.stats.depth
            distance = np.hypot(depth, source_offset)
            start = distance / upper_velocity
            end = distance / lower_velocity + wave_size
            trace.cut(start, end)
            if taper is not None:
                trace.taper(taper)

    def map_depth_filepath(self) -> dict[float, list[str | Path]]:
        dic = defaultdict(list)
        s = set(self.get_unique_depths(Station.upper))
        for trace in self.stream.select(component="X"):
            depth = trace.stats[DEPTH]
            if depth in s:
                dic[depth].append(trace.stats[FILEPATH])
        return {k: dic[k] for k in sorted(dic)}

    def samples_per_wavelength(self) -> np.floating:
        """Classic indicator if signal is sampled sufficiently"""
        wavelength = 1 / self.dominant_frequency()
        return wavelength / self.sample_interval()

    def sample_interval(self) -> float:
        # Relies on validation on the dataset first
        # if not all_same(trace.stats.delta for trace in stream):
        #     raise
        return self.stream[0].stats.delta


class SingleComponentScpt(DualSensorScpt):
    """
    Encapsulated all data related to a typical dual-sensor, 1-component, Seismic CPT setup with a single source.

    The 1-component can be artificial, the result of getting the best component of a 3-component survey.

    Contained data in the underlying obspy.Stream has several guarantees to it related to SCPT survey setup.
    See method validate.
    - Each shot has exactly 2 sensors.
    - The distance between these 2 sensors is constant (sensor_separation)

    Data can have variable length and starting time, contained in obspy Trace objects.
    """

    def __str__(self):
        probe_positions = self.get_unique_depths(Station.upper)

        string = (
            f"Seismic CPT with 2 sensors, 1 component\n"
            f"Sensor separation: {self.sensor_separation} m\n"
            f"Source offset: {self.get_source_offsets()} m\n"
            f"Number of shots: {len(self.stream) // 6} (traces: {len(self.stream)})\n"
            f"Top sensor range: {probe_positions[0]} - {probe_positions[-1]} m, {len(probe_positions)} positions\n"
            f"Dominant frequency: {self.dominant_frequency():.2f} Hz\n"
            f"Samples per wavelength: ≈{int(self.samples_per_wavelength())}\n"
        )
        return string

    def __post_init__(self):
        self.validate_stream()

    def validate_stream(self):
        """Check if every shot exactly 2 traces for each station expected of SCPT"""
        try:
            groups = [
                (identifier, list(group))
                for identifier, group in groupby(
                    sorted(self.stream, key=self.shot_identifier),
                    key=self.shot_identifier,
                )
            ]
            bad_shot = next(
                identifier for identifier, group in groups if len(group) != 2
            )
        except StopIteration:
            pass
        else:
            raise ValueError(
                f"Shot {bad_shot} does not have exactly 2 traces for the 2 sensors)"
            )

        if any(
            [
                True
                for _, group in group_stream(self.stream, self.shot_identifier, STATION)
                if len(group) != 1
            ]
        ):
            raise ValueError(f"There is a shot with not exactly 3 components")

    def experiment_repetition(self):
        """How often a measurement is taken at a certain probe depth"""
        groups_sizes = {
            depth: sum(self.get_trace_positions() == depth)
            for depth in self.get_unique_depths(sensor=Station.upper)
        }
        return groups_sizes


@dataclass
class ScptSurvey(DualSensorScpt):
    """
    Encapsulates data related to a typical dual-sensor, 3-component, Seismic CPT setup with a single source.

    Contained data is stored as an underlying obspy.Stream has several guarantees to it related to SCPT survey setup.
    See method validate.
    - Data has exactly 3 components (orthogonal directions).
    - Each shot has exactly 2 sensors.
    - The distance between these 2 sensors is constant (sensor_separation)

    Data can have variable length and starting time, contained in obspy Trace objects.
    """

    components: tuple[str, ...] = ("X", "Y", "Z")

    def __str__(self):
        probe_positions = self.get_unique_depths(Station.upper)

        string = (
            f"Seismic CPT with 2 sensors, each 3 components\n"
            f"Sensor separation: {self.sensor_separation} m\n"
            f"Source offset: {self.get_source_offsets()} m\n"
            f"Number of shots: {len(self.stream) // 6} (traces: {len(self.stream)})\n"
            f"Top sensor range: {probe_positions[0]} - {probe_positions[-1]} m, {len(probe_positions)} positions\n"
            f"Dominant frequency: {self.dominant_frequency():.2f} Hz\n"
            f"Samples per wavelength: ≈{int(self.samples_per_wavelength())}\n"
        )
        return string

    def __post_init__(self):
        self.validate_stream()

    def validate_stream(self):
        """Check if every shot has all components expected of SCPT"""
        try:
            groups = [
                (identifier, list(group))
                for identifier, group in groupby(
                    sorted(self.stream, key=self.shot_identifier),
                    key=self.shot_identifier,
                )
            ]
            bad_shot = next(
                identifier for identifier, group in groups if len(group) != 6
            )
        except StopIteration:
            pass
        else:
            raise ValueError(
                f"Shot {bad_shot} does not have exactly 6 traces (3 components × 2 sensors)"
            )

        if any(
            [
                True
                for _, group in group_stream(self.stream, self.shot_identifier, STATION)
                if len(group) != 3
            ]
        ):
            raise ValueError(f"There is a shot with not exactly 3 components")

    def spherical_divergence_correction(self):
        for trace in self.stream:
            spherical_divergence_correction(trace)

    def select_component(self, component) -> SingleComponentScpt:
        return SingleComponentScpt(
            self.stream.select(component=component),
            sensor_separation=self.sensor_separation,
            shot_identifier=self.shot_identifier,
        )

    def iter_over_depth(
        self, component: Optional[str] = None
    ) -> Iterator[tuple[Any, Stream]]:
        """Iterate data over data depth, shallow to deeper"""
        if component is not None:
            stream = self.stream.select(component=component)
        else:
            stream = self.stream
        return group_stream(stream, DEPTH, STATION)

    def iter_traces(
        self, component: str, station: Station, depth_ordered: bool = False
    ) -> Iterable[Trace]:
        """
        Iterate

        :param component:
        :param station:
        :param depth_ordered: if False, ordered time-wise
        :return:
        """
        selection = self.select(component=component, station=station)
        if not depth_ordered:
            for traces in selection.iter_shots():
                assert len(traces) == 1
                yield traces[0]
        else:
            raise NotImplementedError

    def iter_components(
        self, station: Station, depth_ordered: bool = False
    ) -> Iterable[tuple[Trace, Trace, Trace]]:
        """
        returns a tuple of 3 traces, the trace of each component.

        :param station:
        :param depth_ordered: if False, ordered time-wise
        :return:
        """
        if not depth_ordered:
            for _, traces in iter_shots(
                self.stream.select(station=station), identifier=self.shot_identifier
            ):
                assert len(traces) == 3
                yield cast(tuple[Trace, Trace, Trace], tuple(traces.traces))
        else:
            raise NotImplementedError
            # for _, traces in selection.iter_over_depth():
            #     assert len(traces) == 3
            #     yield tuple(traces.traces)


def stack(stream: Stream | Iterable[ndarray] | Iterable[Trace]):
    return np.sum([trace for trace in stream], axis=0)


def all_equal(iterable):
    iterator = iter(iterable)

    try:
        first_item = next(iterator)
    except StopIteration:
        return True

    for x in iterator:
        if x != first_item:
            return False
    return True


def pad_to_earliest(start_times, signals, dt, fill_value=0.0):
    """
    Pad signals to start at the earliest time and end at latest.
    Rounds subsample interval
    """
    t_min = np.min(start_times)
    t_max = np.max(
        [
            len(signal) * dt + start_time
            for start_time, signal in zip(start_times, signals)
        ]
    )
    n_signals = len(signals)
    # Create common time array
    n_samples = int(np.round((t_max - t_min) / dt)) + 1
    t_min + np.arange(n_samples) * dt

    # Pad each signal
    normalized_signals = []
    for i in range(n_signals):
        # Calculate number of samples to prepend
        n_prepend = int(np.round((start_times[i] - t_min) / dt))

        # Create padded signal
        padded = np.full(n_samples, fill_value)
        padded[n_prepend : n_prepend + len(signals[i])] = signals[i]
        normalized_signals.append(padded)
    return normalized_signals, t_min


def stack_traces(traces: Sequence[DelayedTrace] | Stream) -> DelayedTrace:
    identical_sampling = all_equal(trace.stats.delta for trace in traces)
    identical_delay = all_equal(trace.delay for trace in traces)
    identical_npts = all_equal(trace.stats.npts for trace in traces)
    if identical_npts and identical_sampling and identical_delay:
        return DelayedTrace(stack(traces), header=traces[0].stats)
    elif identical_sampling and identical_delay:
        # stack with shift.
        # noinspection PyArgumentList
        data = np.sum(list(itertools.zip_longest(*traces, fillvalue=0)), axis=1)
        header = traces[0].stats
        header.npts = len(data)
        return DelayedTrace(data, header=header)
    elif identical_sampling:
        shifted_signals, t_min = pad_to_earliest(
            [trace.delay for trace in traces], traces, traces[0].stats.delta
        )
        data = np.sum(shifted_signals, axis=0)
        header = traces[0].stats
        header.npts = len(data)
        return DelayedTrace.create(data=data, header=header, delay=t_min)
    else:
        raise ValueError(
            "Sampling rate not identical. "
            "Tip: use method obspy.Stream.resample to resample to equal sampling rate"
        )


class DataMode(StrEnum):
    """Name of dataset derived from Seismic CPT's is used for a process"""

    all = "all"
    sequential_only = "sequential_only"
    true_interval = "true_interval"
    pseudo_interval = "pseudo_interval"
    top = "top"
    bottom = "bottom"
    nearby = "nearby"


@dataclass
class RelativeTravelTime:
    """
    Observation derived from the difference in arrival time

    Source depth assumed to be zero: observation depths are relative to source.
    """

    relative_travel_time: float
    depth: tuple[float, float]
    station: tuple[Station, Station]
    offset: tuple[float, float]

    def __post_init__(self):
        if self.distance_difference() == 0 and (self.station[0] == self.station[1]):
            raise ValueError("Sensors are the same and relative distance is zero.")

    def straight_distance(self) -> tuple[float, float]:
        # noinspection PyTypeChecker
        return tuple([np.hypot(self.depth[i], self.offset[i]) for i in range(2)])

    def distance_difference(self):
        """Assuming straight ray paths. Positive when second path is larger than first path"""
        a, b = self.straight_distance()
        return b - a


@dataclass
class UncertainRelativeTravelTime:
    """Observation derived from the difference in arrival time"""

    relative_travel_time: Variable
    depth: tuple[float, float]
    station: tuple[Station, Station]
    offset: tuple[float, float]

    def __post_init__(self):
        if self.distance_difference() == 0:
            raise ValueError("Relative distance traveled cannot be zero")

    def get_nominal_observation(self):
        return RelativeTravelTime(
            self.relative_travel_time.nominal_value,
            self.depth,
            self.station,
            self.offset,
        )

    def straight_distance(self) -> tuple[float, float]:
        # noinspection PyTypeChecker
        return tuple(
            [(self.depth[i] ** 2 + self.offset[i] ** 2) ** 0.5 for i in range(2)]
        )

    def distance_difference(self):
        """Assuming straight ray paths. Positive when second path is larger than first path"""
        a, b = self.straight_distance()
        return b - a


def interval_method(
    observations: Sequence[RelativeTravelTime],
    invalidate_negatives=True,
    mean_depth=True,
) -> list[tuple[float, float]] | list[tuple[float, float, float]]:
    """

    :param observations:
    :param invalidate_negatives: if True, assign np.nan on negative speeds
    :param mean_depth: if True, return result in the form of (mean_depth, speed) instead of (depthA, depthB, speed)
    :return:
    """
    result = []
    for i_row, obs in enumerate(observations):
        speed = obs.distance_difference() / obs.relative_travel_time
        if invalidate_negatives and (speed <= 0):
            speed = np.nan
        if mean_depth:
            mean_depth = np.mean([obs.depth[0], obs.depth[1]])
            result.append((mean_depth, speed))
        else:
            result.append((obs.depth[0], obs.depth[1], speed))
    return result


def interval_method_ray_traced(
    observations: Sequence[RelativeTravelTime],
    layer_bounds,
    speed_model,
    invalidate_negatives=True,
    mean_depth=True,
) -> list[tuple[float, float]] | list[tuple[float, float, float]]:
    """

    :param observations:
    :param invalidate_negatives: if True, assign np.nan on negative speeds
    :param mean_depth: if True, return result in the form of (mean_depth, speed) instead of (depthA, depthB, speed)
    :return:
    """
    result = []

    slowness_model = 1 / np.asarray(speed_model)
    for i_row, obs in enumerate(observations):
        relative_path_length = relative_path_length_ray_traced(
            layer_bounds, obs, slowness_model
        )
        speed = relative_path_length / obs.relative_travel_time
        if invalidate_negatives and (speed <= 0):
            speed = np.nan
        if mean_depth:
            mean_depth = np.mean([obs.depth[0], obs.depth[1]])
            result.append((mean_depth, speed))
        else:
            result.append((obs.depth[0], obs.depth[1], speed))
    return result


def relative_path_length_ray_traced(
    layer_bounds, obs: RelativeTravelTime, slowness_model: ndarray
) -> int:
    depth_a = obs.depth[0]
    ray_parameter_A = solve_ray_parameter_between_points(
        obs.offset[0], 0, 0, depth_a, layer_bounds, slowness_model
    )
    ray_A = compute_ray_lengths(
        0, depth_a, layer_bounds, slowness_model, ray_parameter_A
    )
    depth_b = obs.depth[1]
    ray_parameter_B = solve_ray_parameter_between_points(
        obs.offset[0], 0, 0, depth_b, layer_bounds, slowness_model
    )
    ray_B = compute_ray_lengths(
        0, depth_b, layer_bounds, slowness_model, ray_parameter_B
    )
    relative_path_length = sum(ray_B) - sum(ray_A)
    return relative_path_length


def cluster_depths(survey: ScptSurvey, distance: float = 0.01):
    """
    Consider depth values less than <distance> apart identical. Replace those values with a mean.

    Will never move value more than <distance>
    """
    all_mapping = {}
    for station in (Station.upper, Station.lower):
        original_depths = survey.get_unique_depths(sensor=station)
        clusters = cluster_1d(original_depths, distance)
        new_depths = np.concatenate(
            [
                [np.round(np.mean(depths), DEPTH_ROUNDING)] * len(depths)
                for depths in clusters
            ]
        )
        mapping = {
            key: value
            for key, value in zip(original_depths, new_depths)
            if key != value
        }
        for trace in survey.stream:
            if trace.stats.depth in mapping:
                trace.stats.depths = mapping[trace.stats.depth]
        all_mapping[station] = mapping
    return mapping


def v_resolution(sample_interval, d=0.5, v=300):
    """ "
    d: reference interval
    v: reference speed
    """
    return sample_interval * v**2 / (d + sample_interval * v)


def check_v_resolution(survey, d=0.5, v=300, threshold=5):
    if (v_res := v_resolution(survey.sample_interval(), d=d, v=v)) > threshold:
        logger.warning(
            f"Sample interval of survey is {survey.sample_interval():.2g}. "
            f"Consider upsampling the signal, or use sub-sample precision picking or cross-correlation methods. Else, precision will be about {v_res:.1f}m/s per sample of picking precision"
        )
