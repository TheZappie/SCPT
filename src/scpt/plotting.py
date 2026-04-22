from typing import Sequence, Optional, Iterable

import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
from obspy import Stream, Trace

from scpt.organisation import (
    group_stream,
    stack_traces,
    DEPTH,
    STATION,
    cc,
    PLOT_FILL,
    PLOT_BUTTERFLY,
)
from scpt.picking import without_uncertainty


def stacked_picking_with_uncertainties(
    stream: Stream,
    picks,
    *args,
    **kwargs,
):
    ax = stacked_picking(
        stream=stream,
        picks=without_uncertainty(picks),
        *args,
        **kwargs,
    )
    key_trace = [(key, group) for key, group in group_stream(stream, DEPTH, STATION)]
    groups = [d[0] for d in key_trace if d[0] in picks]
    groups = sorted(groups, key=lambda k: k[0])

    geophone_depths = np.array([d[0] for d in groups])

    first_arrivals = np.asarray([picks[group].nominal_value for group in groups])
    first_arrivals_std = np.asarray([picks[group].std_dev for group in groups])

    ax.fill_betweenx(
        geophone_depths,
        first_arrivals - 2 * first_arrivals_std,
        first_arrivals + 2 * first_arrivals_std,
        alpha=0.5,
        linewidth=0,
        label="95% confidence interval",
    )
    ax.legend(loc="upper right")


def stacked_picking(
    stream: Stream,
    picks,
    ax=None,
    **kwargs,
):
    traces_depth_sorted = [
        (key, group) for key, group in group_stream(stream, DEPTH, STATION)
    ]
    groups = [d[0] for d in traces_depth_sorted]
    geophone_depths = np.array([d[0] for d in groups])
    stacked_traces = [stack_traces(d[1]) for d in traces_depth_sorted]

    first_arrivals = np.asarray([picks[group] for group in groups])

    time_sequences = [trace.times() for trace in stacked_traces]
    ax = plot_arrays(
        [trace.data for trace in stacked_traces],
        time_sequences=time_sequences,
        depths=geophone_depths,
        normalize=True,
        ax=ax,
        **kwargs,
    )
    ax.plot(
        first_arrivals,
        geophone_depths,
        marker=".",
        markerfacecolor="none",
        markeredgecolor="red",
        markersize=3,
        label="picks",
    )

    return ax


def cc_qc(data1, data2):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    delay = cc(data1, data2)
    fig, ax = plt.subplots()
    scale = 0.4
    ax.plot(data1 / data1.max() * scale, label="y1")
    ax.plot(data2 / data2.max() * scale - 1, label="y2")
    ax.plot(
        scipy.ndimage.shift(data1 / data1.max() * scale - 1, -delay, cval=np.nan),
        label="Shifted y1",
        linestyle="--",
    )
    ax.legend()
    fig.show()
    return delay


def plot_traces(
    traces: Sequence[Trace],
    ax=None,
    normalize=True,
    title=None,
    depths: Optional[Iterable[float]] = None,
) -> plt.Axes:
    arrays = [tr.data if isinstance(tr, Trace) else tr for tr in traces]
    times = [trace.times() * 1000 for trace in traces]
    ax = plot_arrays(
        arrays,
        times,
        ax=ax,
        normalize=normalize,
        title=title,
        depths=depths,
    )
    if depths is None:
        ax.set_yticks(
            range(len(arrays)),
            [trace.stats.station + trace.stats.component for trace in traces],
        )
        ax.set_ylabel("Geophone & component")
    return ax


def plot_arrays(
    traces: Sequence[np.ndarray],
    time_sequences: Sequence[np.ndarray],
    ax=None,
    normalize=True,
    title=None,
    depths: Optional[Iterable[float]] = None,
    kind=PLOT_FILL,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots()
    if not normalize:
        vmax = max(trace.max() for trace in traces)
        for trace in traces:
            trace /= vmax
    else:
        for trace in traces:
            if maxi := trace.max():
                trace /= maxi

    scale = -0.5
    if depths is None:
        depths = range(len(traces))
    for offset, trace, timestamps in zip(depths, traces, time_sequences):
        ax.plot(
            timestamps,
            trace * scale + offset,
            "k",
        )
        if kind == PLOT_FILL:
            ax.fill_between(
                timestamps,
                offset,
                trace * scale + offset,
                where=(trace * scale + offset < offset),
                color="k",
            )
        elif kind == PLOT_BUTTERFLY:
            ax.plot(
                timestamps,
                trace * scale + offset,
                color="grey",
                ls="dashed",
                zorder=-2,
            )
        elif kind is None:
            pass
        else:
            raise ValueError(
                f"unknown plot kind: {kind}, needs to be one of {PLOT_FILL, PLOT_BUTTERFLY}"
            )
    ax.set_xlabel("Time [s]")
    ax.set_title(title)
    ax.yaxis.set_inverted(True)
    if depths is not None:
        ax.set_ylabel("Depth [m]")
    return ax


def plot_speed(speed, depth, source_offset=0, offset=0, ax=None):
    """Straight rays assumed"""
    if ax is None:
        fig, ax = plt.subplots()
    arrival_time = (np.asarray(depth) ** 2 + source_offset**2) ** 0.5 / speed + offset
    ax.plot(arrival_time, depth, label=f"arrival time {speed:.0f} m/s")


def plot_layering(
    interfaces: Sequence[float],
    ax=None,
    zorder=-10,
    alpha=0.5,
    color="grey",
    label="Layering",
    **kwargs,
):
    for i, depth in enumerate(interfaces):
        ax.axhline(
            depth,
            zorder=zorder,
            alpha=alpha,
            color=color,
            label=label if i == 0 else None,
            **kwargs,
        )
