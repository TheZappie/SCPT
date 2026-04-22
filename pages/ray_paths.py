import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

from scpt.inversion_model import Layering
from scpt.organisation import DataMode, Station, RelativeTravelTime, interval_method
from scpt.picking import (
    relative_observations_from_picks,
    shift_picks_to_zero_offset,
    pick_by_cc_stacks,
)
from scpt.plotting import stacked_picking
from scpt.synthetic import create_scpt_survey, RAY_PATH
from scpt.velocity_models import sinus

st.title("Seismic CPT inversion Demonstration")
st.text(
    "Data simulated with ray tracing (using an unshown, fine discretization of a sinus-like velocity model)"
)
st.text("Arrival times picked from the synthetic data. Inversion errors can also originate from this step mirroring actual data processing")
st.text("Inversion using slope method, using true interval data only")

source_offset = st.slider("Source Offset [m]", 0.0, 20.0, 10.0, step=0.1)
n_measurements = st.slider("Number of measurements", 10, 40, 15, step=1)

probe_depths = np.linspace(1, 50, n_measurements)

synthetic_layering = Layering.linspace(0, 52, 200)
slowness = 1 / sinus(synthetic_layering, amplitude=100)

n_samples = 1300
dt = 0.01 / 50

survey = create_scpt_survey(
    n_samples=n_samples,
    dt=dt,
    probe_depths=probe_depths,
    synthetic_layering=synthetic_layering,
    slowness=slowness,
    source_offset=source_offset,
    azimuth=60,
)

single_component_survey = survey.select_component('X')

picks = pick_by_cc_stacks(single_component_survey)

offset = single_component_survey.get_source_offset()
travel_times = relative_observations_from_picks(
    picks, mode=DataMode.true_interval, constant_offset=offset
)

key = sorted(picks)[0]
obs = RelativeTravelTime(
    relative_travel_time=picks[key],
    offset=(0, offset),
    depth=(0, key[0]),
    station=(Station.upper, key[1]),
)
travel_times.insert(0, obs)

result = interval_method(
    relative_observations_from_picks(
        shift_picks_to_zero_offset(picks, offset),
        mode=DataMode.true_interval,
        constant_offset=0,
    )
)
depth_velocities, inverted_velocities = np.array(result).T

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10, 8))

stacked_picking(single_component_survey.stream, picks, ax=ax2)

ax3.set_xlabel("Velocity (m/s)")
ax3.set_ylabel("Depth (m)")

ax3.plot(
    1 / slowness,
    synthetic_layering.midpoints,
    label="Model",
    color="black",
    alpha=0.5,
)
ax3.scatter(
    inverted_velocities, depth_velocities, marker="o", label="Inversion", color="red"
)

ax3.legend()

for trace in single_component_survey.stream:
    ax1.plot(*np.array(trace.stats[RAY_PATH]), color="grey")

ax1.set_xlabel("Horizontal [m]")
ax1.set_ylabel("Depth [m]")
ax2.set_ylabel(None)
ax3.set_ylabel(None)
ax2.legend()

st.pyplot(fig)
