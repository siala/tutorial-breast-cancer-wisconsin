"""Histogram and boxplot of one feature, split by class."""

import numpy as np
import streamlit as st
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from lab_helpers_breast_cancer import load_data, CLASS_NAMES, COLOURS, FEATURES

df = load_data()["train"]
groups = {name: df[df["class"] == value] for value, name in CLASS_NAMES.items()}

st.write("The use case we consider here is comparing one measurement between benign and malignant tumours.")

feature = st.selectbox("Feature", FEATURES)

st.write(f"Selected attribute : {feature}")
edges = np.histogram_bin_edges(df[feature], bins=st.slider("Bins", 5, 60, 20))

fig = Figure(figsize=(8, 4), layout="constrained")
ax1, ax2 = fig.subplots(1, 2, width_ratios=[3, 1])

for name, group in groups.items():
    colour = COLOURS[name]
    ax1.hist(group[feature], bins=edges, density=True, histtype="stepfilled",
             facecolor=to_rgba(colour, 0.25), edgecolor=colour,
             linewidth=1.3, label=f"{name}, n = {len(group)}")

ax1.set(xlabel="Observed value", ylabel="Density", xlim=(edges[0], edges[-1]))
ax1.legend(frameon=False)

ax2.boxplot([g[feature] for g in groups.values()], tick_labels=list(groups))
ax2.set_ylabel("Observed value")

st.pyplot(fig)
