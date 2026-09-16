"""PCA page: 2D and 3D projections of the training data."""

import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from lab_helpers_breast_cancer import load_data, CLASS_NAMES, COLOURS

df = load_data()["train"]
labels = df["class"].map(CLASS_NAMES).to_numpy()
Z = PCA(n_components=3).fit_transform(StandardScaler().fit_transform(df.drop(columns="class")))
names = {"x": "PCA 1", "y": "PCA 2", "z": "PCA 3", "color": "Diagnosis"}

st.write("The use case we consider here is projecting the 30 measurements onto 2 and 3 components.")

st.plotly_chart(px.scatter(x=Z[:, 0], y=Z[:, 1], color=labels,
                           color_discrete_map=COLOURS, labels=names))

st.plotly_chart(px.scatter_3d(x=Z[:, 0], y=Z[:, 1], z=Z[:, 2], color=labels,
                              color_discrete_map=COLOURS, labels=names).update_traces(marker_size=3))
