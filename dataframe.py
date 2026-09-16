"""Show the training data as a table."""

import streamlit as st

from lab_helpers_breast_cancer import load_data


st.write("The use case we consider here is reading the training data as it is.")

st.dataframe(load_data()["train"])
