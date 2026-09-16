"""Load the breast cancer data and split it into a train and a test set."""

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

CLASS_NAMES = {0: "Benign", 1: "Malignant"}
COLOURS = {"Benign": "#3176B5", "Malignant": "#C97932"}

DATA_FILE = Path(__file__).resolve().parent / "data_breast_cancer.data"
MEASUREMENTS = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave points", "symmetry", "fractal dimension",
]
FEATURES = (
    [f"mean {m}" for m in MEASUREMENTS]
    + [f"{m} error" for m in MEASUREMENTS]
    + [f"worst {m}" for m in MEASUREMENTS]
)


@st.cache_data
def load_data() -> dict:
    """Return {"train": ..., "test": ...}, each with a 0/1 class column.

    The file stores the diagnosis as M or B. Here malignant = 1.
    The split is a fixed, stratified 455 / 114.
    """
    raw = pd.read_csv(DATA_FILE, header=None, names=["id", "diagnosis"] + FEATURES)
    df = raw[FEATURES].astype(float)
    df["class"] = raw["diagnosis"].eq("M").astype(int)
    train, test = train_test_split(df, test_size=0.20, stratify=df["class"], random_state=42)
    return {"train": train, "test": test}
