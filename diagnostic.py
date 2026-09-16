"""Explain the decision tree: the model, one patient's path, feature importance."""

import pandas as pd
import streamlit as st
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.tree import DecisionTreeClassifier, plot_tree

from lab_helpers_breast_cancer import load_data, CLASS_NAMES, FEATURES

data = load_data()
train, test = data["train"], data["test"]

st.write("The use case we consider here is explaining how the tree reaches that diagnosis.")

depth = st.slider("Max depth", 1, 10, 3)
tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(train[FEATURES], train["class"])


st.write("The model")
fig = Figure(figsize=(12, 6), layout="constrained")
FigureCanvasAgg(fig)  # plot_tree needs a renderer
ax = fig.subplots()
plot_tree(tree, ax=ax, feature_names=FEATURES, class_names=["B", "M"],
          filled=True, label="none", fontsize=14)
for text in ax.texts:  # keep only the split, or the class at a leaf
    lines = text.get_text().split("\n")
    text.set_text(lines[0].replace(" <= ", "\n<=\n") if "<=" in lines[0] else lines[-1])
st.pyplot(fig)

st.write("Tick one patient, then press OK")
rows = st.dataframe(test, on_select="rerun", selection_mode="single-row").selection.rows

if st.button("OK") and rows:
    x = test[FEATURES].iloc[rows]
    feature, threshold = tree.tree_.feature, tree.tree_.threshold
    for node in tree.decision_path(x).indices[:-1]:
        name = FEATURES[feature[node]]
        value = x[name].item()
        st.write(f"{name} = {value:.4g}  {'<=' if value <= threshold[node] else '>'}  {threshold[node]:.4g}")
    st.write("Prediction:", CLASS_NAMES[tree.predict(x)[0]], "| Actual:", CLASS_NAMES[test["class"].iloc[rows[0]]])

st.write("Feature importance")
importance = pd.Series(tree.feature_importances_, index=FEATURES)
st.bar_chart(importance[importance > 0].sort_values())
