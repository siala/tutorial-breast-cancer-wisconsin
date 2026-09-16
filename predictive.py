"""Train one decision tree and predict a single patient."""

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from lab_helpers_breast_cancer import load_data, CLASS_NAMES, FEATURES

data = load_data()
train, test = data["train"], data["test"]

st.write("The use case we consider here is predicting the diagnosis of a tumour with a decision tree.")

depth = st.slider("Max depth", 1, 10, 3)
tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(train[FEATURES], train["class"])
predicted = tree.predict(test[FEATURES])


st.write(f"Accuracy on the test set: {accuracy_score(test['class'], predicted):.1%}")

st.write("Confusion matrix (rows = actual, columns = predicted)")
st.dataframe(pd.DataFrame(confusion_matrix(test["class"], predicted),
                          index=CLASS_NAMES.values(), columns=CLASS_NAMES.values()))

st.write("Tick one patient, then press OK")
rows = st.dataframe(test, on_select="rerun", selection_mode="single-row").selection.rows

if st.button("OK") and rows:
    st.write("Prediction:", CLASS_NAMES[tree.predict(test[FEATURES].iloc[rows])[0]])
