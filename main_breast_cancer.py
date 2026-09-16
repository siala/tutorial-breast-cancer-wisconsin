import streamlit as st

st.write("# :red[Welcome to my dashboard]")

st.write("## Dataset: [Breast Cancer Wisconsin (Diagnostic)]"
         "(https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic)")

table = st.Page("dataframe.py", title="Data table", icon="💃")
histograms = st.Page("histograms.py", title="Histograms", icon="🎉")
pca = st.Page("pca.py", title="PCA", icon="🤠")
diagnostic = st.Page("diagnostic.py", title="Diagnostic", icon="🔍")
predictive = st.Page("predictive.py", title="Predictive", icon="🌳")

st.navigation([table, histograms, pca, predictive, diagnostic]).run()
