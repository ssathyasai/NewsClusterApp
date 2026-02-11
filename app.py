# ==============================
# NewsNest-AI
# Streamlit News Clustering App
# ==============================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="NewsNest-AI", layout="wide")

st.title("📰 NewsNest-AI")
st.write("Automatically Group News Articles Using Hierarchical Clustering")

# Upload CSV
uploaded_file = st.file_uploader("Upload News Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="latin1")

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select text column
    text_column = st.selectbox("Select Text Column", df.columns)
    texts = df[text_column].astype(str)

    # Number of clusters
    n_clusters = st.slider("Select Number of Clusters", 2, 10, 4)

    if st.button("Run Clustering"):

        # TF-IDF
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        X = vectorizer.fit_transform(texts)

        # Dimensionality Reduction
        svd = TruncatedSVD(n_components=50, random_state=42)
        X_reduced = svd.fit_transform(X)

        # Hierarchical Clustering
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward"
        )

        labels = model.fit_predict(X_reduced)

        # Silhouette Score
        score = silhouette_score(X_reduced, labels)

        df["Cluster"] = labels

        st.success("Clustering Completed ✅")

        st.write("### Silhouette Score:", round(score, 4))
        st.write("### Cluster Distribution")
        st.write(df["Cluster"].value_counts())

        st.write("### Clustered Data Preview")
        st.write(df.head())

        st.download_button(
            "Download Clustered Dataset",
            df.to_csv(index=False),
            file_name="clustered_output.csv"
        )
