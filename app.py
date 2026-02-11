# ==========================================
# NewsClustering - Streamlit App
# Safe & Production Ready Version
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="NewsClustering", layout="wide")

st.title("📰 NewsClustering")
st.write("Automatically group similar news articles using Hierarchical Clustering")


# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file, encoding="latin1")
    except Exception:
        st.error("Error reading file. Please upload a valid CSV.")
        st.stop()

    if df.shape[0] == 0:
        st.warning("Uploaded file is empty.")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())


    # ------------------------------
    # Select Text Column
    # ------------------------------
    text_column = st.selectbox("Select Text Column", df.columns)

    texts = df[text_column].astype(str)

    if texts.str.strip().eq("").all():
        st.error("Selected column contains empty text.")
        st.stop()


    # ------------------------------
    # Cluster Selection
    # ------------------------------
    n_clusters = st.slider("Select Number of Clusters", 2, 10, 4)


    # ------------------------------
    # Run Clustering
    # ------------------------------
    if st.button("Run Clustering"):

        # TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english"
        )

        X = vectorizer.fit_transform(texts)

        if X.shape[1] < 2:
            st.error("Not enough unique words for clustering.")
            st.stop()

        # Safe SVD Components
        n_components = min(50, X.shape[1] - 1)

        svd = TruncatedSVD(
            n_components=n_components,
            random_state=42
        )

        X_reduced = svd.fit_transform(X)


        # Hierarchical Clustering
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward"
        )

        labels = model.fit_predict(X_reduced)


        # Silhouette Score (only if valid)
        if len(set(labels)) > 1:
            score = silhouette_score(X_reduced, labels)
            st.success(f"Silhouette Score: {round(score, 4)}")
        else:
            st.warning("Silhouette Score cannot be calculated (only one cluster found).")


        # Add cluster column
        df["Cluster"] = labels

        st.subheader("Cluster Distribution")
        st.write(df["Cluster"].value_counts())

        st.subheader("Clustered Data Preview")
        st.dataframe(df.head())

        # Download button
        st.download_button(
            label="Download Clustered Dataset",
            data=df.to_csv(index=False),
            file_name="clustered_output.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a CSV file to begin.")
