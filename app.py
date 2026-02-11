import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.cluster.hierarchy as sch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ----------------------
# Page Config
# ----------------------
st.set_page_config(page_title="News Topic Discovery", layout="wide")

# Load CSS safely
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

st.markdown('<div class="main-title">🟣 News Topic Discovery Dashboard</div>', unsafe_allow_html=True)

st.write(
    "This system uses **Hierarchical Clustering** to automatically group similar news articles "
    "based on textual similarity."
)

# ----------------------
# Sidebar – Dataset Upload
# ----------------------
st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if not uploaded_file:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# ----------------------
# SAFE CSV LOADING (UTF-8 / Latin-1)
# ----------------------
try:
    df = pd.read_csv(uploaded_file, encoding="utf-8")
except:
    df = pd.read_csv(
        uploaded_file,
        encoding="latin1",
        sep="\t",
        header=None,
        names=["label", "text"]
    )


# If headerless dataset → assign default column names
if df.shape[1] == 2 and df.columns.tolist()[0] == 0:
    df.columns = ["label", "text"]

# Clean column names
df.columns = df.columns.astype(str).str.strip()

# ----------------------
# Detect text column
# ----------------------
text_columns = df.select_dtypes(include="object").columns.tolist()

if len(text_columns) == 0:
    st.error("No text column detected in dataset.")
    st.stop()

text_column = st.sidebar.selectbox("Select Text Column", text_columns)

# ----------------------
# TF-IDF Controls
# ----------------------
st.sidebar.header("📝 Text Vectorization")

max_features = st.sidebar.slider(
    "Maximum TF-IDF Features", 100, 2000, 1000
)

use_stopwords = st.sidebar.checkbox(
    "Use English Stopwords", value=True
)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_option == "Unigrams":
    ngram_range = (1, 1)
elif ngram_option == "Bigrams":
    ngram_range = (2, 2)
else:
    ngram_range = (1, 2)

# ----------------------
# Clustering Controls
# ----------------------
st.sidebar.header("🌳 Hierarchical Clustering")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

subset_size = st.sidebar.slider(
    "Number of Articles for Dendrogram", 20, min(200, len(df)), 50
)

# ----------------------
# Vectorization
# ----------------------
vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english" if use_stopwords else None,
    ngram_range=ngram_range
)

X = vectorizer.fit_transform(
    df[text_column].astype(str)
).toarray()

# ----------------------
# DENDROGRAM
# ----------------------
if st.button("🟦 Generate Dendrogram"):

    st.markdown(
        '<div class="section-title">Dendrogram</div>',
        unsafe_allow_html=True
    )

    subset = X[:subset_size]

    fig = plt.figure(figsize=(10, 6))
    sch.dendrogram(
        sch.linkage(subset, method=linkage_method)
    )
    plt.xlabel("Article Index")
    plt.ylabel("Distance")
    st.pyplot(fig)

    st.info(
        "Large vertical gaps indicate natural topic separation. "
        "Choose cluster count based on these gaps."
    )

# ----------------------
# APPLY CLUSTERING
# ----------------------
num_clusters = st.sidebar.slider(
    "Number of Clusters", 2, 10, 3
)

if st.button("🟩 Apply Clustering"):

    hc = AgglomerativeClustering(
        n_clusters=num_clusters,
        metric="euclidean",
        linkage=linkage_method
    )

    labels = hc.fit_predict(X)

    # ----------------------
    # PCA Visualization
    # ----------------------
    st.markdown(
        '<div class="section-title">Cluster Visualization (PCA)</div>',
        unsafe_allow_html=True
    )

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)

    plot_df = pd.DataFrame({
        "PCA1": reduced[:, 0],
        "PCA2": reduced[:, 1],
        "Cluster": labels.astype(str),
        "Snippet": df[text_column].astype(str).str[:120]
    })

    fig = px.scatter(
        plot_df,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        hover_data=["Snippet"],
        title="2D Projection of News Clusters"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------
    # Silhouette Score
    # ----------------------
    score = silhouette_score(X, labels)

    st.markdown(
        '<div class="section-title">Validation</div>',
        unsafe_allow_html=True
    )

    st.metric("Silhouette Score", round(score, 3))

    st.write(
        "Close to **1** → well-separated clusters  \n"
        "Close to **0** → overlapping clusters  \n"
        "Negative → poor clustering"
    )

    # ----------------------
    # Cluster Summary
    # ----------------------
    st.markdown(
        '<div class="section-title">Cluster Summary</div>',
        unsafe_allow_html=True
    )

    terms = vectorizer.get_feature_names_out()
    summary = []

    for i in range(num_clusters):
        idx = np.where(labels == i)[0]
        mean_tfidf = X[idx].mean(axis=0)
        top_terms = np.array(terms)[mean_tfidf.argsort()[-10:]][::-1]

        summary.append({
            "Cluster ID": i,
            "Articles": len(idx),
            "Top Keywords": ", ".join(top_terms)
        })

    st.dataframe(pd.DataFrame(summary))

    # ----------------------
    # Business Interpretation
    # ----------------------
    st.markdown(
        '<div class="section-title">Business Interpretation</div>',
        unsafe_allow_html=True
    )

    for i in range(num_clusters):
        st.write(
            f"🟣 Cluster {i}: Articles share similar themes and vocabulary."
        )

    st.markdown(
        """
        <div class="insight-box">
        Articles grouped in the same cluster share common themes.
        These clusters can be used for automatic tagging,
        recommendation systems, and editorial organization.
        </div>
        """,
        unsafe_allow_html=True
    )