import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(r"D:\Viki\Guvi DS\Capstone Projects\Amazon Music Clustering\notebook\final_clustered_music_data.csv")

st.title("🎵 Amazon Music Clustering App")

# -----------------------------
# Feature Selection
# -----------------------------
features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo'
]

# Ensure features exist
features = [col for col in features if col in df.columns]

# -----------------------------
# Scaling (NO NPY FILE)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# -----------------------------
# KMeans Clustering
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# -----------------------------
# Sidebar Filter
# -----------------------------
st.sidebar.header("Filters")
selected_cluster = st.sidebar.selectbox(
    "Select Cluster",
    sorted(df['cluster'].unique())
)

cluster_df = df[df['cluster'] == selected_cluster]

# -----------------------------
# Safe Column Selection
# -----------------------------
display_cols = []

if 'track_name' in df.columns:
    display_cols.append('track_name')
elif 'name' in df.columns:
    display_cols.append('name')

if 'name_artists' in df.columns:
    display_cols.append('name_artists')

# -----------------------------
# Show Sample Songs
# -----------------------------
st.subheader(f"🎧 Sample Songs in Cluster {selected_cluster}")
st.dataframe(cluster_df[display_cols].head(10))

# -----------------------------
# PCA Visualization
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

st.subheader("📊 Cluster Visualization (PCA)")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

for cluster in sorted(df['cluster'].unique()):
    subset = df[df['cluster'] == cluster]
    plt.scatter(subset['PC1'], subset['PC2'], label=f'Cluster {cluster}', alpha=0.6)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.title("PCA Cluster Visualization")

st.pyplot(plt)

# -----------------------------
# Cluster Summary
# -----------------------------
st.subheader("📌 Cluster Feature Summary")

cluster_summary = df.groupby('cluster')[features].mean()
st.dataframe(cluster_summary)

# -----------------------------
# Top Artists per Cluster
# -----------------------------
st.subheader("🎤 Top Artists in Selected Cluster")

if 'name_artists' in df.columns:
    top_artists = (
        cluster_df['name_artists']
        .value_counts()
        .head(5)
    )
    st.write(top_artists)
else:
    st.write("Artist column not available")