# 🎵 Amazon Music Clustering Project

## 📌 Project Overview

This project focuses on clustering songs from an Amazon Music dataset based on their audio features. The goal is to group similar songs together and understand different music patterns such as energetic tracks, calm instrumentals, and vocal-heavy content.

The project follows a complete machine learning pipeline:

* Data preprocessing
* Feature engineering
* Scaling and outlier handling
* Clustering using multiple algorithms
* Visualization and interpretation
* Streamlit app deployment

---

## 📂 Dataset

The dataset contains song-level information including:

* Danceability
* Energy
* Loudness
* Speechiness
* Acousticness
* Instrumentalness
* Liveness
* Valence
* Tempo
* Artist names and genres

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing

* Removed unnecessary columns
* Handled missing values
* Ensured correct data types
* Selected only relevant numerical audio features

---

### 2️⃣ Feature Engineering

* Focused on core audio features influencing music behavior
* Included `duration_ms` initially but excluded from clustering to avoid bias

---

### 3️⃣ Scaling & Outlier Handling

* Applied **StandardScaler** for normalization
* Used clipping technique to reduce extreme outliers

---

### 4️⃣ Clustering Techniques

#### 🔹 KMeans Clustering (Primary Model)

* Applied KMeans to group songs into 3 clusters
* Optimal K chosen using Elbow Method & Silhouette Score

**Result:**

* Cluster 0 → Instrumental / Calm
* Cluster 1 → Energetic / Mainstream
* Cluster 2 → Live / Vocal Heavy

---

#### 🔹 DBSCAN (Additional)

* Density-based clustering used to identify natural groupings
* Helps detect noise and outliers
* No deep analysis performed (used for experimentation only)

---

#### 🔹 Hierarchical Clustering (Additional)

* Agglomerative clustering applied
* Useful for understanding cluster hierarchy via dendrogram
* No detailed evaluation performed

---

## 📊 Visualization

### PCA (Principal Component Analysis)

* Reduced dimensions to 2D
* Visualized cluster separation clearly

### t-SNE

* Used for advanced visualization of high-dimensional data
* Showed non-linear cluster boundaries

---

## 📈 Cluster Interpretation

### Cluster 0 — Instrumental / Calm

* Lower energy and loudness
* Higher acousticness
* Mostly instrumental or soft music

### Cluster 1 — Energetic / Mainstream

* High energy and loudness
* Higher tempo
* Popular, mainstream tracks

### Cluster 2 — Live / Vocal Heavy

* High speechiness and liveness
* Strong vocal presence
* Includes live recordings and spoken content

---

## 🖥️ Streamlit Application

An interactive web app was built using Streamlit with the following features:

* Cluster selection filter
* Display of sample songs
* PCA-based visualization
* Top artists per cluster
* Cluster feature summary

---

## 🚀 How to Run the Project

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd amazon-music-clustering
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit

---

## 📌 Key Learnings

* Importance of feature scaling in clustering
* Choosing optimal number of clusters
* Understanding cluster interpretability
* Comparing different clustering techniques
* Building end-to-end ML projects with deployment

---

## 🔮 Future Improvements

* Add recommendation system based on clusters
* Tune DBSCAN parameters for better results
* Use advanced embeddings for music similarity
* Deploy app on cloud (Streamlit Cloud / AWS)

---

## ✅ Conclusion

This project successfully groups songs into meaningful clusters using unsupervised learning techniques. It demonstrates how machine learning can be used to understand music patterns and build interactive applications.

---
