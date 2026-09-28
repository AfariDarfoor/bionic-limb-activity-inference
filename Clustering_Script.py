
'''This pipeline performs data loading, signal preprocessing, and time-frequency feature extraction. 
Features are standardized before being evaluated across multiple clustering algorithms, with performance metrics generated for comparison.'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score

#  Initialization

'''Please add the path to your file thaqt contains your motion data to load it'''
csv_path = 'path/to/your/sensor_data/.csv'
sampling_rate = 148  # Hz Use input of 148 when analysing trigno data, otherwise use sampling rate of 10 when analyzing smartphone data
window_duration = 2  # seconds
top_n_harmonics = 3
window_size = sampling_rate * window_duration

bands = {
    'low': (0.5, 3),
    'mid': (3, 8),
    'high': (8, 15)
}

# Load and preprocess
df = pd.read_csv(csv_path)
df['Magnitude'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
df = df.dropna().sort_values(by='Activity')

features = []
labels = []

# Feature Extraction 
for activity in df['Activity'].unique():
    activity_df = df[df['Activity'] == activity]
    mags = activity_df['Magnitude'].values
    num_windows = len(mags) // window_size

    for i in range(num_windows):
        segment = mags[i * window_size: (i + 1) * window_size]
        if len(segment) < window_size:
            continue

        # Time-domain features
        time_feats = [
            np.mean(segment),
            np.std(segment),
            skew(segment),
            kurtosis(segment),
            np.min(segment),
            np.max(segment),
            np.median(segment),
        ]

        # Frequency-domain features
        freqs, psd = welch(segment, fs=sampling_rate, nperseg=len(segment))
        peaks, _ = find_peaks(psd)
        top_peaks = sorted(peaks, key=lambda x: psd[x], reverse=True)[:top_n_harmonics]
        harmonic_freqs = [freqs[p] for p in top_peaks]
        while len(harmonic_freqs) < top_n_harmonics:
            harmonic_freqs.append(0)

        spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
        band_powers = [np.sum(psd[np.logical_and(freqs >= b[0], freqs < b[1])]) for b in bands.values()]

        freq_feats = harmonic_freqs + [spectral_centroid] + band_powers
        features.append(time_feats + freq_feats)
        labels.append(activity)

# Prepare for clustering 
X = np.array(features)
label_encoder = LabelEncoder()
y_true = label_encoder.fit_transform(labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#modifications to remove top and right bar begins here paper version
#Plot Ground Truth with Shapes and Colors
marker_shapes = ['s', 'o', 'D', '^', 'P', 'X', '*', 'v', '<', '>']  # max 10 unique shapes
unique_labels = np.unique(y_true)
label_names = label_encoder.inverse_transform(unique_labels)

# Same colours but with grayscale-distinguishable brightness
distinct_colors = [
    "#0B3D91",  # deep blue (dark)
    "#FFD700",  # bright yellow
    "#6CC96F",  # light green
    "#D62728",  # classic medium red
]

plt.figure(figsize=(6, 5))

for i, label in enumerate(unique_labels):
    shape = marker_shapes[i % len(marker_shapes)]
    activity_name = label_names[i]

    plt.scatter(
        X_pca[y_true == label, 0],
        X_pca[y_true == label, 1],
        label=activity_name,
        marker=shape,
        s=40,
        color=distinct_colors[i % len(distinct_colors)]
    )

# Axis & styling adjustments
plt.xlabel("PCA 1", fontsize=16)
plt.ylabel("PCA 2", fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend(
    title="Activity",
    loc='upper right',
    fontsize=16,
    title_fontsize=16
)

plt.tight_layout()
plt.savefig("GroundTruth_by_Activity_GrayscaleSafe.svg", format='svg')
plt.show()



#paper version 2, extended x and y axis

#Plot Ground Truth with Shapes and Colors
marker_shapes = ['s', 'o', 'D', '^', 'P', 'X', '*', 'v', '<', '>']  # max 10 unique shapes
unique_labels = np.unique(y_true)
label_names = label_encoder.inverse_transform(unique_labels)

#Same COLOURS but with grayscale-distinguishable brightness
distinct_colors = [
    "#0B3D91",  # deep blue (dark)
    "#FFD700",  # bright yellow
    "#6CC96F",  # light green
    "#D62728",  # classic medium red
]

plt.figure(figsize=(6, 5))

for i, label in enumerate(unique_labels):
    shape = marker_shapes[i % len(marker_shapes)]
    activity_name = label_names[i]

    plt.scatter(
        X_pca[y_true == label, 0],
        X_pca[y_true == label, 1],
        label=activity_name,
        marker=shape,
        s=40,
        color=distinct_colors[i % len(distinct_colors)]
    )

# Axis limits added here
plt.xlim(-6, 10)
plt.ylim(-4, 10)

#Axis & styling adjustments
plt.xlabel("Principal Component 1", fontsize=16)
plt.ylabel("Principal Component 2", fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend(
    title="Activity",
    loc='upper right',
    bbox_to_anchor=(1.0, 1.1),
    fontsize=16,
    title_fontsize=16
)

plt.tight_layout()
plt.savefig("GroundTruth_by_Activity_GrayscaleSafe.svg", format='svg')
plt.show()




# Plot Ground Truth with Shapes and Colors
marker_shapes = ['s', 'o', 'D', '^', 'P', 'X', '*', 'v', '<', '>']  # max 10 unique shapes
unique_labels = np.unique(y_true)
label_names = label_encoder.inverse_transform(unique_labels)

plt.figure(figsize=(6, 5))

for i, label in enumerate(unique_labels):
    shape = marker_shapes[i % len(marker_shapes)]
    activity_name = label_names[i]
    plt.scatter(X_pca[y_true == label, 0], X_pca[y_true == label, 1],
                label=activity_name,
                marker=shape,
                s=40)

plt.title("Ground Truth ",  pad=20 )
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Activity", loc='upper right', fontsize='small', title_fontsize='small')
plt.tight_layout()
plt.savefig("GroundTruth_by_Activity_Shapes.svg", format='svg')
plt.show()

# Clustering algorithms (with little to no modifications to hyper-parameters, off-the-shelf)
clustering_algos = {
    'KMeans': KMeans(n_clusters=4, random_state=50),
    'DBSCAN': DBSCAN(eps=0.8, min_samples=3),
    'Agglomerative': AgglomerativeClustering(n_clusters=4),
    'GMM': GaussianMixture(n_components=4, random_state=50),
    'OPTICS': OPTICS(min_samples=15, xi=0.05, min_cluster_size=0.1)
}

#Evaluate and plot clustering results
for name, algo in clustering_algos.items():
    if name == 'GMM':
        cluster_labels = algo.fit_predict(X_scaled)
    else:
        cluster_labels = algo.fit_predict(X_scaled)

    # Metrics
    ari = adjusted_rand_score(y_true, cluster_labels)
    ri = rand_score(y_true, cluster_labels)
    nmi = normalized_mutual_info_score(y_true, cluster_labels)

    # Print metrics
    print(f"{name} Clustering:")
    print(f"  Adjusted Rand Index (ARI): {ari:.3f}")
    print(f"  Rand Index (RI): {ri:.3f}")
    print(f"  Normalized Mutual Info (NMI): {nmi:.3f}")
    print("-" * 50)

    # Plot clusters (color only for now)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', s=40)
    plt.title(f'{name} Clustering')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(scatter, label='Cluster Label')
    plt.tight_layout()
    plt.savefig(f"{name}_Clustering.svg", format='svg')
    plt.show()
