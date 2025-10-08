from typing import Any, Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from rocketleague_ml.config import FEATURE_LABELS
from rocketleague_ml.models.feature_extractor import Rocket_League_Feature_Extractor
from rocketleague_ml.utils.logging import Logger
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture


class Playstyle_Model:
    def __init__(
        self,
        extractor: Rocket_League_Feature_Extractor,
        model: str = "kmeans",
        params: Dict[str, Any] = {},
        logger: Logger = Logger(),
    ):
        self.model_name = model
        self.model = self.get_cluster_model(model)(**params)
        self.extractor = extractor
        self.logger = logger

    def get_cluster_model(self, name: str):
        """Factory for clustering models."""
        models: Dict[str, Any] = {
            "kmeans": KMeans,
            "dbscan": DBSCAN,
            "agglomerative": AgglomerativeClustering,
            "birch": Birch,
            "gmm": GaussianMixture,
        }
        if name not in models:
            raise ValueError(f"Unknown model {name}. Options: {list(models.keys())}")
        return models[name]

    def evaluate(
        self,
        features: pd.DataFrame | None = None,
        pca_variance: float | None = None,
    ):
        self.logger.print(f"Processing game data files...")

        if features is None:
            features = self.extractor.load_features()

        X = features.drop(columns=["id", "round"], errors="ignore")

        if pca_variance:
            pca = PCA(n_components=0.95)  # keep 95% variance
            X = pca.fit_transform(X)  # type: ignore

        cluster_labels = self.model.fit_predict(X)
        features["cluster"] = cluster_labels

        self.logger.print(f"Done.")
        self.logger.print()
        return features

    def visualize_clusters_on_flat_grid(self, clusters: pd.DataFrame):
        X = clusters.drop(columns=["id", "round"], errors="ignore")
        pca_2d = PCA(n_components=2)
        X_vis = pca_2d.fit_transform(X)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_vis[:, 0], X_vis[:, 1], c=clusters["cluster"], cmap="tab10", s=50
        )
        plt.title("Clusters for Player")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.show()
        return None

    def visualize_clusters_on_3D_grid(self, clusters: pd.DataFrame):
        X = clusters.drop(columns=["id", "round"], errors="ignore")
        pca_3d = PCA(n_components=3)
        X_3d = pca_3d.fit_transform(X)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            X_3d[:, 0],
            X_3d[:, 1],
            X_3d[:, 2],
            c=clusters["cluster"],
            cmap="tab10",
            s=50,
        )

        ax.set_title("3D PCA of Player Clusters")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_zlabel("PCA 3")

        # Optional: legend for clusters
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        plt.show()
        return None

    def summarize_clusters(self, clusters: pd.DataFrame):
        X = clusters.drop(columns=["id", "round"], errors="ignore")
        summary = X.groupby("cluster").median()
        return summary

    def plot_radar_chart_for_clusters(self, clusters: pd.DataFrame):
        # 1. Create mapping concept -> list of features
        concepts: defaultdict[str, List[str]] = defaultdict(list)
        for feature, meta in FEATURE_LABELS.items():
            concept = meta.get("concept", "uncategorized")
            if concept != "metadata":
                concepts[concept].append(feature)

        # 2. Group by cluster and take median
        X = clusters.drop(columns=["id", "round"], errors="ignore")
        summary = X.groupby("cluster").median()

        # 3. Compute concept scores per cluster
        concept_scores = pd.DataFrame(index=summary.index)
        for concept, feats in concepts.items():
            # Only include features that exist in summary
            existing_feats = [f for f in feats if f in summary.columns]
            if existing_feats:
                concept_scores[concept] = summary[existing_feats].median(axis=1)

        # 4. Prepare radar chart
        labels = concept_scores.columns
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # close loop

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        for cluster_id in concept_scores.index:
            values = concept_scores.loc[cluster_id].values
            values = np.concatenate((values, [values[0]]))
            ax.plot(angles, values, "o-", linewidth=2, label=f"Cluster {cluster_id}")
            ax.fill(angles, values, alpha=0.25)

        ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
        ax.set_title("Cluster Playstyle by Concept")
        ax.grid(True)
        ax.legend()
        plt.show()
        return None
