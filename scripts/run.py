"""
Run script: overwrite everything and run it all
"""

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from rocketleague_ml.models.pipeline import Rocket_League_Pipeline
from rocketleague_ml.models.playstyle_model import Playstyle_Model


def visualize_clusters_on_flat_grid(clusters: pd.DataFrame):
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


def visualize_clusters_on_3D_grid(clusters: pd.DataFrame):
    X = clusters.drop(columns=["id", "round"], errors="ignore")
    pca_3d = PCA(n_components=3)
    X_3d = pca_3d.fit_transform(X)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=clusters["cluster"], cmap="tab10", s=50
    )

    ax.set_title("3D PCA of Player Clusters")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")

    # Optional: legend for clusters
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    plt.show()


def summarize_clusters(clusters: pd.DataFrame):
    X = clusters.drop(columns=["id", "round"], errors="ignore")
    summary = X.groupby("cluster").median()
    print("Cluster summary (median values per feature):")
    print(summary.T)


def main():
    pipeline = Rocket_League_Pipeline()
    pipeline.preprocessor.convert_replays(overwrite=True)
    pipeline.processor.process_games(overwrite=True)
    pipeline.extractor.extract_features(main_player="RL_LionHeart", overwrite=True)
    pipeline.model = Playstyle_Model(
        extractor=pipeline.extractor, model="kmeans", params={"n_clusters": 3}
    )
    clusters = pipeline.model.evaluate()

    visualize_clusters_on_flat_grid(clusters)
    visualize_clusters_on_3D_grid(clusters)
    summarize_clusters(clusters)


if __name__ == "__main__":
    main()
