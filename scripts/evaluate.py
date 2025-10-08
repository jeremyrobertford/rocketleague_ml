"""
Evaluate script: evaluate model using extracted features
"""

from rocketleague_ml.models.pipeline import Rocket_League_Pipeline


def main():
    pipeline = Rocket_League_Pipeline()
    clusters = pipeline.model.evaluate()
    pipeline.model.visualize_clusters_on_flat_grid(clusters)
    pipeline.model.visualize_clusters_on_3D_grid(clusters)


if __name__ == "__main__":
    main()
