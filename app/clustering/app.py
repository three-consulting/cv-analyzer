"""Simple Gradio app for demoing semantic search."""
from typing import Tuple

import gradio as gr
import pandas as pd

from clustering import Clusterer
from data_types import ClusteringModelEnum, ModelEnum
from embedding_generator import read_csv_data

df = read_csv_data()

def clustering(model: str, clustering_model: str, num_of_classes: int):
    """Generate cluster visualizations with UMAP
    Args:
        model (str): embedding model
        clustering_model (str): clustering model
        num_of_classes (int): how many clusters to form
    Returns:
        dataframe: dataframe with reduced data representation and cluster labels
    """
    clusterer = Clusterer(model_name=ModelEnum(model))
    clusterer.load_embeddings(df)
    clusterer.cluster_data(
        clustering_model=ClusteringModelEnum(clustering_model),
        data=clusterer.embeddings,
        n_classes=num_of_classes,
    )
    clusterer.reduce_dimension(clusterer.embeddings)
    df_with_labels = pd.DataFrame(clusterer.reduced_embedding, columns=["x", "y"])
    df_with_labels["int_label"] = clusterer.cluster_labels
    df_with_labels["label"] = df_with_labels["int_label"].astype(str)
    df_with_labels["category"] = df["jobs"]
    df_with_labels["size"] = 2
    return df_with_labels.drop(["int_label"], axis=1), df_with_labels.drop(
        ["x", "y", "size", "label"], axis=1
    ).rename(columns={"int_label": "label"}).sort_values(by="label")


clustering_demo = gr.Interface(
    fn=clustering,
    inputs=[
        gr.components.Dropdown(
            label="Embedding Model", choices=[x.value for x in ModelEnum]
        ),
        gr.components.Dropdown(
            label="Clustering Model", choices=[x.value for x in ClusteringModelEnum]
        ),
        gr.components.Slider(2, 30, value=8, label="num_of_clusters", step=1),
    ],
    outputs=[
        gr.ScatterPlot(
            x="x",
            y="y",
            color="label",
            tooltip="category",
            title="Cluster data",
            color_legend_title="Cluster label",
            caption="UMAP representation of clusters",
            height=500,
            width=500,
            size="size",
        ),
        gr.components.DataFrame(
            label="Clustered data",
            wrap=True,
        ),
    ],
    examples=[[ModelEnum.ADA2, ClusteringModelEnum.SPECTRAL, 8]],
    cache_examples=True,
    title="Clustering Demo",
    description="""This demo implements simple clustering and data
    visualization with UMAP.""",
)


def elbow_point() -> Tuple[str, str, str]:
    """Compute metrics and generate the respective plots to visualize.
    Returns:
        Tuple[str, str, str]: paths to the plots to show.
    """
    clusterer = Clusterer(model_name=ModelEnum.ADA2)
    clusterer.load_embeddings(df)
    clusterer.compute_metrics(clusterer.embeddings)
    clusterer.plot_metrics()
    clusterer.elbow_method()

    return "data/metrics_plot.png", "data/normalized_metrics_plot.png", "data/elbow_plot.png"


elbow_demo = gr.Interface(
    fn=elbow_point,
    inputs=[],
    outputs=[gr.Image(), gr.Image(), gr.Image()],
    title="Elbow Point Demo",
    description="""This Demo looks for an elbow point to
    determine the optimal number of clusters.""",
)


demo = gr.TabbedInterface(
    [clustering_demo, elbow_demo],
    ["Clustering", "Elbow method"],
)

if __name__ == "__main__":
    demo.launch(share=False)
