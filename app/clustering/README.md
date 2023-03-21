# Clustering

## Clustering Metrics

Inertia is defined as the sum of intra-cluster squared Euclidian distance between all data points of the cluster, summed over all clusters.

Calinski and Harabasz score, also known as the Variance Ratio Criterion. The score is defined as ratio of the sum of between-cluster dispersion and of within-cluster dispersion.

Silhouette Coefficient is defined via the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) with the formula (b - a)/max(a, b). The value of the coefficient ranges between -1 and 1. The best value is 1 (well formed clusters), while a value close to 0 means overlapping clusters and -1 that a sample is in the wrong cluster.

The Davies-Bouldin score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances. Thus, clusters which are farther apart and less dispersed will result in a better score. The minimum score is zero, with lower values indicating better clustering.

## Elbow method

The library ```kneed``` can be used to estimate the elbow point. This is typically calculated with inertia as the measure of fitness. First inertia is calculated for different number of clusters and inertia is plotted over the number of clusters. Then the elbow point of the plotted curve is taken as the optimal number of clusters. The optimal number of clusters found by the elbow method should optimally coincide with the maximum of the Silhouette coefficient.
