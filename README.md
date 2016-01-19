# K-means clustering with missing values
Implementation of the K-means clustering algorithm, for a dataset in which data points can have missing values for some coordinates.

We expect the following arguments to the class:
- Data is expected as a matrix X, where rows are data points, and columns are coordinates. 
- The mask matrix M of the same dimensions as X, having value 0 at M_ij where the coordinate j for data point i is unknown.
- K, the number of clusters.

When computing the distance from each data point to the cluster centroids, we use the Euclidean distance.
For data points with values for certain dimensions missing, we only use the known dimensions (overlap between known values of data point and centroid).
When we recompute the cluster centroids, if all data points for a cluster have unknown values for a certain dimension, we set that dimension to unknown for the centroid as well.

For initialisation, we:
- Compute the maximum and minimum for each coordinate across all data points.
- For each cluster centroid, randomly pick a value uniformly from [min,max] for each coordinate.
- This random initialisation can be seeded by passing a value to the <seed> argument in the function KMeans.initialise(seed).

We iterate until no points change cluster anymore.

The cluster assignments can then be retrieved as an from KMeans.cluster_assignments, or as a matrix from KMeans.clustering_results.

If a cluster becomes empty, we either reassign its centroid randomly ('random'), or assign the point furthest away from its current cluster centroid ('singleton').

Usage is as follows:
- kmeans = KMeans(X,M,K,resolve_empty='singleton')
- kmeans.initialise()
- kmeans.cluster()
- return kmeans.clustering_results
