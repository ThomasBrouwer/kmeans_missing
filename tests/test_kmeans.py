"""
Unit tests for the K-means clustering algorithm implementation with missing
values in the data points.
"""

from kmeans_missing.code.kmeans import KMeans
import numpy, pytest, random


""" Test constructor """
def test_init():
    # Test getting an exception when X and M are different sizes, X is not a 2D array, and K <= 0
    X1 = numpy.ones(3)
    M = numpy.ones((2,3))
    K = 0
    with pytest.raises(AssertionError) as error:
        KMeans(X1,M,K)
    assert str(error.value) == "Input matrix X is not a two-dimensional array, but instead 1-dimensional."
    
    X2 = numpy.ones((4,3,2))
    with pytest.raises(AssertionError) as error:
        KMeans(X2,M,K)
    assert str(error.value) == "Input matrix X is not a two-dimensional array, but instead 3-dimensional."
    
    X3 = numpy.ones((3,2))
    with pytest.raises(AssertionError) as error:
        KMeans(X3,M,K)
    assert str(error.value) == "Input matrix X is not of the same size as the indicator matrix M: (3, 2) and (2, 3) respectively."
    
    X4 = numpy.ones((2,3))
    K1 = 0
    with pytest.raises(AssertionError) as error:
        KMeans(X4,M,K1)
    assert str(error.value) == "K should be greater than 0."
    
    
    # Test getting an exception if a row or column is entirely unknown
    X = numpy.ones((2,3))
    M1 = [[1,1,1],[0,0,0]]
    M2 = [[1,1,0],[1,0,0]]
    K = 1
    
    with pytest.raises(AssertionError) as error:
        KMeans(X,M1,K)
    assert str(error.value) == "Fully unobserved row in X, row 1."
    with pytest.raises(AssertionError) as error:
        KMeans(X,M2,K)
    assert str(error.value) == "Fully unobserved column in X, column 2."
    
    # Test completely observed case
    X = numpy.ones((2,3))
    M = numpy.ones((2,3))
    
    omega_rows = [[0,1,2],[0,1,2]]
    omega_columns = [[0,1],[0,1],[0,1]]    

    kmeans = KMeans(X,M,K)
    assert numpy.array_equal(omega_rows,kmeans.omega_rows)    
    assert numpy.array_equal(omega_columns,kmeans.omega_columns)  
    assert kmeans.no_points == 2
    assert kmeans.no_coordinates == 3
    
    # Test partially observed case
    M = [[1,0,1],[0,1,1]]
    
    omega_rows = [[0,2],[1,2]]
    omega_columns = [[0],[1],[0,1]] 
    
    kmeans = KMeans(X,M,K)  
    assert numpy.array_equal(omega_rows,kmeans.omega_rows)    
    assert numpy.array_equal(omega_columns,kmeans.omega_columns)
    
    
""" Test initialisation """
def test_initialise():
    X = [[1,2,3],[4,5,6]]
    M = [[0,1,1],[1,0,1]]
    K = 2
    seed = 0
    
    kmeans = KMeans(X,M,K)
    kmeans.initialise(seed)
    
    mins = [4.0,2.0,3.0]
    maxs = [4.0,2.0,6.0]
    assert numpy.array_equal(mins,kmeans.mins)
    assert numpy.array_equal(maxs,kmeans.maxs)
    
    mask_centroids = [[1,1,1],[1,1,1]]
    assert numpy.array_equal(mask_centroids,kmeans.mask_centroids)
    
    cluster_assignments = [-1,-1]
    assert numpy.array_equal(cluster_assignments,kmeans.cluster_assignments)    
    
    centroids = [[4.0,2.0,4.2617147424925346],[4.0,2.0,4.2148024123512426]]
    assert numpy.array_equal(centroids,kmeans.centroids)
    
    
""" Run an entire clustering on a simple example. """
def test_cluster():
    ### No missing values case.
    # Points 1,2 will first go to cluster 2, and point 3 to cluster 1.
    # Then point 1 will switch to cluster 1.
    X = [[2,5],[7,5],[2,3]]
    M = numpy.ones((3,2))
    K = 2
    kmeans = KMeans(X,M,K)
    
    kmeans.centroids = [[2.0,2.0],[4.0,5.0]]
    kmeans.mask_centroids = numpy.ones((2,2))
    kmeans.cluster_assignments = [-1,-1,-1]
    
    expected_centroids = [[2.0,4.0],[7.0,5.0]] 
    expected_cluster_assignments = [0,1,0]
    expected_data_point_assignments = [[0,2],[1]]
    expected_clustering_results = [[1,0],[0,1],[1,0]]
    
    kmeans.cluster()
    assert numpy.array_equal(expected_centroids,kmeans.centroids)
    assert numpy.array_equal(expected_cluster_assignments,kmeans.cluster_assignments)
    assert numpy.array_equal(expected_data_point_assignments,kmeans.data_point_assignments)
    assert numpy.array_equal(expected_clustering_results,kmeans.clustering_results)
    
    ### Missing values case.
    # Points 2,3,4 will first go to cluster 2, and point 1 to cluster 1.
    # Then point 2 will switch to cluster 1.
    X = [[2,5],[3,-1],[10,1],[-1,2]]
    M = [[1,1],[1,0],[1,1],[0,1]]
    K = 2
    kmeans = KMeans(X,M,K)
    
    kmeans.centroids = [[2.0,7.0],[3.0,2.0]]
    kmeans.mask_centroids = numpy.ones((2,2))
    kmeans.cluster_assignments = [-1,-1,-1,-1]
    
    expected_centroids = [[2.5,5.0],[10.0,1.5]] 
    expected_cluster_assignments = [0,0,1,1]
    expected_data_point_assignments = [[0,1],[2,3]]
    expected_clustering_results = [[1,0],[1,0],[0,1],[0,1]]
    
    kmeans.cluster()
    assert numpy.array_equal(expected_centroids,kmeans.centroids)
    assert numpy.array_equal(expected_cluster_assignments,kmeans.cluster_assignments)
    assert numpy.array_equal(expected_data_point_assignments,kmeans.data_point_assignments)
    assert numpy.array_equal(expected_clustering_results,kmeans.clustering_results)
    
    ### Cluster with None coordinate.
    # Cluster 1 gets points 1 and 2, cluster 2 gets 3 and 4.
    X = [[2,5],[3,-1],[-1,1],[-1,2]]
    M = [[1,1],[1,0],[0,1],[0,1]]
    K = 2
    kmeans = KMeans(X,M,K)
    
    kmeans.centroids = [[2.0,7.0],[4.0,4.0]]
    kmeans.mask_centroids = numpy.ones((2,2))
    kmeans.cluster_assignments = [-1,-1,-1,-1]
    
    expected_centroids = [[2.5,5.0],[None,1.5]] 
    expected_cluster_assignments = [0,0,1,1]
    expected_data_point_assignments = [[0,1],[2,3]]
    expected_clustering_results = [[1,0],[1,0],[0,1],[0,1]]
    
    kmeans.cluster()
    assert numpy.array_equal(expected_centroids,kmeans.centroids)
    assert numpy.array_equal(expected_cluster_assignments,kmeans.cluster_assignments)
    assert numpy.array_equal(expected_data_point_assignments,kmeans.data_point_assignments)
    assert numpy.array_equal(expected_clustering_results,kmeans.clustering_results)


""" Test reassigning the points to the closest cluster. """
def test_assignment():
    X = numpy.array([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])
    M = numpy.array([[1,1,0],[0,1,0],[1,1,1]])
    K = 2
    kmeans = KMeans(X,M,K)
    
    # Test change - new closest clusters are [0,0,1] - see test_closest_cluster
    centroids = [[1.0,3.0,1.0],[2.0,1.0,3.0]]
    mask_centroids = [[0,1,1],[1,1,0]] 
    cluster_assignments = [0,1,1]
    kmeans.centroids = centroids
    kmeans.mask_centroids = mask_centroids
    kmeans.cluster_assignments = cluster_assignments
    
    change = kmeans.assignment()
    assert change == True
    assert numpy.array_equal([0,0,1],kmeans.cluster_assignments)
    assert numpy.array_equal([[0,1],[2]],kmeans.data_point_assignments)
    
    # Test no change
    centroids = [[1.0,3.0,1.0],[2.0,1.0,3.0]]
    mask_centroids = [[0,1,1],[1,1,0]] 
    cluster_assignments = [0,0,1]
    kmeans.centroids = centroids
    kmeans.mask_centroids = mask_centroids
    kmeans.cluster_assignments = cluster_assignments
    
    change = kmeans.assignment()
    assert change == False
    assert numpy.array_equal([0,0,1],kmeans.cluster_assignments)
    assert numpy.array_equal([[0,1],[2]],kmeans.data_point_assignments)


""" Test updating the cluster centroid coordinates. """
def test_update():
    X = numpy.array([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])
    M = numpy.array([[1,1,0],[0,1,0],[1,1,1]])
    K = 2
    kmeans = KMeans(X,M,K)
    kmeans.data_point_assignments = numpy.array([[0,1],[2]]) #points 0,1 to cluster 0, point 2 to cluster 1
    kmeans.centroids = [[0.0,0.0,0.0],[0.0,0.0,0.0]]
    kmeans.mask_centroids = [[0.0,0.0,0.0],[0.0,0.0,0.0]]
    
    new_centroids = [[1.0,3.5,None],[7.0,8.0,9.0]]
    new_mask_centroids = [[1,1,0],[1,1,1]]
    kmeans.update()
    assert numpy.array_equal(new_centroids,kmeans.centroids)
    assert numpy.array_equal(new_mask_centroids,kmeans.mask_centroids)
    

""" Test random cluster initialisation """
def test_random_cluster_centroid():
    X = [[1,2,3],[4,5,6]]
    M = [[0,1,1],[1,0,1]]
    K = 2
    
    kmeans = KMeans(X,M,K)
    kmeans.mins = [4.0,2.0,3.0]
    kmeans.maxs = [4.0,2.0,6.0]
    
    expected_centroid = [4.0,2.0,4.2617147424925346]
    random.seed(0)
    centroid = kmeans.random_cluster_centroid()
    assert numpy.array_equal(expected_centroid,centroid)
    

""" Test finding the closest cluster for a given data point. """
def test_closest_cluster():
    X = numpy.array([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])
    M = numpy.array([[1,1,0],[0,1,0],[1,1,1]])
    K = 2
    kmeans = KMeans(X,M,K)
    
    # Equal distance for point 0
    centroids = [[1.0,3.0,1.0],[2.0,1.0,3.0]]
    mask_centroids = [[0,1,1],[1,1,0]] 
    kmeans.centroids = centroids
    kmeans.mask_centroids = mask_centroids
    
    expected_closest_cluster_0 = 0 # MSE = 1.0 vs 1.0
    expected_closest_cluster_1 = 0 # MSE = 4.0 vs 16.0
    expected_closest_cluster_2 = 1 # MSE = 44.5 vs 37.0
    closest_cluster_0 = kmeans.closest_cluster(X[0],M[0])
    closest_cluster_1 = kmeans.closest_cluster(X[1],M[1])
    closest_cluster_2 = kmeans.closest_cluster(X[2],M[2])
    
    assert expected_closest_cluster_0 == closest_cluster_0
    assert expected_closest_cluster_1 == closest_cluster_1
    assert expected_closest_cluster_2 == closest_cluster_2
    
    # Test when all MSEs return None (impossible but still testing behaviour)
    centroids = numpy.ones((2,3))
    mask_centroids = [[0,0,1],[0,0,0]]
    kmeans.centroids = centroids
    kmeans.mask_centroids = mask_centroids
    
    expected_closest_cluster = 1
    closest_cluster = kmeans.closest_cluster(X[0],M[0])
    assert expected_closest_cluster == closest_cluster


""" Test the Mean Square Error computation with masks """
def test_compute_MSE():
    # Test case: no overlap
    X = numpy.ones((1,5))
    M = numpy.ones((1,5))
    K = 1
    
    x1 = [1.0,2.0,3.0,4.0,5.0]
    x2 = [5.0,4.5,3.0,2.5,1.0]
    mask1 = [0,1,1,0,0]
    mask2 = [1,0,0,0,1]
    kmeans = KMeans(X,M,K)
    
    output = kmeans.compute_MSE(x1,x2,mask1,mask2)
    assert output == None
    
    # Overlap
    mask1 = [1,1,1,0,1]
    mask2 = [0,1,1,1,1]
    
    expected_output = ( 2.5**2 + 4.0**2 ) / 3.0
    output = kmeans.compute_MSE(x1,x2,mask1,mask2)
    assert expected_output == output


""" Test computing the known coordinate values of all points assigned to a given cluster centroid """
def test_find_known_coordinate_values():
    X = numpy.array([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])
    M = numpy.array([[1,1,0],[0,1,0],[1,1,1]])
    K = 2
    kmeans = KMeans(X,M,K)
    kmeans.data_point_assignments = numpy.array([[0,1],[2]]) #points 0,1 to cluster 0, point 2 to cluster 1
    
    expected_lists_known_coordinate_values_0 = [[1.0],[2.0,5.0],[]]
    expected_lists_known_coordinate_values_1 = [[7.0],[8.0],[9.0]]
    lists_known_coordinate_values_0 = kmeans.find_known_coordinate_values(0)
    lists_known_coordinate_values_1 = kmeans.find_known_coordinate_values(1)
    
    assert numpy.array_equal(expected_lists_known_coordinate_values_0,lists_known_coordinate_values_0)
    assert numpy.array_equal(expected_lists_known_coordinate_values_1,lists_known_coordinate_values_1)
    

""" Test the construction of the clustering matrix. """
def test_create_matrix():
    X = numpy.array([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])
    M = numpy.array([[1,1,0],[0,1,0],[1,1,1]])
    K = 2
    kmeans = KMeans(X,M,K)
    kmeans.cluster_assignments = numpy.array([1,0,1])
    kmeans.create_matrix()
    
    expected_clustering_results = [[0,1],[1,0],[0,1]]
    clustering_results = kmeans.clustering_results
    assert numpy.array_equal(expected_clustering_results,clustering_results)