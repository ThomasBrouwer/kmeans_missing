import numpy, random

class KMeans:
    def __init__(self,X,M,K):
        self.X = numpy.array(X,dtype=float)
        self.M = numpy.array(M,dtype=float)
        self.K = K
        
        assert len(self.X.shape) == 2, "Input matrix X is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.X.shape)
        assert self.X.shape == self.M.shape, "Input matrix X is not of the same size as " \
            "the indicator matrix M: %s and %s respectively." % (self.X.shape,self.M.shape)
        assert self.K > 0, "K should be greater than 0."
        
        (self.no_points,self.no_coordinates) = self.X.shape
        
        # Compute lists of which indices are known (from M) for each row/column
        self.omega_rows = [[j for j in range(0,self.no_coordinates) if self.M[i,j]] for i in range(0,self.no_points)]        
        self.omega_columns = [[i for i in range(0,self.no_points) if self.M[i,j]] for j in range(0,self.no_coordinates)]
                    
        # Assert none of the rows or columns are entirely unknown values
        for i,omega_row in enumerate(self.omega_rows):
            assert len(omega_row) != 0, "Fully unobserved row in X, row %s." % i
        for j,omega_column in enumerate(self.omega_columns):
            assert len(omega_column) != 0, "Fully unobserved column in X, column %s." % j
    
    
    """ Initialise the cluster centroids randomly """
    def initialise(self,seed=None):
        if seed is not None:
            random.seed(seed)
        
        # Compute the mins and maxes of the columns
        self.mins = [min([self.X[i,j] for i in self.omega_columns[j]]) for j in range(0,self.no_coordinates)]
        self.maxs = [max([self.X[i,j] for i in self.omega_columns[j]]) for j in range(0,self.no_coordinates)]
        
        # Randomly initialise the cluster centroids
        self.centroids = [self.random_cluster_centroid() for k in xrange(0,self.K)]
        self.cluster_assignments = numpy.array([-1 for d in xrange(0,self.no_points)])
        self.mask_centroids = numpy.ones((self.K,self.no_coordinates))


    # Randomly place a new cluster centroids, picking uniformly between the min and max of each coordinate
    def random_cluster_centroid(self):
        centroid = []
        for coordinate in xrange(0,self.no_coordinates):
            value = random.uniform(self.mins[coordinate],self.maxs[coordinate])
            centroid.append(value)     
        return centroid    
    
            
    """ Perform the clustering, until there is no change """
    def cluster(self):
        iteration = 1
        change = True
        while change:
            print "Iteration: %s." % iteration
            iteration += 1
            
            change = self.assignment()
            self.update()
            
        # At the end, we create a binary matrix indicating which points were assigned to which cluster
        self.create_matrix()


    """ Assign each data point to the closest cluster, and return whether any reassignments were made """
    def assignment(self):
        self.data_point_assignments = [[] for k in xrange(0,self.K)]
        
        change = False
        for d,(data_point,mask) in enumerate(zip(self.X,self.M)):
            old_c = self.cluster_assignments[d]
            new_c = self.closest_cluster(data_point,mask)
            
            self.cluster_assignments[d] = new_c
            self.data_point_assignments[new_c].append(d)
            
            change = (change or old_c != new_c)
        return change
    
    
    # Compute the MSE to each of the clusters, and return the index of the closest cluster.
    # If two clusters are equally far, we return the cluster with the lowest index.
    def closest_cluster(self,data_point,mask_d):
        closest_index = None
        closest_MSE = None
        for c,(centroid,mask_c) in enumerate(zip(self.centroids,self.mask_centroids)):
            MSE = self.compute_MSE(data_point,centroid,mask_d,mask_c)
            
            if (closest_MSE is None) or (MSE is not None and MSE < closest_MSE): #either no closest centroid yet, or MSE is defined and less
                closest_MSE = MSE
                closest_index = c
        return closest_index
    
    
    # Compute the Euclidean distance between the data point and the cluster centroid.
    # If they have no known values in common, we return None (=infinite distance).
    def compute_MSE(self,x1,x2,mask1,mask2):
        overlap = [i for i,(m1,m2) in enumerate(zip(mask1,mask2)) if (m1 and m2)]
        return None if len(overlap) == 0 else sum([(x1[i]-x2[i])**2 for i in overlap]) / float(len(overlap))
        
        
    """ Update the centroids to the mean of the points assigned to it. 
        If for a coordinate there are no known values, we set this cluster's mask to 0 there.
        If a cluster has no points assigned to it at all, we randomly re-initialise it. """
    def update(self):
        for c in xrange(0,self.K):          
            known_coordinate_values = self.find_known_coordinate_values(c)
            
            if known_coordinate_values is None:
                # Randomly re-initialise this point
                self.centroids[c] = self.random_cluster_centroid()
                self.mask_centroids[c] = numpy.ones(self.no_coordinates)
            else:
                # For each coordinate set the centroid to the average, or to None if no values are observed
                for coordinate,coordinate_values in enumerate(known_coordinate_values):
                    if len(coordinate_values) == 0:
                        new_coordinate = None              
                        new_mask = 0
                    else:
                        new_coordinate = sum(coordinate_values) / float(len(coordinate_values))
                        new_mask = 1
                    
                    self.centroids[c][coordinate] = new_coordinate
                    self.mask_centroids[c][coordinate] = new_mask
    
    
    # For a given centroid c, construct a list of lists, each list consisting of
    # all known coordinate values of data points assigned to the centroid.
    # If no points are assigned to a cluster, return None.
    def find_known_coordinate_values(self,c):
        assigned_data_indexes = self.data_point_assignments[c]
        data_points = numpy.array([self.X[d] for d in assigned_data_indexes])
        masks = numpy.array([self.M[d] for d in assigned_data_indexes])
        
        if len(assigned_data_indexes) == 0:
            lists_known_coordinate_values = None
        else: 
            lists_known_coordinate_values = [
                [v for d,v in enumerate(data_points.T[coordinate]) if masks[d][coordinate]]
                for coordinate in xrange(0,self.no_coordinates)
            ]
            
        return lists_known_coordinate_values
        
    
    # Create a binary matrix indicating the clustering (so size [no_points x K])
    def create_matrix(self):
        self.clustering_results = numpy.zeros((self.no_points,self.K))
        for d in range(0,self.no_points):
            c = self.cluster_assignments[d]
            self.clustering_results[d][c] = 1
            
        print [sum(column) for column in self.clustering_results.T]