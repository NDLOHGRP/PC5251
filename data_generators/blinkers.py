import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import ImageGrid

class gaussian_blinkers:
    def __init__(self, num_clusters=5, cov=np.eye(2,2), 
                 avg_cluster_size=100, scale=50,
                num_iter=50, init='random'):
        """
        Function overview....
        
        Parameters
        ----------
        ....
        
        Returns
        -------
        ....
        """
        self.scale=scale
        self.cov = cov
        self.num_clusters = num_clusters
        self.avg_cluster_size = avg_cluster_size
        self.truemeans = []
        self.truelabels = []
        self.meas_positions = []
        self.likely_means = None
        self.likely_means_history = []
        self.likely_labels = None
        self.likelihoods = None
        
        #Optimization parameters
        self.num_iter = num_iter
        self.init = init
        
    def gen_2D_gaussian_cluster(self, num_samples_from_cluster):
        """
        Insert your Docstring
        """
        mean = self.scale*np.random.rand(2)
        positions = np.random.multivariate_normal(mean, self.cov, size=num_samples_from_cluster)
        return (mean, positions)

    def gen_data(self, rand_seed=None):
        """
        Insert your Docstring
        """
        labels = []
        means = []
        pos = []
        
        #For each cluster generate different number of samples  
        if rand_seed is not None:
            np.random.seed(rand_seed)
        for i in range(self.num_clusters):
            #Generate random cluster size for each cluster
            cluster_size = int(np.random.normal(self.avg_cluster_size, 
                                                np.sqrt(self.avg_cluster_size)))
            m,p = self.gen_2D_gaussian_cluster(cluster_size)
            labels.append(i*np.ones(cluster_size, dtype=int))
            means.append([m])
            pos.append(p)
        np.random.seed()
        
        self.truemeans = np.concatenate(means, axis=0)
        self.truelabels = np.concatenate(labels, axis=0)
        self.meas_positions = np.concatenate(pos, axis=0)
        
        #Randomize positions
        num_samples = len(self.truelabels)
        rand_ordering = np.random.permutation(num_samples)
        self.truelabels = self.truelabels[rand_ordering]
        self.meas_positions = self.meas_positions[rand_ordering]

    def plot_positions(self, ax=None, plot_means="true", in_figsize=(5,5)):
        """
        Insert your Docstring
        """
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=in_figsize)
        ax.scatter(self.meas_positions[:,0], 
                    self.meas_positions[:,1], 
                    c=self.truelabels, s=1)
        if plot_means == "true":
            ax.scatter(self.truemeans[:,0], self.truemeans[:,1], c='black', s=30)
        elif plot_means == "likely" and (self.likely_means_history is not None):
            ax.scatter(self.likely_means[:,0], self.likely_means[:,1], c='black', s=30)
            #means_history = np.asarray(self.likely_means_history)
            #for i in range(self.num_clusters):
            #    plt.plot(means_history[:,i,0], means_history[:,i,1], '-', lw=1)

    def gen_rand_initial_points(self):
        """
        Insert your Docstring
        """
        k_points_loc = np.random.choice(len(self.meas_positions), size=self.num_clusters)
        return self.meas_positions[k_points_loc]
    
    def gen_rand_initial_points_plus_plus(self):
        """
        Insert your Docstring
        """
        loc = self.meas_positions[np.random.choice(len(self.meas_positions))]
        points = [loc]
        for i in range(1, self.num_clusters):
            dist_to_picked_means = np.asarray([np.linalg.norm(self.meas_positions-locs,axis=1) for locs in points])
            min_dist_to_picked_means = dist_to_picked_means.min(axis=0)
            new_loc = self.meas_positions[np.argsort(min_dist_to_picked_means)[-1]]
            points.append(new_loc)
        return np.asarray(points)
                
    def init_clustering(self):
        """
        Insert your Docstring
        """
        if self.init == 'random':
            k_points = self.gen_rand_initial_points()
        elif self.init == 'kpp':
            k_points = self.gen_rand_initial_points_plus_plus()
        else:
            print("Invalid init option. Pick 'random' or 'kpp'")
            return 
            
        self.likely_means = k_points
        self.compute_most_likely_labels()
        
    def compute_most_likely_labels_1(self):
        """
        Insert your Docstring
        """
        resp = []
        for m in self.likely_means:
            z = self.meas_positions-m
            resp.append(np.einsum("ij,ij->i", np.dot(z,self.cov),z))
        self.likelihoods = np.asarray(resp)
        self.likely_labels = np.argmin(resp, axis=0)

    def compute_most_likely_labels(self):
        """
        Insert your Docstring
        """
        resp = []
        for m in self.likely_means:
            z = self.meas_positions-m
            resp.append(np.linalg.norm(z, axis=1))
        self.likelihoods = np.asarray(resp)
        self.likely_labels = np.argmin(resp, axis=0)

    
    def compute_most_likely_means(self):
        """
        Insert your Docstring
        """
        self.likely_means_history.append(self.likely_means.copy())
        for i in range(self.num_clusters):
            self.likely_means[i] = self.meas_positions[self.likely_labels==i].mean(axis=0)

    def my_fit(self, init=None, num_iter=None):
        """
        Insert your Docstring
        """
        if init is not None:
            self.init = init
        if num_iter is not None:
            self.num_iter = num_iter
        #Insert your algorithm for fitting!
            
    def sklearn_k_means(self):
        """
        Insert your Docstring
        """
        kmeans = KMeans(n_clusters=self.num_clusters, init='k-means++').fit(self.meas_positions)
        self.likely_labels = kmeans.labels_
        self.likely_means = kmeans.cluster_centers_

    def plot_relative_errors(self):
        pass
