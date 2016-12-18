import numpy as np 
import config, random


def euclidean_distance(vec1, vec2):
    return np.sqrt(((vec1 - vec2)**2).sum())

class Kmean:

	def __init__(self, data, n_cluster = 8, n_iter = 20):
		self.clusters = np.array([random.choice(data) for i in range(n_cluster)])
		self.clusters = np.array([c.feature_list for c in self.clusters])
		count_cluster_change = 10
		c_inter = 0
		while (count_cluster_change > 0 and c_inter < n_iter):
			count_cluster_change = 0
			for d in data:
				d.cluster = self.get_cluster(self.clusters, d)
			for i in range(len(self.clusters)):
				data_p_clust = np.array([d.feature_list for d in data if d.cluster == i])
				new_cluster = data_p_clust.mean(axis=0)
				dist = euclidean_distance(new_cluster, self.clusters[i])
				if dist > 1e-4:
					count_cluster_change += 1
					self.clusters[i] = new_cluster

			c_inter += 1 
			print 'runing ' + str(c_inter)

 
 	def get_cluster(self, clusters, data):
 		best = -1
 		distance = 99999
 		for i in range(len(clusters)):
 			d = euclidean_distance(clusters[i], data.feature_list)
 			if d < distance:
 				distance = d
 				best = i
 		return best


        
        