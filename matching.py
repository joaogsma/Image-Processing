import config
import numpy as np
import os
import random
import sys
from math import sqrt
from multiprocessing import Process, Queue
from scipy.spatial import KDTree

def euclidean_distance(vec1, vec2):
    return np.sqrt(((vec1 - vec2)**2).sum())

# ==============================================================================
# ============================== KD-TREE MATCHING ==============================
# ==============================================================================

def compute_matches_parallel(blocks, features, begin_idx, end_idx, matches_indices_queue):
    sys.setrecursionlimit(1000000)
    kd_tree = KDTree(features)
    matches_indices = list()

    previous_percentage = -1

    pos = begin_idx
    while pos < end_idx:
        found_matches = kd_tree.query_ball_point( features[pos], 
            config.similarity_threshold, eps=1e-6 )
        
        # Exclude nearby blocks
        filtered_matches = filter( 
            lambda x: 
                euclidean_distance( np.array([blocks[x].center_row, blocks[x].center_col]),
                    np.array([blocks[pos].center_row, blocks[pos].center_col]) ) >
                2 * config.block_radius,
            found_matches )

        if len(filtered_matches) > 0:  #Check if there are matches
            matches_indices.extend( filtered_matches )
            matches_indices.append( pos )

        pos += 1

        percentage = int( 100 * float(pos - begin_idx) / (end_idx - begin_idx) )
        if percentage > previous_percentage:
            previous_percentage = percentage
            print ( str('    Process id: ' + str(os.getpid()) + 
                ' - Percentage: ' + str(percentage) + '%') )

    matches_indices_queue.put( matches_indices )


def kd_tree_matching(blocks, features, matches):
    print "Finding matches..."

    increment = int( round(len(blocks) / float(config.num_threads)) )
    matches_indices_queue = Queue()
    matches_indices = set()
    num_processes = 0
    
    pos = 0
    while pos < len(blocks):
        new_process = Process( target = compute_matches_parallel,
            args = (blocks, features, pos, min(pos+increment, len(features)), 
                matches_indices_queue) )
        new_process.start()

        num_processes += 1
        pos += increment

    while num_processes > 0:
        matches_indices.update( matches_indices_queue.get() )
        num_processes -= 1

    del matches[:]
    matches.extend( map(lambda i: blocks[i], matches_indices) )

# ==============================================================================



# ==============================================================================
# ========================== LEXICOGRAPHICAL MATCHING ==========================
# ==============================================================================

def compare_fn(vec1, vec2):
    i = 0
    size = vec1[1].shape[0]

    while i < size:
        if vec1[1][i] < vec2[1][i]:
            return -1
        if vec1[1][i] > vec2[1][i]:
            return 1
        i += 1

    return 0


def lexicographical_matching(blocks, features, matches):
    blocks = zip(blocks, features)
    
    # ==================== Sort features ====================
    print "Sorting features..."
    # Sort blocks lexicographically based on their feature lists
    blocks.sort(cmp = compare_fn)
    # =======================================================


    # ==================== Find matches ====================
    print "Finding matches..."
    pos = 0     # Position of the current block in the blocks list

    for (current_block, current_block_features) in blocks:
        extra = 0
        #similar_block_list = blocks[pos + 1 : pos + 1 + config.distance_threshold]

        best_distance = sys.maxint
        best_match = None

        similar_pos = pos + 1
        limit = min( len(blocks), pos + config.distance_threshold + extra + 1 )

        while similar_pos < min( len(blocks), pos+config.distance_threshold+extra+1 ):
            (similar_block, similar_block_features) = blocks[similar_pos]
                
            # Distance between the centers of both blocks
            center_distance = euclidean_distance( 
                np.array([current_block.center_row, current_block.center_col]), 
                np.array([similar_block.center_row, similar_block.center_col]))


            # Block centers are too close, this block is to be ignored
            if center_distance < 2 * config.block_radius:
                # If there are still blocks after the ones in the range, include
                # the next one
                if (config.expanded_matching and 
                        len(blocks) > pos+config.distance_threshold+extra+1):
                    extra += 1
            # Blocks are far enough apart, they might be matched
            else:
                # Compute the euclidean distance between the feature vectors
                features_distance = euclidean_distance( current_block_features, 
                    similar_block_features )
                    
                # Update pointers if this is a better match
                if features_distance < best_distance:
                    best_match = similar_block
                    best_distance = features_distance

            similar_pos += 1

        pos += 1

        # Do nothing if no suitable match was found
        if best_distance < config.similarity_threshold:
            matches.append( current_block )
            matches.append( best_match )
    # ======================================================

# ==============================================================================



# ==============================================================================
# ============================== K-MEANS MATCHING ==============================
# ==============================================================================

class K_Means_Block:
    def __init__(self, circular_block, features_list, cluster = -1):
        self.block = circular_block
        self.feature_list = features_list


class K_Means_Cluster:
    def __init__(self, cluster_center):
        self.center = cluster_center
        self.data = list()

def get_cluster(clusters, data):
    best = -1
    distance = 99999
    for i in range(len(clusters)):
        dist = euclidean_distance(clusters[i].center, data.feature_list)
        if dist < distance:
            distance = dist
            best = i
    return best


def k_mean(blocks, n_cluster = 8, n_iter = 50):
    # chose n_cluster block at random to be the first clusters
    clusters = np.array([random.choice(blocks) for i in range(n_cluster)])
    # keep just the features vector
    clusters = np.array([K_Means_Cluster(c.feature_list) for c in clusters])
    # stop condition
    count_cluster_change = n_cluster
    c_inter = 0
    while (count_cluster_change > 0 and c_inter < n_iter):
        # Empty the cluster list of assigned points
        for c in clusters:
            del c.data[:]

        count_cluster_change = 0
        # update the clust of each block
        for block in blocks:
            cluster_num = get_cluster(clusters, block)
            clusters[cluster_num].data.append( block )
        # find new cluster
        for i in range(len(clusters)):
            # Get all blocks with the same cluster
            data_per_clust = np.array([b.feature_list for b in clusters[i].data])
            new_center = data_per_clust.mean(axis=0)
            dist = euclidean_distance(new_center, clusters[i].center)
            # condition to update the cluster
            if dist > 1e-4: 
                count_cluster_change += 1
                clusters[i].center = new_center

        c_inter += 1

    return clusters

 
def k_mean_matching(blocks, features, matches):
    
    # Create a list of k-means blocks
    blocks = [K_Means_Block(blocks[i], features[i]) for i in range(len(blocks))]

    # to mark which pixel has been matched
    max_row = max(blocks, key = lambda b: b.block.center_row).block.center_row
    max_col = max(blocks, key = lambda b: b.block.center_col).block.center_col

    matched = np.zeros(
        (max_row+2*config.block_radius+1, max_col+2*config.block_radius+1), 
        dtype=bool )
    
    print "Computing clusters..."
    
    clusters = k_mean(blocks)

    print "Finding matches..."
    
    # for each cluster
    for cluster in clusters:
        data_per_clust = np.array(cluster.data)
        
        # for each block in the cluster
        for j in range(len(data_per_clust)):
            block = data_per_clust[j]
        
            # all blocks in the same cluster
            for k in range(j, len(data_per_clust)):
                block_comp = data_per_clust[k]

                if ( (not matched[block.block.center_row][block.block.center_col]) and
                        (not matched[block_comp.block.center_row][block_comp.block.center_col]) ):

                    center_distance = euclidean_distance( 
                            np.array([block.block.center_row, block.block.center_col]), 
                            np.array([block_comp.block.center_row, block_comp.block.center_col]))

                    if center_distance > 2 * config.block_radius:
                        dist = euclidean_distance(block.feature_list, block_comp.feature_list)

                        if dist < config.similarity_threshold:
                            if not matched[block.block.center_row][block.block.center_col]:
                                matched[block.block.center_row][block.block.center_col] = True
                                matches.append( block.block )
                            matched[block_comp.block.center_row][block_comp.block.center_col] = True
                            matches.append( block_comp.block )

# ==============================================================================