from image import Image, black_image
from circular_block import Circular_Block
from math import sqrt, ceil
from multiprocessing import Process, Queue
from scipy.spatial import KDTree
from sys import maxint
import numpy as np
import config, sys, random
import os

def compress_features(vec):
    compressed = np.zeros( config.P+2, dtype=int );
    
    for i in vec:
        compressed[i] += 1

    return compressed

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


def euclidean_distance(vec1, vec2):
    return np.sqrt(((vec1 - vec2)**2).sum())


def compute_features_line(positions, min_row, max_row, image, blocks):
    row = min_row

    local_blocks = list()

    while row < max_row:
        col = config.block_radius
        
        while col + config.block_radius < image.width:
            # Check if this is a valid position
            if positions[row][col]:
                # Create circular block
                block = Circular_Block(image, row, col)
                # Create a tuple with the block and its features
                block_tuple = ( block, block.features() )
                
                # Add it to the queue of blocks
                local_blocks.append( block_tuple )
            
            col += 1

        row += 1
        
    blocks.put(local_blocks)


def compute_matches_parallel(blocks, features, begin_idx, end_idx, matches_indices_queue):
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


def k_mean(blocks, n_cluster = 8, n_iter = 50):
    # chose n_cluster block at random to be the firsts clusters
    clusters = np.array([random.choice(blocks) for i in range(n_cluster)])
    # keep just the feature
    clusters = np.array([c.feature_list for c in clusters])
    # stop condition
    count_cluster_change = n_cluster
    c_inter = 0
    while (count_cluster_change > 0 and c_inter < n_iter):
        count_cluster_change = 0
        # update the clust of each block
        for block in blocks:
            block.cluster = get_cluster(clusters, block)
        # find new cluster
        for i in range(len(clusters)):
            # Get all blocks with the same cluster
            data_per_clust = np.array([d.feature_list for d in blocks if d.cluster == i])
            new_cluster = data_per_clust.mean(axis=0)
            dist = euclidean_distance(new_cluster, clusters[i])
            # condition to update the cluster
            if dist > 1e-4: 
                count_cluster_change += 1
                clusters[i] = new_cluster

        c_inter += 1 
    return clusters

 
def get_cluster(clusters, data):
    best = -1
    distance = 99999
    for i in range(len(clusters)):
        dist = euclidean_distance(clusters[i], data.feature_list)
        if dist < distance:
            distance = dist
            best = i
    return best


def k_mean_matching(blocks, matches):
    # to mark which pixel has been matched
    matched = np.zeros((image.height, image.width))
    clusters = k_mean(blocks)
    # for each cluster
    for i in range(len(clusters)):
        data_per_clust = np.array([b for b in blocks if b.cluster == i])
        # for each block in the cluster
        for j in range(len(data_per_clust)):
            block = data_per_clust[j]
            # all blocks in the same cluster
            for k in range(j, len(data_per_clust)):
                block_comp = data_per_clust[k]

                if matched[block_comp.center_row][block_comp.center_col] == 0:

                    center_distance = euclidean_distance( 
                            np.array([block.center_row, block.center_col]), 
                            np.array([block_comp.center_row, block_comp.center_col]))

                    if center_distance > 2 * config.block_radius:
                        dist = euclidean_distance(block.feature_list, block_comp.feature_list)

                        if dist < config.similarity_threshold:
                            if matched[block.center_row][block.center_col] == 0:
                                matched[block.center_row][block.center_col] = 1
                                matches.append(block)
                            matched[block_comp.center_row][block_comp.center_col] = 1
                            matches.append(block_comp)

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

    return None, None


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

        best_distance = maxint
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


def find_positions(image):
    spacing = config.blocks_spacing
    epsilon = 1e-6
    spacing_increments = range(-spacing, spacing+1)

    positions = np.ones(shape=(image.height, image.width), dtype=bool)

    row = 0
    while row < image.height:
        col = 0

        while col < image.width:
            # If this position is available, make all nearby positions unavailable
            if positions[row][col]:
                for row_inc in spacing_increments:
                    for col_inc in spacing_increments:
                        # Check if this is a valid position
                        if ( row + row_inc < 0 or row + row_inc >= image.height or
                                col + col_inc < 0 or col + col_inc >= image.width ):
                            continue    # Skip invalid position

                        # Distance between current position and this nearby position
                        distance = euclidean_distance( np.array([row, col]), 
                            np.array([row + row_inc, col + col_inc]) )

                        if distance <= spacing + epsilon and distance > 0:
                            positions[row + row_inc][col + col_inc] = False
            col += 1

        row += 1

    return positions


if __name__ == "__main__":
    colored_image = None
    
    if len(sys.argv) < 2:
        # Read the image file
        colored_image = Image( raw_input("Type the input file name: ") )
    else:
        colored_image = Image( sys.argv[1])
    
    # ==================== Pre-processing ====================
    # Convert it to grayscale
    grayscale_image = Image.grayscale(colored_image)

    del colored_image

    # Apply low-pass filter
    if config.low_pass_filter_type == 'gaussian':
        image = Image.gaussian_filter( grayscale_image, is_grayscale=True, 
            times=config.gaussian_times )
    elif config.low_pass_filter_type == 'mean':
        image = Image.mean_filter( grayscale_image, config.mean_kernel_size )
    elif config.low_pass_filter_type == 'none':
        image = grayscale_image
    else:
        raise Exception('Invalid low-pass filter type')
    # ========================================================


    # ==================== Compute circular block features ====================
    print "Computing LBP values..."
    # Compute the LBP value for each pixel
    image.fill_lbp()

    # Row variable for iteration
    row = config.block_radius
    # One-past the last row number for circular blocks
    block_row_end = image.height - config.block_radius

    # Available positions for circular blocks
    available_positions = find_positions(image)

    # Increment for range handled by threads
    row_increment = int( round(block_row_end / float(config.num_threads)) )
    
    blocks = list()
    features = list()
    blocks_queue = Queue()
    num_processes = 0

    print "Computing features of circular blocks..."
    while row < block_row_end:
        new_process = Process( target=compute_features_line, 
            args=(available_positions, row, min(row+row_increment, block_row_end), 
                image, blocks_queue) )
        new_process.start()
        row += row_increment
        num_processes += 1

    while num_processes > 0:
        #blocks.extend( blocks_queue.get() )
        block_tuples = blocks_queue.get()
        for (b, f) in block_tuples:
            blocks.append(b)
            if config.compress_features:
                features.append( compress_features(f) )
            else:
                features.append( f )
        num_processes -= 1              
    # =========================================================================
    
    
    del blocks_queue
    
    matches = list()

    if config.matching_type == 'lex':
        lexicographical_matching(blocks, features, matches)
    elif config.matching_type == 'kd-tree':
        kd_tree_matching(blocks, features, matches)
    elif config.matching_type == 'k-mean':
        k_mean_matching(blocks, matches)
    else:
        raise Exception('Invalid matching type')


    del blocks

    # ==================== Post-processing ====================
    # Paint matched blocks in a black image
    matched_mask = black_image(image.height, image.width)

    if config.default_match_radius:
        match_radius = 1
        increments = range(-1, 1)
    else:
        match_radius = config.blocks_spacing + 1
        increments = range(-config.blocks_spacing-1, config.blocks_spacing+2)

    for block in matches:
        epsilon = 1e-6
        for row_inc in increments:
            for col_inc in increments:
                row = block.center_row + row_inc
                col = block.center_col + col_inc
                
                # Check if this is a valid position
                if ( row < 0 or row >= matched_mask.height or 
                        col < 0 or col >= matched_mask.width):
                    continue

                distance1 = euclidean_distance( np.array([row+0.5, col+0.5]), 
                    np.array([block.center_row, block.center_col]) )
                distance2 = euclidean_distance( np.array([row+0.5, col-0.5]), 
                    np.array([block.center_row, block.center_col]) )
                distance3 = euclidean_distance( np.array([row-0.5, col+0.5]), 
                    np.array([block.center_row, block.center_col]) )
                distance4 = euclidean_distance( np.array([row-0.5, col-0.5]), 
                    np.array([block.center_row, block.center_col]) )
                
                if ( distance1 <= match_radius + epsilon and 
                        distance2 <= match_radius + epsilon and
                        distance3 <= match_radius + epsilon and
                        distance4 <= match_radius + epsilon ):
                    matched_mask.set_pixel_gray(row, col, 255)


    del matches

    print "Raw matches..."
    matched_mask.show()

    Image.custom_filter(matched_mask)

    print "After filtering..."
    matched_mask.show()

    Image.opening(matched_mask, np.ones((3, 3), np.uint8), True)

    print "After morphological opening..."
    matched_mask.show()
    # =========================================================