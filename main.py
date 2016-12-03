from image import Image, black_image
from circular_block import Circular_Block
from math import sqrt, ceil
from multiprocessing import Process, Queue
from sys import maxint
import numpy as np
import config, sys

def compare_fn(vec1, vec2):
    i = 0
    size = vec1[1].shape[0]

    while i < size:
        if vec1[1][i] < vec2[1][i]:
            return -1
        if vec1[1][i] > vec2[1][i]:
            return 1
        i += 1

    return -1


def euclidean_distance(vec1, vec2):
    return np.sqrt(((vec1 - vec2)**2).sum())


def match_circular_blocks(min_row, max_row, blocks, distance, matches_queue):
    pos = min_row

    matches = list()

    while pos < max_row:
        (current_block, current_block_features) = blocks[pos]

        extra = 0
        similar_block_list = blocks[pos + 1 : pos + 1 + config.distance_threshold]

        best_distance = maxint
        best_match = None

        for (similar_block, similar_block_features) in similar_block_list:
            # Distance between the centers of both blocks
            center_distance = euclidean_distance( 
                np.array([current_block.center_row, current_block.center_col]), 
                np.array([similar_block.center_row, similar_block.center_col]))

            # Block centers are too close, this block is to be ignored
            if center_distance < 2 * config.block_radius:
                # If there are still blocks after the ones in similar_block_list, 
                # append the next one
                if (config.expanded_matching and 
                        len(blocks) > pos+1+config.distance_threshold+extra):
                    similar_block_list.append( 
                        blocks[pos + 1 + config.distance_threshold + extra] )
                    
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

        pos += 1

        # Do nothing if no suitable match was found
        if best_distance < config.similarity_threshold:
            matches.append( current_block )
            matches.append( best_match )

        #if pos % 1000 == 0:
        #    print float(pos) / len(blocks)

    matches_queue.put( matches )
    print "done"


def compute_features_line(min_row, max_row, image, blocks):
    row = min_row

    local_blocks = list()

    while row < max_row:
        col = config.block_radius
        
        while col + config.block_radius < image.width:
            # Create circular block
            block = Circular_Block(image, row, col)
            # Create a tuple with the block and its features
            block_tuple = ( block, block.features() )
            
            # Add it to the queue of blocks
            local_blocks.append( block_tuple )
            
            col += 1

        row += 1
        
    blocks.put(local_blocks)
    print "done"


if __name__ == "__main__":
    colored_image = None
    
    if len(sys.argv) < 2:
        # Read the image file
        colored_image = Image( raw_input("Type the input file name: ") )
    else:
        colored_image = Image( sys.argv[1])
    
    # Convert it to grayscale
    grayscale_image = Image.grayscale(colored_image)

    del colored_image

    # Apply gaussian_filter
    image = Image.gaussian_filter(grayscale_image, is_grayscale=True, 
        times=config.gaussian_times)

    blocks = list()
    blocks_queue = Queue()

    # Compute the LBP value for each pixel
    image.fill_lbp()

    print image._lbp_img

    # Row variable for iteration
    row = config.block_radius
    # One-past the last row number for circular blocks
    block_row_end = image.height - config.block_radius
    # Increment for range handled by threads
    row_increment = int( round(block_row_end / float(config.num_threads)) )
    
    total_blocks = 0
    while row < block_row_end:
        new_process = Process(target=compute_features_line, 
            args=(row, min(row+row_increment, block_row_end), image, blocks_queue))
        new_process.start()

        total_blocks += min(row+row_increment, block_row_end) - row

        print ("Circular Block features: min row: " + str(row) + 
            "   max row: " + str(min(row+row_increment, block_row_end)))
        
        row += row_increment

    num_blocks = 0

    print "going to wait"
    
    while num_blocks < config.num_threads:
        print blocks_queue.qsize()
        blocks.extend( blocks_queue.get() )
        print blocks_queue.qsize()

        num_blocks += 1              

    print "done waiting"

    del blocks_queue

    print "began sorting"
    # Sort blocks lexicographically based on their feature lists
    blocks.sort(cmp = compare_fn)
    print "finished sorting"

    #blocks_size = len(blocks)
    #row_increment = int( round(blocks_size / float(config.num_threads)) )
    
    #matches_queue = Queue()
    matches = list()

#    print "will begin creating processes"
#
#    row = 0
#    while row < blocks_size:
#        new_process = Process( target=match_circular_blocks, 
#            args=(row, min(row+row_increment, blocks_size), blocks, 
#                euclidean_distance, matches_queue) )
#        new_process.start()
#
#        print ("Circular Block Matching: min row: " + str(row) + 
#            "   max row: " + str(min(row+row_increment, blocks_size)))
#
#        row += row_increment
#
#    print "going to wait"
#    
#    num_blocks = 0
#    while num_blocks < config.num_threads:
#        print matches_queue.qsize()
#        matches.extend( matches_queue.get() )
#        print matches_queue.qsize()
#
#        num_blocks += 1              
#
#    print "done waiting"

    pos = 0     # Position of the current block in the blocks list

    for (current_block, current_block_features) in blocks:
        extra = 0
        similar_block_list = blocks[pos + 1 : pos + 1 + config.distance_threshold]

        best_distance = maxint
        best_match = None

        for (similar_block, similar_block_features) in similar_block_list:
            # Distance between the centers of both blocks
            center_distance = euclidean_distance( 
                np.array([current_block.center_row, current_block.center_col]), 
                np.array([similar_block.center_row, similar_block.center_col]))

            # Block centers are too close, this block is to be ignored
            if center_distance < 2 * config.block_radius:
                # If there are still blocks after the ones in similar_block_list, 
                # append the next one
                if (config.expanded_matching and 
                        len(blocks) > pos+1+config.distance_threshold+extra):
                    similar_block_list.append( 
                        blocks[pos + 1 + config.distance_threshold + extra] )
                    
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

        pos += 1

        # Do nothing if no suitable match was found
        if best_distance < config.similarity_threshold:
            matches.append( current_block )
            matches.append( best_match )

        if pos % 1000 == 0:
            print float(pos) / len(blocks)


    del blocks

    # Paint matched blocks in a black image
    matched_mask = black_image(image.height, image.width)

    for block in matches:
        matched_mask.set_pixel_gray(block.center_row, block.center_col, 255)
        matched_mask.set_pixel_gray(block.center_row-1, block.center_col, 255)
        matched_mask.set_pixel_gray(block.center_row+1, block.center_col, 255)
        matched_mask.set_pixel_gray(block.center_row, block.center_col-1, 255)
        matched_mask.set_pixel_gray(block.center_row, block.center_col+1, 255)

    del matches

    matched_mask.show()

    Image.custom_filter(matched_mask)

    matched_mask.show()