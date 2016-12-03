from image import Image, black_image
from circular_block import Circular_Block
from math import sqrt, ceil
from multiprocessing import Process, Queue
from sys import maxint
import numpy as np
import config, sys

def comparation_funcion(vec1, vec2):
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

    if len(sys.argv) < 2:
        # Read the image file
        colored_image = Image( raw_input("Type the input file name: ") )
    else:
        colored_image = Image( sys.argv[1])
    # Convert it to grayscale
    grayscale_image = Image.grayscale(colored_image)

    # Apply gaussian_filter
    image = Image.gaussian_filter(grayscale_image, is_grayscale=True, 
        times=config.gaussian_times)

    blocks = list()
    queue = Queue()

    # Compute the LBP value for each pixel
    image.fill_lbp()
    # Row variable for iteration
    row = config.block_radius
    # One-past the last row number for circular blocks
    block_row_end = image.height - config.block_radius
    # Increment for range handled by threads
    row_increment = int( round(block_row_end / float(config.num_threads)) )
    
    processes = list()

    total_blocks = 0
    while row < block_row_end:
        new_process = Process(target=compute_features_line, 
            args=(row, min(row+row_increment, block_row_end), image, queue))
        processes.append( new_process )
        new_process.start()

        total_blocks += min(row+row_increment, block_row_end) - row

        print ("Circular Block features: min row: " + str(row) + 
            "   max row: " + str(min(row+row_increment, block_row_end)))
        
        row += row_increment

    num_blocks = 0

    print "going to wait"
    
    while num_blocks < config.num_threads:
        print queue.qsize()
        blocks.extend( queue.get() )
        print queue.qsize()

        num_blocks += 1              

    print "done waiting"

    # Sort blocks lexicographically based on their feature lists
    blocks.sort(cmp = comparation_funcion)
    matches = list()

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


    # Paint matched blocks in a black image
    matched_mask = black_image(image.height, image.width)

    for block in matches:
        matched_mask.set_pixel_gray(block.center_row, block.center_col, 255)
        matched_mask.set_pixel_gray(block.center_row-1, block.center_col, 255)
        matched_mask.set_pixel_gray(block.center_row+1, block.center_col, 255)
        matched_mask.set_pixel_gray(block.center_row, block.center_col-1, 255)
        matched_mask.set_pixel_gray(block.center_row, block.center_col+1, 255)

    matched_mask.show()

    Image.custom_filter(matched_mask)

    matched_mask.show()