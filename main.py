import numpy as np
import config
import sys
from circular_block import Circular_Block
from image import Image, black_image
from matching import ( euclidean_distance, kd_tree_matching, 
     lexicographical_matching, k_mean_matching )
from multiprocessing import Process, Queue

def compress_features(vec):
    compressed = np.zeros( config.P+2, dtype=int );
    
    for i in vec:
        compressed[i] += 1

    return compressed

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
        k_mean_matching(blocks, features, matches)
    else:
        raise Exception('Invalid matching type')


    del blocks
    del features

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