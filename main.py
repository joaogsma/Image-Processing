from image import Image, black_image
from circular_block import Circular_Block
from math import sqrt, ceil
from multiprocessing import Process, Queue
from sys import maxint
import numpy as np
import config, sys

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

    # Increment for range handled by threads
    row_increment = int( round(block_row_end / float(config.num_threads)) )
    
    blocks = list()
    blocks_queue = Queue()
    num_processes = 0

    print "Computing features of circular blocks..."
    while row < block_row_end:
        new_process = Process(target=compute_features_line, 
            args=(row, min(row+row_increment, block_row_end), image, blocks_queue))
        new_process.start()
        row += row_increment
        num_processes += 1

    while num_processes > 0:
        blocks.extend( blocks_queue.get() )
        num_processes -= 1              
    # =========================================================================
    

    del blocks_queue


    # ==================== Sort features ====================
    print "Sorting features..."
    # Sort blocks lexicographically based on their feature lists
    blocks.sort(cmp = compare_fn)
    # =======================================================


    # ==================== Find matches ====================
    print "Finding matches..."
    matches = list()
    pos = 0     # Position of the current block in the blocks list

    for (current_block, current_block_features) in blocks:
        extra = 0
        #similar_block_list = blocks[pos + 1 : pos + 1 + config.distance_threshold]

        best_distance = maxint
        best_match = None

        similar_pos = pos + 1
        limit = min( len(blocks), pos + config.distance_threshold + extra + 1 )

        #if current_block.center_row == 11 and current_block.center_col == 12:
        #    a = (pos, current_block, current_block_features)
        #    print a
        #    
        #if current_block.center_row == 16 and current_block.center_col == 40:
        #    b = (pos, current_block, current_block_features)
        #    print b

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

    #print "pos a: " + str(a[0])
    #print "pos b: " + str(b[0])
    #print
    #print "block a:\n" + str(a[2])
    #print "block b:\n" + str(b[2])
    #print
    ##print "compressed block a:\n" + str(compress_features(a[2]))
    ##print "compressed block b:\n" + str(compress_features(b[2]))
    ##print
    #print "distance: " + str(euclidean_distance( a[2], b[2] ))
    ##print "distance of compressed blocks: " + str(
    ##    euclidean_distance( compress_features(a[2]), compress_features(b[2]) ))
    #print
    #pb_sz = 3
    #for r in range(a[1].center_row-pb_sz, a[1].center_row+pb_sz+1):
    #    row = ''
    #    for c in range(a[1].center_col-pb_sz, a[1].center_col+pb_sz+1):
    #        if r == a[1].center_row and c == a[1].center_col:
    #            row += '|' + str(image._lbp_img[r][c]) + '|\t'
    #        else:
    #            row += str(image._lbp_img[r][c]) + '\t'
    #    print row
    #print
    #for r in range(b[1].center_row-pb_sz, b[1].center_row+pb_sz+1):
    #    row = ''
    #    for c in range(b[1].center_col-pb_sz, b[1].center_col+pb_sz+1):
    #        if r == b[1].center_row and c == b[1].center_col:
    #            row += '|' + str(image._lbp_img[r][c]) + '|\t'
    #        else:
    #            row += str(image._lbp_img[r][c]) + '\t'
    #    print row
    #print (a[1].center_row, a[1].center_col)
    #print (b[1].center_row, b[1].center_col)


    del blocks


    # ==================== Post-processing ====================
    # Paint matched blocks in a black image
    matched_mask = black_image(image.height, image.width)

    for block in matches:
        matched_mask.set_pixel_gray(block.center_row, block.center_col, 255)
        matched_mask.set_pixel_gray(block.center_row-1, block.center_col, 255)
        matched_mask.set_pixel_gray(block.center_row+1, block.center_col, 255)
        matched_mask.set_pixel_gray(block.center_row, block.center_col-1, 255)
        matched_mask.set_pixel_gray(block.center_row, block.center_col+1, 255)

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