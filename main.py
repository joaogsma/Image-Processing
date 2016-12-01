from image import Image, black_image
from circular_block import Circular_Block
from math import sqrt
from sys import maxint
import config
import threading

def euclidean_distance(vec1, vec2):
    acc = 0

    for i in zip(vec1, vec2):
        acc += (i[0] - i[1])**2

    return sqrt(acc)


# Read the image file
colored_image = Image( raw_input("Type the input file name: ") )
# Convert it to grayscale
grayscale_image = Image.grayscale(colored_image)

# Apply gaussian_filter
image = Image.gaussian_filter(grayscale_image, is_grayscale=True, 
    times=config.gaussian_times)

blocks = list()

# Compute the LBP value for each pixel
#image.fill_lbp()

row = config.block_radius
while row + config.block_radius < image.height:
    col = config.block_radius
    
    while col + config.block_radius < image.width:
        # Create circular block
        block = Circular_Block(image, row, col)
        # Create a tuple with the block and its features
        block_tuple = ( block, block.features(True) )
        # Add it to the list of blocks
        blocks.append( block_tuple )

        col += 1
    
    row += 1

    print "Circular Block features: row #" + str(row)

# Sort blocks lexicographically based on their feature lists
blocks.sort(key = lambda x: x[1])

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
            [current_block.center_row, current_block.center_col], 
            [similar_block.center_row, similar_block.center_col] )

        # Block centers are too close, this block is to be ignored
        if center_distance < 2 * config.block_radius:
            # If there are still blocks after the ones in similar_block_list, 
            # append the next one
            if len(blocks) > pos + 1 + config.distance_threshold + extra:
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