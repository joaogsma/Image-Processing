from image import Image
from circular_block import Circular_Block
import config

# Read the image file
colored_image = Image( raw_input("Type the input file name: ") )
image = Image.grayscale(colored_image)


# TODO: gaussian filtering


blocks = list()

print image.height
print image.width

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
    print (row, col)

# Sort blocks lexicographically based on their feature lists
blocks.sort(key = lambda x: x[1])

for i in blocks:
    print i