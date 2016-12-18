num_threads = 8

# Radius of circular blocks
block_radius = 9 #9

# Distance between circular blocks
blocks_spacing = 2


# ========== LBP ==========

# LBP type, 'lbp' or 'cslbp'
lbp_type = 'lbp'

# LBP number of points
P = 24

# LBP radius
R = 3

# =========================


# ========== LOW-PASS FILTER ==========

# Type of filter, 'gaussian', 'mean' or 'none'
low_pass_filter_type = 'mean'

# Kernel size for the mean filter. Must be an odd number
mean_kernel_size = 5

# Sigma of gaussian filter
gaussian_sigma = 2

# Number of times the gaussian filter should be applied
gaussian_times = 2

# =====================================


# ========== MATCHING ==========

# True if the histograms should be compressed to P+1 values. 
# False if they should be extended
compress_features = True

# Matching type, 'lex', 'kd-tree' or 'k-mean'
matching_type = 'k-mean'

# Maximum matching distance searched per block
distance_threshold = 30

# Maximum distance for blocks to be considered a match
# Reasonable values:
#   compress_features = False => 6.2 
#   compress_features = True => 11 - 13
similarity_threshold = 7

# Set True if the verification for matches in the circular blocks should go 
# beyond the distance_threshold to compensate for blocks that are too close 
# (and therefore ignored)
expanded_matching = False

# ==============================


# Specifications of the custom filter used to reduce false positive matches
custom_filter_height = 8
custom_filter_width = 8

# Reasonable values:
#   blocks_spacing = 0 => 15
#   blocks_spacing = 1 => 25
#   blocks_spacing = 2 => 39
#   blocks_spacing = 3 => 45
custom_filter_threshold = 39

# Radius of a matched area in the resulting mask. If True, the default_value is 
# 1. If False, the value will be equivalent to the spacing between blocks + 1
default_match_radius = False