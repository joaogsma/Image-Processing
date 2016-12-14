num_threads = 8

# Radius of circular blocks
block_radius = 9

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
mean_kernel_size = 3

# Sigma of gaussian filter
gaussian_sigma = 2

# Number of times the gaussian filter should be applied
gaussian_times = 2

# =====================================



# ========== THRESHOLDS ==========

# Maximum matching distance searched per block
distance_threshold = 30

# Maximum distance for blocks to be considered a match
similarity_threshold = 6.2

# ================================

# Specifications of the custom filter used to reduce false positive matches
custom_filter_height = 8
custom_filter_width = 8
custom_filter_threshold = 15

# Set True if the verification for matches in the circular blocks should go 
# beyond the distance_threshold to compensate for blocks that are too close 
# (and therefore ignored)
expanded_matching = False

