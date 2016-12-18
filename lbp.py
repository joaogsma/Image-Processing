from math import sin, cos, pi, floor, ceil

# s function used in the LBP
def lbp_s(value, t = 0):
    if (value >= t):
        return 1
    return 0


def get_bit(value, i):
    return ((1 << i) & value) >> i


# Computes the LBP value in the given image, centered at the point (row, col).
# P and R are the number of points in the radius and the value of the radius,
# respectively.
def lbp(image, P, R, row, col):
    # Color of the center pixel
    gc_color = int(image.get_pixel(row, col))

    # Accumulator for the LBP value
    lbp = int(0)

    p = 0
    while p < P:
        gp_c = -R * sin(2*pi*p/P)
        gp_r = R * cos(2*pi*p/P)

        # Estimate the color of gp
        gp_color = int(bilinear_interpolation(image, gp_r + row, gp_c + col))

        increment = lbp_s(gp_color - gc_color) << p
        
        lbp |= increment
        p += 1

    return lbp


def center_symmetric_lbp(image, P, R, row, col):
    # Accumulator for the sequence
    lbp = int(0)

    colors = list()

    p = 0
    while p < P:
        # Increments in row and col coordinates
        gp_c = -R * sin(2*pi*p/P)
        gp_r = R * cos(2*pi*p/P)

        # Compute and store the interpolated color
        gp_color = int(bilinear_interpolation(image, gp_r + row, gp_c + col))
        colors.append(gp_color)
        
        p += 1

    p = 0
    while p < (P / 2) - 1:
        increment = lbp_s( colors[p] - colors[p + P/2], 2.5 ) << p
        lbp |= increment
        p += 1

    return lbp


# Computes the U value of a given LBP value (binary sequence)
def u_value(lbp_sequence, P):
    # Accumulator for the U value
    u_val = 0

    # Compute the number of transitions from the first bit 
    # to the last
    p = 1
    while p < P:
        current = get_bit(lbp_sequence, p)
        previous = get_bit(lbp_sequence, p-1)

        u_val += abs(current - previous)
        p += 1

    # Check for a possible transition between the last bit 
    # and the first
    last = get_bit(lbp_sequence, P-1)
    first = get_bit(lbp_sequence, 0)
    u_val += abs(last - first)

    return u_val


def rotation_invariant_uniform_lbp(image, P, R, row, col):
    lbp_sequence = lbp(image, P, R, row, col)

    if u_value(lbp_sequence, P) > 2:
        return P+1

    result = 0

    p = 0
    while p < P:
        result += get_bit(lbp_sequence, p)
        p +=1 

    return result


def circular_right_shift(value, sequence_length):
    return (((1 & value) << sequence_length) | value) >> 1


def rotation_invariant_center_symmetric_lbp(image, P, R, row, col):
    cs_lbp_sequence = center_symmetric_lbp(image, P, R, row, col)

    value = cs_lbp_sequence
    smallest = cs_lbp_sequence
    
    # Try all shifted values. Because the cslbp sequence has length P/2,
    # it opccupies P/2 bits and P/2 - 1 shifts are necessary to iterate 
    # through all shifted values
    i = 0
    while i < P/2:
        # Circularly right shift the value
        value = circular_right_shift(value, P/2)
        # Compare with smallest so far
        if value < smallest:
            smallest = value
        i += 1

    return smallest


def no_interpolation(image, row, col):
    return image.get_pixel( int(round(row)), int(round(col)) )


def bilinear_interpolation(image, row, col):
    # Tolerable difference
    epsilon = 1e-06

    # If the distance from position (row, col) to position 
    # ( round(row), round(col) ) is small, consider both pixel 
    # positions to be equal
    if ( abs(row - round(row)) <= epsilon and abs(col - round(col)) <= epsilon ):
        return image.get_pixel( int(round(row)), int(round(col)) )

    # Pixels whose values will be used in the interpolation:
    # [p11   p12]
    # [p21   p22]
    p11 = image.get_pixel( int(floor(row)), int(floor(col)) )
    p12 = image.get_pixel( int(floor(row)), int(ceil(col)) )
    p21 = image.get_pixel( int(ceil(row)), int(floor(col)) )
    p22 = image.get_pixel( int(ceil(row)), int(ceil(col)) )

    # Interpolate in the j-direction. The resulting values are
    # [q1]
    # [q2]
    q1 = (ceil(col)-col)*p11 + (col-floor(col))*p12
    q2 = (ceil(col)-col)*p21 + (col-floor(col))*p22

    # Interpolate in the i-direction. The resulting value is 
    # the interpolated result
    return (ceil(row)-row)*q1 + (row-floor(row))*q2

# At the end to avoid circular dependency
from image import Image
