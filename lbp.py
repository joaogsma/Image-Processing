from math import sin, cos, pi, floor, ceil

# s function used in the LBP
def lbp_s(value):
    if (value >= 0):
        return 1
    return 0


def get_bit(value, i):
    return ((1 << i) & value) >> i


# Computes the LBP value in the given image, centered at the point (i, j).
# P and r are the number of points in the radius and the value of the radius,
# respectively.
def lbp(image, P, R, r, c):
    # Color of the center pixel
    gc_color = image[r][c]

    # Accumulator for the LBP value
    lbp = 0

    for p in range(0, P):
        gp_c = -R * sin(2*pi*p/P)
        gp_r = R * cos(2*pi*p/P)

        # Estimate the color of gp
        gp_color = bilinear_interpolation(image, gp_r+r, gp_c+c)

        increment = lbp_s(gp_color - gc_color) << p
        lbp |= increment

    return lbp


# Computes the U value of a given LBP value (binary sequence)
def u_value(lbp_sequence):
    # Accumulator for the U value
    u_val = 0

    # Compute the number of transitions from the first bit 
    # to the last
    for p in range(1, P):
        current = get_bit(lbp_sequence, p)
        previous = get_bit(lbp_sequence, p-1)

        u_val += abs(current - previous)

    # Check for a possible transition between the last bit 
    # and the first
    last = get_bit(lbp_sequence, P-1)
    first = get_bit(lbp_sequence, 0)
    u_val += abs(last - first)

    return u_val


def rotation_invariant_uniform_lbp(image, P, R, r, c):
    lbp_sequence = lbp(image, P, R, r, c)

    if u_value(lbp_sequence) > 2:
        return P+1

    result = 0

    for p in range(0, P):
        result += get_bit(lbp_sequence, p)

    return result


def bilinear_interpolation(image, r, c):
    # Tolerable difference
    epsilon = 1e-06

    # If the distance from position (r, c) to position ( round(r), round(c) ) 
    # is small, consider (r, c) both pixel positions to be equal
    if ( abs(r - round(r)) <= epsilon and abs(c - round(c)) <= epsilon ):
        return image[ int(round(r)) ][ int(round(c)) ]

    # Pixels whose values will be used in the interpolation:
    # [p11   p12]
    # [p21   p22]
    p11 = image[int(floor(r))][int(floor(c))]
    p12 = image[int(floor(r))][int(ceil(c))]
    p21 = image[int(ceil(r))][int(floor(c))]
    p22 = image[int(ceil(r))][int(ceil(c))]

    # Interpolate in the j-direction. The resulting values are
    # [q1]
    # [q2]
    q1 = (ceil(c)-c)*p11 + (c-floor(c))*p12
    q2 = (ceil(c)-c)*p21 + (c-floor(c))*p22

    # Interpolate in the i-direction. The resulting value is 
    # the interpolated result
    return (ceil(r)-r)*q1 + (r-floor(r))*q2

# Test case
image = [ [7, 1, 12], [2, 5, 5], [5, 3, 0] ]
P = 8
R = 1
lbp_sequence = lbp(image, P, R, 1, 1)

print( lbp_sequence )
print( "U value: " + str(u_value(lbp_sequence)) )
print( "Rotation invariant uniform LBP: " + 
    str(rotation_invariant_uniform_lbp(image, P, R, 1, 1)) )