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
def lbp(image, P, R, i, j):
    # Color of the center pixel
    gc_color = image[i][j]

    # Accumulator for the LBP value
    lbp = 0

    for p in range(0, P):
        gp_j = -R * sin(2*pi*p/P)
        gp_i = R * cos(2*pi*p/P)

        # Estimate the color of gp
        gp_color = bilinear_interpolation(image, gp_i+i, gp_j+j)

        increment = lbp_s(gp_color - gc_color) << p
        lbp |= increment

    return lbp


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


def rotation_invariant_uniform_lbp(image, P, R, i, j):
    lbp_sequence = lbp(image, P, R, i, j)

    if u_value(lbp_sequence) > 2:
        return P+1

    result = 0

    for p in range(0, P):
        result += get_bit(lbp_sequence, p)

    return result


def bilinear_interpolation(image, i, j):
    # Tolerable difference
    epsilon = 1e-06

    # If the distance from position (i, j) to position ( round(i), round(j) ) 
    # is small, consider (i, j) both pixel positions to be equal
    if ( abs(i - round(i)) <= epsilon and abs(j - round(j)) <= epsilon ):
        return image[int(round(i))][int(round(j))]

    # Pixels whose values will be used in the interpolation:
    # [p11   p12]
    # [p21   p22]
    p11 = image[int(floor(i))][int(floor(j))]
    p12 = image[int(floor(i))][int(ceil(j))]
    p21 = image[int(ceil(i))][int(floor(j))]
    p22 = image[int(ceil(i))][int(ceil(j))]

    # Interpolate in the j-direction. The resulting values are
    # [q1]
    # [q2]
    q1 = (ceil(j)-j)*p11 + (j-floor(j))*p12
    q2 = (ceil(j)-j)*p21 + (j-floor(j))*p22

    # Interpolate in the i-direction. The resulting value is 
    # the interpolated result
    return (ceil(i)-i)*q1 + (i-floor(i))*q2

# Test case
image = [ [7, 1, 12], [2, 5, 5], [5, 3, 0] ]
P = 8
R = 1
lbp_sequence = lbp(image, P, R, 1, 1)

print( lbp_sequence )
print( "U value: " + str(u_value(lbp_sequence)) )
print( "Rotation invariant uniform LBP: " + 
    str(rotation_invariant_uniform_lbp(image, P, R, 1, 1)) )