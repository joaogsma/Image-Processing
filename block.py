import math
from lbp import rotation_invariant_uniform_lbp

class Circular_Block:
    def __init__(self, image, row, col, radius):
        self._img = img
        self._center_row = row
        self._center_col = col
        self._radius = radius

    # Check if this is a valid block (its entire area is contained inside 
    # the image
    def valid(self):
        return (self._radius <= self._center_row and 
            self._center_row <= self._img.height - self._radius and
            self._radius <= self._center_col and 
            self._center_col <= self._img.width - self._radius)


    def features(self, P, R, do_sort=False):
        if not self.valid():
            raise Exception("Block not entirely inside image")

        feature_list = list()

        center_feature = self._features_in_direction(self._center_row, 
            self._center_col, 0, 0, P, R)
        
        north_features = self._features_in_direction(self._center_row - 1, 
            self._center_col, -1, 0, P, R)
        south_features = self._features_in_direction(self._center_row + 1, 
            self._center_col, 1, 0, P, R)
        east_features = self._features_in_direction(self._center_row, 
            self._center_col + 1, 0, 1, P, R)
        west_features = self._features_in_direction(self._center_row, 
            self._center_col - 1, 0, -1, P, R)

        northeast_features = self._features_in_direction(self._center_row - 1, 
            self._center_col + 1, -1, 1, P, R)
        northwest_features = self._features_in_direction(self._center_row - 1, 
            self._center_col - 1, -1, -1, P, R)
        southeast_features = self._features_in_direction(self._center_row + 1, 
            self._center_col + 1, 1, 1, P, R)
        southwest_features = self._features_in_direction(self._center_row + 1, 
            self._center_col - 1, 1, -1, P, R)

        feature_list.extend(center_feature)
        feature_list.extend(north_feature)
        feature_list.extend(south_feature)
        feature_list.extend(east_feature)
        feature_list.extend(west_feature)
        feature_list.extend(northeast_feature)
        feature_list.extend(northwest_feature)
        feature_list.extend(southeast_feature)
        feature_list.extend(southwest_feature)

        if do_sort:
            feature_list.sort()

        return feature_list
    

    def _features_in_direction(self, center_row, center_col, row_increment, 
            col_increment, P, R):
        feature_list = list()

        # Compute the maximum row and column numbers
        max_row = center_row + row_increment * self._radius
        max_col = center_col + col_increment * self._radius

        # Modify the row and column increments to have absolute value of at least 
        # 1. This is done to avoid infinite loops when either is 0, and does not
        # change their values otherwise, since they have discrete values
        col_increment = np.sign(col_increment) * max(abs(col_increment), 1)
        row_increment = np.sign(row_increment) * max(abs(row_increment), 1)

        row = center_row

        while row <= max_row:
            col = center_col

            while col <= max_col:
                if not self._inside_circle(row, col):
                    break

                # Compute the pixel's LBP position and add it to the feature list
                lbp_sequence = rotation_invariant_uniform_lbp(self._img, P, 
                    R, row, col)
                feature_list.append( lbp_sequence )
                
                col += col_increment

            row += row_increment

        return feature_list


    def _inside_circle(self, row, col):
        epsilon = 1e-06

        distance = math.sqrt( math.pow(row - self._center_row, 2) + 
            math.pow(col - self._center_col, 2))

        if abs(distance) <= epsilon:
            return True

        return False