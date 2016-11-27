import math
import numpy as np
from lbp import rotation_invariant_uniform_lbp
import config

class Circular_Block:
    def __init__(self, image, row, col):
        self._img = image
        self._center_row = row
        self._center_col = col

    # Check if this is a valid block (its entire area is contained inside 
    # the image
    def valid(self):
        return (config.block_radius <= self._center_row and 
            self._center_row <= self._img.height - config.block_radius and
            config.block_radius <= self._center_col and 
            self._center_col <= self._img.width - config.block_radius)


    def features(self, do_sort=False):
        if not self.valid():
            raise Exception("Block not entirely inside image")

        feature_list = list()

        #print "begin center feature"
        center_feature = self._features_in_direction(self._center_row, 
            self._center_col, 0, 0)
        #print "end center feature"
        #print center_feature
        #print

        #print "begin north feature"
        north_features = self._features_in_direction(self._center_row - 1, 
            self._center_col, -1, 0)
        #print "end north feature"
        #print north_features
        #print

        #print "begin south feature"
        south_features = self._features_in_direction(self._center_row + 1, 
            self._center_col, 1, 0)
        #print "end south feature"
        #print south_features
        #print

        #print "begin east feature"
        east_features = self._features_in_direction(self._center_row, 
            self._center_col + 1, 0, 1)
        #print "end east feature"
        #print east_features
        #print
        
        #print "begin west feature"
        west_features = self._features_in_direction(self._center_row, 
            self._center_col - 1, 0, -1)
        #print "end west feature"
        #print west_features
        #print

        #print "begin northeast feature"
        northeast_features = self._features_in_direction(self._center_row - 1, 
            self._center_col + 1, -1, 1)
        #print "end northeast feature"
        #print northeast_features
        #print

        #print "begin northwest feature"
        northwest_features = self._features_in_direction(self._center_row - 1, 
            self._center_col - 1, -1, -1)
        #print "end northwest feature"
        #print northwest_features
        #print
        
        #print "begin southeast feature"
        southeast_features = self._features_in_direction(self._center_row + 1, 
            self._center_col + 1, 1, 1)
        #print "end southeast feature"
        #print southeast_features
        #print
        
        #print "begin southwest feature"
        southwest_features = self._features_in_direction(self._center_row + 1, 
            self._center_col - 1, 1, -1)
        #print "end southwest feature"
        #print southwest_features
        #print

        feature_list.extend(center_feature)
        feature_list.extend(north_features)
        feature_list.extend(south_features)
        feature_list.extend(east_features)
        feature_list.extend(west_features)
        feature_list.extend(northeast_features)
        feature_list.extend(northwest_features)
        feature_list.extend(southeast_features)
        feature_list.extend(southwest_features)

        if do_sort:
            feature_list.sort()

        return feature_list
    

    def _sign(self, val):
        if val == 0:
            return 1

        return np.sign(val)


    def _features_in_direction(self, center_row, center_col, row_increment, 
            col_increment):
        feature_list = list()

        # Compute the maximum row and column numbers
        max_row = center_row + row_increment * config.block_radius
        max_col = center_col + col_increment * config.block_radius

        #print "max rows: " + str(max_row)
        #print "max cols: " + str(max_col)

        #print "Parameters:"
        #print "row increment: " + str(row_increment)
        #print "col increment: " + str(col_increment)

        # Modify the row and column increments to have absolute value of at least 
        # 1. This is done to avoid infinite loops when either is 0, and does not
        # change their values otherwise, since they have discrete values
        #col_increment = self._sign(col_increment) * max(abs(col_increment), 1)
        #row_increment = self._sign(row_increment) * max(abs(row_increment), 1)

        #print "Modified:"
        #print "row increment: " + str(row_increment)
        #print "col increment: " + str(col_increment)

        row = center_row

        while True:
            col = center_col

            while True:
                #print (row, col)

                if not self._inside_circle(row, col):
                    break

                #print "inside circle"

                # Get the pixel's LBP position and add it to the feature list
                lbp_sequence = self._img.get_lbp(row, col)
                feature_list.append( lbp_sequence )
                
                col += col_increment
                if col == max_col:
                    break

            row += row_increment
            if row == max_row:
                break

        return feature_list


    def _inside_circle(self, row, col):
        epsilon = 1e-06

        distance = math.sqrt( math.pow(row - self._center_row, 2) + 
            math.pow(col - self._center_col, 2))

        if distance <= config.block_radius + epsilon:
            return True

        return False