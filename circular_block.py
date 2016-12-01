import math
import numpy as np
from lbp import rotation_invariant_uniform_lbp
import config

class Circular_Block:
    def __init__(self, image, row, col):
        self._img = image
        self.center_row = row
        self.center_col = col

    # Check if this is a valid block (its entire area is contained inside 
    # the image
    def valid(self):
        return (config.block_radius <= self.center_row and 
            self.center_row <= self._img.height - config.block_radius and
            config.block_radius <= self.center_col and 
            self.center_col <= self._img.width - config.block_radius)


    def features(self, do_sort=True):
        if not self.valid():
            raise Exception("Block not entirely inside image")

        #print "begin center feature"
        center_feature = self._features_in_direction(self.center_row, 
            self.center_col, 0, 0)
        #print "end center feature"
        #print center_feature
        #print

        #print "begin north feature"
        north_features = self._features_in_direction(self.center_row - 1, 
            self.center_col, -1, 0)
        #print "end north feature"
        #print north_features
        #print

        #print "begin south feature"
        south_features = self._features_in_direction(self.center_row + 1, 
            self.center_col, 1, 0)
        #print "end south feature"
        #print south_features
        #print

        #print "begin east feature"
        east_features = self._features_in_direction(self.center_row, 
            self.center_col + 1, 0, 1)
        #print "end east feature"
        #print east_features
        #print
        
        #print "begin west feature"
        west_features = self._features_in_direction(self.center_row, 
            self.center_col - 1, 0, -1)
        #print "end west feature"
        #print west_features
        #print

        #print "begin northeast feature"
        northeast_features = self._features_in_direction(self.center_row - 1, 
            self.center_col + 1, -1, 1)
        #print "end northeast feature"
        #print northeast_features
        #print

        #print "begin northwest feature"
        northwest_features = self._features_in_direction(self.center_row - 1, 
            self.center_col - 1, -1, -1)
        #print "end northwest feature"
        #print northwest_features
        #print
        
        #print "begin southeast feature"
        southeast_features = self._features_in_direction(self.center_row + 1, 
            self.center_col + 1, 1, 1)
        #print "end southeast feature"
        #print southeast_features
        #print
        
        #print "begin southwest feature"
        southwest_features = self._features_in_direction(self.center_row + 1, 
            self.center_col - 1, 1, -1)
        #print "end southwest feature"
        #print southwest_features
        #print

        feature_list = np.concatenate((
            center_feature, north_features,
            south_features, east_features ,
            west_features , northeast_features ,
            northwest_features , southeast_features ,
            southwest_features) , axis = 0)



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

        #print "Parameters:"
        #print "row increment: " + str(row_increment)
        #print "col increment: " + str(col_increment)

        # Compute the maximum row and column numbers
        max_row = center_row + row_increment * config.block_radius
        max_col = center_col + col_increment * config.block_radius

        #print "max rows: " + str(max_row)
        #print "max cols: " + str(max_col)


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

        distance = math.sqrt( math.pow(row - self.center_row, 2) + 
            math.pow(col - self.center_col, 2))

        return distance <= config.block_radius + epsilon
        
