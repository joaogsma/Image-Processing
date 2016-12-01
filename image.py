import matplotlib.pyplot as plt
import numpy as np
import config
from copy import deepcopy
from lbp import rotation_invariant_uniform_lbp
from math import ceil
from multiprocessing import Process
from scipy import misc
from scipy.ndimage.filters import gaussian_filter

def fill_lbp_line(image, min_row, max_row):
    row = min_row

    while row < max_row:
        col = 0
        
        while col < image.width:
            if image._lbp_img[row][col] == -1:
                image._lbp_img[row][col] = rotation_invariant_uniform_lbp( 
                    image, config.P, config.R, row, col )
            col += 1

        row += 1


def black_image(num_rows, num_cols):
    result = Image()

    result._img = np.zeros(shape=(num_rows, num_cols), dtype=np.uint8)
    result.height = num_rows
    result.width = num_cols
    result._gray = True

    result._lbp_img = np.zeros(shape=(num_rows, num_cols), dtype=int)

    row = 0
    while row < num_rows:
        col = 0
        while col < num_cols:
            result._lbp_img[row][col] = -1
            col += 1
        row += 1

    return result


class Image:
    def __init__(self, path=''):
        self._img = np.array(list(), dtype=np.uint8)
        self.height = 0
        self.width = 0
        self._gray=False
        
        if len(path) > 0:
            self.load(path)


    def get_pixel(self, row, col):
        # Pick the replicated equivalent row, if the specified one is out of bounds
        if row < 0:
            row = config.P - row
        elif row >= self.height:
            row = row - config.P
        
        # Pick the replicated equivalent column, if the specified one is out of bounds
        if col < 0:
            col = config.P - col
        elif col >= self.width:
            col = col - config.P

        return self._img[row][col]


    def set_pixel_rgb(self, row, col, red, green, blue):
        if (row < 0 or row > self.height or col < 0 or col > self.width):
            raise Exception("Image out of bounds")

        self._img[row][col][0] = np.uint8(red)
        self._img[row][col][1] = np.uint8(green)
        self._img[row][col][2] = np.uint8(blue)


    def set_pixel_gray(self, row, col, color):
        if (row < 0 or row > self.height or col < 0 or col > self.width):
            raise Exception("Image out of bounds")

        self._img[row][col] = np.uint8(color)


    def get_lbp(self, row, col):
        if (row < 0 or row > self.height or col < 0 or col > self.width):
            raise Exception("Image out of bounds")

        if self._lbp_img[row][col] == -1:
            self._lbp_img[row][col] = rotation_invariant_uniform_lbp(self, 
                config.P, config.R, row, col)

        return self._lbp_img[row][col]


    def fill_lbp(self):
        row = 0

        processes = list()

        increment = int( ceil(self.height / float(config.num_threads)) )

        print "increment: " + str(increment)

        while row < self.height:
            new_process = Process( target=fill_lbp_line, 
                args=(self, row, min(row+increment, self.height))  )
            processes.append( new_process )
            new_process.start()

            print "row: " + str(row) + "   min: " + str(row) + "   max: " + str(min(row+increment, self.height))

            row += increment
        
        for p in processes:
            p.join()



    def valid(self):
        return height > 0 and width > 0


    def load(self, path):
        self._img = misc.imread(path)
        self.height = len(self._img)
        self.width = len(self._img[0])

        self._lbp_img = np.zeros(shape=(self.height, self.width), dtype=int)

        row = 0
        while row < self.height:
            col = 0
            while col < self.width:
                self._lbp_img[row][col] = -1
                col += 1
            row += 1



    def save(self, path):
        misc.imsave(path, self._img)


    def show(self):
        if self._gray:
            plt.imshow(self._img, cmap=plt.cm.gray)
        else:
            plt.imshow(self._img, cmap=plt.cm.gray)

        plt.show()


    def grayscale(image):
        if image._gray:
            return deepcopy(image)
        
        result = Image()
        result.height = image.height
        result.width = image.width
        result._initialized = True
        result._img = np.zeros( shape=(image.height, image.width), dtype=np.uint8 )
        result._gray = True

        row = 0
        while row < image.height:
            col = 0

            while col < image.width:
                result._img[row][col] = np.uint8( 
                    round(0.299*image._img[row][col][0] + 
                        0.587*image._img[row][col][1] + 
                        0.114*image._img[row][col][2]) )
                col += 1
            row += 1

        result._lbp_img = np.zeros(shape=(result.height, result.width), dtype=int)

        row = 0
        while row < result.height:
            col = 0
            while col < result.width:
                result._lbp_img[row][col] = -1
                col += 1
            row += 1

        return result


    def gaussian_filter(image, is_grayscale=False, times=1):
        gray_image = image
        
        if not is_grayscale:
            # Obtain grayscale image from parameter
            gray_image = Image.grayscale(image)
        
        # Create the result to be smoothed
        result = deepcopy(gray_image)

        # Ensures the filter is 5x5
        truncate_val = 2/config.gaussian_sigma
        # Apply gaussian filter to grayscale image
        gaussian_filter(gray_image._img, sigma=config.gaussian_sigma, 
            truncate=truncate_val, output=result._img, mode='reflect')
        times -= 1

        # Apply the filter the remaining number of times. The first time is done
        # separately for memory optimization
        while times > 0:
            result._img = gaussian_filter(image._img, 
                sigma=config.gaussian_sigma, truncate=truncate_val)
            times -= 1


        return result


    def custom_filter(image):
        window_pos_row = 0

        while window_pos_row < image.height:
            window_pos_col = 0

            while window_pos_col < image.width:
                print (window_pos_row, window_pos_col)

                # Max row and col for this window
                max_row = min(window_pos_col+config.custom_filter_height, image.height)
                max_col = min(window_pos_col+config.custom_filter_width, image.width)
                
                # Counter for black pixels
                black_pixels = 0

                # ===== Count the number of black pixels in the window =====
                row = window_pos_col
                while row < max_row:
                    col = window_pos_col
                    
                    while col < max_col:
                        if image._img[row][col] == 0:
                            black_pixels += 1
                    
                            if black_pixels == config.custom_filter_threshold:
                                break

                        col += 1
                    
                    if black_pixels == config.custom_filter_threshold:
                        break
                        
                    row += 1
                # ==========================================================

                # If there are enough black pixels, set all of them to black
                if black_pixels == config.custom_filter_threshold:
                    row = window_pos_col
                    while row < max_row:
                        col = window_pos_col
                        while col < max_col:
                            image._img[row][col] = 0
                            col += 1
                        row += 1

                window_pos_col += config.custom_filter_width

            window_pos_row += config.custom_filter_height
