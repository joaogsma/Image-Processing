import matplotlib.pyplot as plt
import numpy as np
import config
import cv2
from copy import deepcopy
from lbp import rotation_invariant_uniform_lbp, rotation_invariant_center_symmetric_lbp
from math import ceil
from multiprocessing import Process, Queue
from scipy import misc
from scipy.ndimage.filters import gaussian_filter

def fill_lbp_line(image, min_row, max_row, result_queue):
    row = min_row

    result = np.zeros(shape=(max_row-min_row, image.width), dtype=int)

    while row < max_row:
        col = 0

        while col < image.width:
            if image._lbp_img[row][col] == -1:
                if config.lbp_type == 'lbp':
                    result[row - min_row][col] = rotation_invariant_uniform_lbp( 
                        image, config.P, config.R, row, col )
                elif config.lbp_type == 'cslbp':
                    result[row - min_row][col] = rotation_invariant_center_symmetric_lbp( 
                        image, config.P, config.R, row, col )
                else:
                    raise Exception('Invalid LBP type')
            col += 1

        row += 1

    result_queue.put( (min_row, max_row, result) )
                

def black_image(num_rows, num_cols):
    result = Image()

    result._img = np.zeros(shape=(num_rows, num_cols), dtype=np.uint8)
    result.height = num_rows
    result.width = num_cols
    result._gray = True

    result._lbp_img = np.full((num_rows, num_cols), -1, dtype=int)

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
            if config.lbp_type == 'lbp':
                self._lbp_img[row][col] = rotation_invariant_uniform_lbp(self, 
                    config.P, config.R, row, col)
            elif config.lbp_type == 'cslbp':
                self._lbp_img[row][col] = rotation_invariant_center_symmetric_lbp(
                    self, config.P, config.R, row, col)
            else:
                raise Exception('Invalid LBP type')

        return self._lbp_img[row][col]


    def fill_lbp(self):
        row = 0

        queue = Queue()

        increment = int( ceil(self.height / float(config.num_threads)) )

        while row < self.height:
            new_process = Process( target=fill_lbp_line, 
                args=(self, row, min(row+increment, self.height), queue)  )
            new_process.start()
            row += increment
        
        for i in range(config.num_threads):
            (min_row, max_row, contents) = queue.get()
            
            row = min_row
            while row < max_row:
                col = 0
                while col < self.width:
                    self._lbp_img[row][col] = contents[row - min_row][col]
                    col += 1
                row += 1


    def valid(self):
        return height > 0 and width > 0


    def load(self, path):
        self._img = misc.imread(path)
        self.height = len(self._img)
        self.width = len(self._img[0])

        self._lbp_img = np.full((self.height, self.width), -1, dtype=int)

    def save(self, path):
        misc.imsave(path, self._img)


    def show(self):
        if self._gray:
            plt.imshow(self._img, cmap=plt.cm.gray)
        else:
            plt.imshow(self._img)

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

        result._lbp_img = np.full((result.height, result.width), -1, dtype=int)

        return result


    def opening(image, kernel, in_place=False):
        if not image._gray:
            raise Exception("Must be a binary image")

        result = cv2.morphologyEx(image._img, cv2.MORPH_OPEN, 
            kernel.astype(np.uint8))

        target = None
        if in_place:
            target = image
        else:
            target = deepcopy(image)
        
        target._img = result
        target._lbp_img = np.full((target.height, target.width), -1, dtype=int)

        if not in_place:
            return target


    def gaussian_filter(image, is_grayscale=False, times=1):
        gray_image = image
        
        if not is_grayscale:
            # Obtain grayscale image from parameter
            gray_image = Image.grayscale(image)
        
        # Create the result to be smoothed
        result = deepcopy(gray_image)

        # Ensures the filter is 5x5
        truncate_val = 2/config.gaussian_sigma
        
        if (times > 0):
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


    def mean_filter(image, kernel_size=3):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise Exception('Invalid kernel size')
        
        result = Image.grayscale(image)
        
        row = 0
        while row < image.height:
            col = 0
            while col < image.width:
                summation = 0.0
                increment = (kernel_size - 1) / 2
                
                # Compute the sum
                row_inc = -increment
                while row_inc <= increment:
                    col_inc = -increment
                    while col_inc <= increment:
                        summation += image.get_pixel( row + row_inc, col + col_inc )
                        col_inc += 1
                    row_inc += 1
                
                mean = summation / (kernel_size**2)

                result._img[row][col] = mean

                col += 1
            row += 1

        return result


    def custom_filter(image):
        window_pos_row = 0

        while window_pos_row < image.height:
            window_pos_col = 0

            while window_pos_col < image.width:
                # Max row and col for this window
                max_row = min(window_pos_row+config.custom_filter_height, image.height)
                max_col = min(window_pos_col+config.custom_filter_width, image.width)
                
                # Counter for white pixels
                white_pixels = 0

                # ===== Count the number of black pixels in the window =====
                row = window_pos_row
                while (row < max_row and 
                        white_pixels < config.custom_filter_threshold):
                    col = window_pos_col
                    
                    while (col < max_col and 
                            white_pixels < config.custom_filter_threshold):
                        if image._img[row][col] == 255:
                            white_pixels += 1
                        col += 1
                    
                    row += 1
                # ==========================================================

                # If there are not enough white pixels, set all of them to black
                if white_pixels < config.custom_filter_threshold:
                    row = window_pos_row
                    while row < max_row:
                        col = window_pos_col
                        while col < max_col:
                            image._img[row][col] = 0
                            col += 1
                        row += 1

                window_pos_col += config.custom_filter_width

            window_pos_row += config.custom_filter_height
