import matplotlib.pyplot as plt
import numpy as np
import config
from copy import deepcopy
from lbp import rotation_invariant_uniform_lbp
from scipy import misc
from scipy.ndimage.filters import gaussian_filter

class Image:
    def __init__(self, path = ''):
        self._img = np.array(list(), dtype=np.uint8)
        self.height = 0
        self.width = 0
        self._initialized = False
        self._gray=False
        
        if len(path) > 0:
            self.load(path)


    def get_pixel(self, row, col):
        if (row < 0 or row >= self.height or col < 0 or col >= self.width):
            if self._gray:
                return np.uint8(0)
            else:
                return np.array([0, 0, 0], dtype=np.uint8)

        return self._img[row][col]


    def set_pixel(self, row, col, red, green, blue):
        if (row < 0 or row > self.height or col < 0 or col > self.width):
            raise Exception("Image out of bounds")

        self._img[row][col][0] = red
        self._img[row][col][1] = green
        self._img[row][col][2] = blue


    def get_lbp(self, row, col):
        if (row < 0 or row > self.height or col < 0 or col > self.width):
            raise Exception("Image out of bounds")

        if self._lbp_img[row][col] == -1:
            self._lbp_img[row][col] = rotation_invariant_uniform_lbp(self, 
                config.P, config.R, row, col)

        return self._lbp_img[row][col]


    def valid(self):
        return height > 0 and width > 0


    def load(self, path):
        self._img = misc.imread(path)
        self.height = len(self._img)
        self.width = len(self._img[0])
        self._initialized = True

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
            truncate=truncate_val, output=result._img)
        times -= 1

        # Apply the filter the remaining number of times. The first time is done
        # separately for memory optimization
        while times > 0:
            result._img = gaussian_filter(image._img, 
                sigma=config.gaussian_sigma, truncate=truncate_val)
            times -= 1


        return result