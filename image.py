from scipy import misc
from copy import copy
import matplotlib.pyplot as plt
import numpy as np


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
        if (row < 0 or row > self.height or col < 0 or col > self.width):
            return np.array([0, 0, 0], dtype=np.uint8)

        return self._img[row][col]


    def set_pixel(self, row, col, red, green, blue):
        if (row < 0 or row > self.height or col < 0 or col > self.width):
            raise Exception("Image out of bounds")

        self._img[row][col][0] = red
        self._img[row][col][1] = green
        self._img[row][col][2] = blue


    def valid(self):
        return height > 0 and width > 0


    def load(self, path):
        self._img = misc.imread(path)
        self.height = len(self._img)
        self.width = len(self._img[0])
        self._initialized = True


    def save(self, path):
        misc.imsave(path, self._img)


    def show(self):
        if self._gray:
            plt.imshow(self._img, cmap=plt.cm.gray)
        else:
            plt.imshow(self._img, cmap=plt.cm.gray)

        plt.show()


    def grayscale(self):
        result = Image()
        result.height = self.height
        result.width = self.width
        result._initialized = True
        result._img = np.zeros(shape=(self.height, self.width))
        result._gray = True

        for row in range(0, self.height):
            for col in range(0, self.width):
                result._img[row][col] = np.uint8( round(0.299*self._img[row][col][0] + 
                    0.587*self._img[row][col][1] + 0.114*self._img[row][col][2]) )


        return result


