from scipy import misc
from copy import copy
import matplotlib.pyplot as plt
import numpy as np


class Image:
    def __init__(self, path = ''):
        self._img = np.array(list())
        self._height = 0
        self._width = 0
        self._initialized = False
        self._gray=False

        if len(path) > 0:
            self.load(path)

    def get_pixel(self, r, c):
        return self._img[r][c]

    def set_pixel(self, r, c, red, green, blue):
        self._img[r][c][0] = red
        self._img[r][c][1] = green
        self._img[r][c][2] = blue

    def height(self):
        return self._height;

    def width(self):
        return self._width;

    def valid(self):
        return _height > 0 and _width > 0

    def load(self, path):
        self._img = misc.imread(path)
        self._height = len(self._img)
        self._width = len(self._img[0])
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
        result._height = self._height
        result._width = self._width
        result._initialized = True
        result._img = np.zeros(shape=(self._height, self._width))
        result._gray = True

        for r in range(0, self._height):
            for c in range(0, self._width):
                result._img[r][c] = np.uint8( round(0.299*self._img[r][c][0] + 
                    0.587*self._img[r][c][1] + 0.114*self._img[r][c][2]) )


        return result


