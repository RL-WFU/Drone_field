#
# Simple image classification in remote sensing (ICRS) simulator
# This version uses simple color thresholding to simulate pixel level classification
#
# Wake Forest University
# Nov. 2019

import cv2
import numpy as np
# from PIL import Image
from matplotlib import pyplot as plt
import math

"""
This file uses rgb classification to change our image maps into classification maps, where each cell in
a 2d array represents the probability of an area of interest, according to the image thresholding. We had to use
this since we were working with full maps with a simulator, but the drone should only need a similar implementation of
the getDroneClassifiedImageAt() function, which returns a 5x5 (droneImgSize) array of the probabilities described above.
"""

class ICRSsimulator(object):
    imgName = ""  # The image filename

    def __init__(self, imageName):
        self.imgName = imageName
        self.classLabels = []  # List of the labels for each class
        self.binaryImgs = []  # List of binary images, one for each class
        self.interestValues = []  # List of interest values, one for each class
        self.mapSize = {'rows': 0, 'cols': 0}  # Size of the classification map
        self.droneImgSize = {'rows': 0, 'cols': 0}  # Size in elements in the map of a drone image
        self.img = None  # Image object
        self.ICRSmap = None  # Overall classification map
        self.navigationMap = None  # Map of positions where the drone can be

    # Load the image and return true if it succeeded
    def loadImage(self):
        self.img = cv2.imread(self.imgName)
        if self.img is not None:
            print("Loaded image of rows x cols = " + str(self.img.shape[0]) + " x " + str(self.img.shape[1]))
            return True
        else:
            return False

    def getSimulationImageSize(self):
        return self.img.shape

    # Simulate classification of a specific class using simple image thresholding
    # The binary image contains the binary classification of each pixel as belonging to this class
    # Also save the corresponding interest value for this
    def classify(self, label, lower, upper, interestValue):
        # cv2.inRange returns 0 (black or 255 (white)
        binmap = cv2.inRange(self.img, lower, upper)
        self.binaryImgs.append(binmap)
        self.interestValues.append(interestValue)
        self.classLabels.append(label)

    # Return the number of classes in the list
    def numberOfClasses(self):
        return len(self.binaryImgs)

    # Set the size of the overall classification map to create
    # Parameter rows and cols must be less than the image size
    def setMapSize(self, rows, cols):
        self.mapSize['rows'] = rows
        self.mapSize['cols'] = cols

    # Set the size of image the drone can capture in number of
    # elements in the classification map (e.g. 4 x 4)
    def setDroneImgSize(self, rows, cols):
        self.droneImgSize['rows'] = rows
        self.droneImgSize['cols'] = cols

    # Define the navigation map, that is where the drone can move to explore
    # the simulation image. The navigation map is defined by the number of elements
    # in the classification map and the number of elements of the drone image
    def setNavigationMap(self):
        rows = self.mapSize['rows']
        cols = self.mapSize['cols']
        dImgrows = self.droneImgSize['rows']
        dImgcols = self.droneImgSize['cols']
        self.navigationMap = np.zeros((rows - dImgrows, cols - dImgcols))
        return self.navigationMap.shape

    # Create the overall classification map
    def createMap(self):
        rows = self.mapSize['rows']
        cols = self.mapSize['cols']
        # DEBUG: print("rows, cols = " + str(rows) + ", " + str(cols))
        # number of pixels per box in the map
        pixelsPerRow = self.img.shape[0] // rows  # Note: // means integer division
        pixelsPerCol = self.img.shape[1] // cols
        # DEBUG: print("ppr, ppc = " + str(pixelsPerRow) + ", " + str(pixelsPerCol))
        # The overall map will be stored in a 3D numpy array of dim: rows x cols x num of classes
        self.ICRSmap = np.zeros((rows, cols, len(self.binaryImgs)))

        # Simulate classification of object class k for each box of the ICRSmap
        for k in range(0, len(self.binaryImgs)):  # for each class type
            # number of remaining pixels per row and column left over if rows and cols
            # don't divide evenly the image height and width, respectively
            rem_pixelsPerRow = self.img.shape[0] % rows
            rem_pixelsPerCol = self.img.shape[1] % cols
            # DEBUG: print("rem_ppr, rem_ppc = " + str(rem_pixelsPerRow) + ", " + str(rem_pixelsPerCol))

            binImg = self.binaryImgs[k]
            startingRow = 0
            for i in range(0, rows):  # rows go with height
                # increase pixelsPerRow by 1 if there are still remaining row pixels
                actualPixelsPerRow = pixelsPerRow
                if rem_pixelsPerRow > 0:
                    actualPixelsPerRow = actualPixelsPerRow + 1
                    rem_pixelsPerRow = rem_pixelsPerRow - 1

                startingCol = 0
                rem_pixelsPerCol = self.img.shape[1] % cols
                for j in range(0, cols):  # cols go with width
                    # increase pixelsPerCol by 1 if there are still remaining col pixels
                    actualPixelsPerCol = pixelsPerCol
                    if rem_pixelsPerCol > 0:
                        actualPixelsPerCol = actualPixelsPerCol + 1
                        rem_pixelsPerCol = rem_pixelsPerCol - 1
                    actualNumPixelsPerBox = actualPixelsPerCol * actualPixelsPerRow
                    # extract a box and compute likelihood of finding object k in it
                    box = binImg[startingRow: startingRow + actualPixelsPerRow,
                          startingCol: startingCol + actualPixelsPerCol]
                    # box = binImg[ i * actualPixelsPerRow : (i + 1) * actualPixelsPerRow,\
                    #	j * actualPixelsPerCol : (j + 1) * actualPixelsPerCol ]

                    box = box.reshape(actualNumPixelsPerBox, 1)
                    white = sum(sum(box == 255))

                    self.ICRSmap[i, j, k] = white / (actualNumPixelsPerBox)
                    startingCol = startingCol + actualPixelsPerCol

                # DEBUG: print("i, j, actual_ppr, rem_ppr, actual_ppc, rem_ppc = " + str(i) + ", " + str(j) + ", "
                # DEBUG: + str(actualPixelsPerRow) + ", " + str(rem_pixelsPerRow) + ", " + str(actualPixelsPerCol) + ","
                # DEBUG: + str(rem_pixelsPerCol))

                startingRow = startingRow + actualPixelsPerRow

    # Get the classification likelihoods in row, col element of the ICRSmap
    def getMapElement(self, row, col):
        return self.ICRSmap[row, col, 0]

    # Get the classification likelihoods in a neighborhood of the ICRSmap
    def getClassifiedDroneImageAt(self, row, col):
        """
        This function should be re-written for implementation in the field.
        Instead of retrieving the 5x5 map from the ICRS map, it should retrieve it from the
        image taken by the drone's camera
        :param row: drone's current row position
        :param col: drone's current col position
        :return: a 5x5 map of mining probabilities
        """
        self.navigationMap[row, col] = self.navigationMap[row, col] + 1
        return self.ICRSmap[row - self.droneImgSize['rows']: row + self.droneImgSize['rows'] + 1,
               col - self.droneImgSize['cols']: col + self.droneImgSize['cols'] + 1, 0]

    # Plot the original image as well as the binary classification images
    def showMap(self):
        fig, axs = plt.subplots(1, len(self.binaryImgs) + 1)
        fig.suptitle('Classification maps for each object')
        # First show the original image and original binary maps
        # Note that cv2 uess BGR format which needs to be converted to RGB for Matplotlib
        b, g, r = cv2.split(self.img)
        rgb_img = cv2.merge([r, g, b])
        axs[0].imshow(rgb_img, interpolation='none')
        axs[0].set_title('Original Image')
        # DEBUG: Show original binary maps in first column
        # for k in range(0, len(self.binaryImgs)):
        #	axs[0, k+1].imshow(self.binaryImgs[k], cmap = 'gray', interpolation = 'none')
        #	axs[0, k+1].set_title(self.classLabels[k])
        # Now show the binary classification maps for the given map size
        for k in range(0, len(self.binaryImgs)):
            axs[k + 1].imshow(self.ICRSmap[:, :, k], cmap='gray', interpolation='none')
            axs[k + 1].set_title(self.classLabels[k])
        plt.show()
        plt.clf()

        plt.imshow(self.ICRSmap[:, :, 0], cmap='gray', interpolation='none')
        plt.savefig('mining.jpg')



