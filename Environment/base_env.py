import random
import sys
from ICRSsimulator import *
import numpy as np
from configurationSimple import *
from copy import deepcopy

"""
This is the parent class for search, trace, and target environments. Certain functions here are preserved or overriden in certain classes
These environment classes work with indices to specify drone and target location. For field use, they may need to be changed to gps

"""

class Env:
    # Create simulator object and load the image
    config = ConfigSimple()
    sim = ICRSsimulator(config.image)
    if not sim.loadImage():
        print("Error: could not load image")
        sys.exit(0)

    # Initialize map of size totalRows x totalCols from the loaded image
    totalRows = config.total_rows
    totalCols = config.total_cols

    # Initialize tracking
    map = np.zeros([totalRows, totalCols])
    visited = np.ones([totalRows, totalCols])

    local_map = np.zeros([25, 25])
    local_map_lower_row = 0
    local_map_lower_col = 0
    local_map_upper_row = 24
    local_map_upper_col = 24

    # Define partitions
    """
    These region partitions are configured for a map of totalRows x totalCols with 9 targets
    """
    region_one = [0, 0]
    region_two = [0, totalCols//3]
    region_three = [0, (totalCols//3) * 2]
    region_four = [totalRows//3, 0]
    region_five = [totalRows//3, totalCols//3]
    region_six = [totalRows//3, (totalCols//3) * 2]
    region_seven = [(totalRows//3) * 2, 0]
    region_eight = [(totalRows//3) * 2, totalCols//3]
    region_nine = [(totalRows//3) * 2, (totalCols//3) * 2]
    regions = [region_one, region_two, region_three, region_six, region_five, region_four, region_seven,
               region_eight, region_nine]

    # Initialize targets
    """
    These targets are in the middle of their respective region. Configured for 9 regions
    """
    targets = [[region_one[0] + totalRows / 6, region_one[1] + totalCols / 6],
               [region_two[0] + totalRows / 6, region_two[1] + totalCols / 6],
               [region_three[0] + totalRows / 6, region_three[1] + totalCols / 6],
               [region_six[0] + totalRows / 6, region_six[1] + totalCols / 6],
               [region_five[0] + totalRows / 6, region_five[1] + totalCols / 6],
               [region_four[0] + totalRows / 6, region_four[1] + totalCols / 6],
               [region_seven[0] + totalRows / 6, region_seven[1] + totalCols / 6],
               [region_eight[0] + totalRows / 6, region_eight[1] + totalCols / 6],
               [region_nine[0] + totalRows / 6, region_nine[1] + totalCols / 6]]
    current_target_index = 0

    # Set environment parameters
    """
    sight_distance = 2 means the drone can see two cells in each direction, for a total of a 5x5 area (vision_size)
    """
    sight_distance = 2
    vision_size = (sight_distance * 2 + 1) * (sight_distance * 2 + 1)
    start_row = sight_distance
    start_col = sight_distance
    row_position = start_row
    col_position = start_col

    def reset_env(self, visits, maps):
        """
        This function is run at the start of each episode. It resets the whole environment, so that the drone
        starts back in the top left and all of its map info it has gathered is wiped
        :param visits: visited.py object for keeping track of visited areas
        :param maps: map.py object for keeping track of the information map
        :return:
        """
        # Initialize tracking
        #self.__class__.map = np.zeros([self.totalRows, self.totalCols])
        #self.__class__.visited = np.ones([self.totalRows, self.totalCols])
        maps.reset_map()
        self.__class__.map = maps.map
        visits.reset_visited()
        self.__class__.visited = visits.visited

        # Reset env parameters
        self.start_row = random.randint(12, 47) #was 47 - 167 for full random
        self.start_col = random.randint(12, 47)
        """
        #This set of code for full random start
        for region_index in range(9):
            if self.regions[region_index][0] < self.start_row < self.regions[region_index][0]+60 and self.regions[region_index][1] \
                    < self.start_col < self.regions[region_index][1]+60:
                self.__class__.current_target_index = region_index
        """
        self.__class__.current_target_index = 0
        self.__class__.row_position = self.start_row
        self.__class__.col_position = self.start_col

    def get_classified_drone_image(self):
        """
        This function returns an image which is the 5x5 vision with mining probabilities
        For the field, this should be replaced with whatever method is used to retrieve the mining probabilities
        directly from the drone's camera
        :return: mining probability for each cell within the drone's current vision
        """
        self.sim.setDroneImgSize(self.sight_distance, self.sight_distance)
        self.sim.setNavigationMap()
        image = self.sim.getClassifiedDroneImageAt(self.__class__.row_position, self.__class__.col_position)
        return image

    def visited_position(self):
        """
        Updates array that keeps track of the previous cells visited by the drone (doesn't consider peripheral vision)
        :return: void
        """
        self.__class__.visited[self.__class__.row_position, self.__class__.col_position] = 0


    """
    The next two functions are used to sync up information between search, trace, and target environments.
    Refer to full_testing.py for proper use
    """
    def update_visited(self, visited):
        for i in range(self.totalRows):
            for j in range(self.totalCols):
                if visited[i][j] == 0:
                    self.__class__.visited[i][j] = 0

    def transfer_map(self, map):
        for i in range(self.totalRows):
            for j in range(self.totalCols):
                if map[i][j] > 0:
                    self.__class__.map[i][j] = map[i][j]

    def update_map(self, image):
        """
        Adds new state information to the map of the entire region
        :param image: 2d array of mining probabilities within the drone's vision (5x5)
        :return: void
        """
        for i in range(self.sight_distance*2):
            for j in range(self.sight_distance*2):
                self.__class__.map[self.__class__.row_position + i - self.sight_distance, self.__class__.col_position + j - self.sight_distance] = image[i, j]

    def save_local_map(self, fname):
        """
        Saves the current local map of the drone to an image file
        :param fname: name of image to save
        :return: void
        """
        plt.imshow(self.local_map[:, :], cmap='gray', interpolation='none')
        plt.title("Local Map")
        plt.savefig(fname)
        # plt.show()
        plt.clf()

    def save_map(self, fname, map_obj):
        """
        Saves the current region map of the drone to an image file
        :param fname: name of image to save
        :return: void
        """
        plt.imshow(map_obj.map[:, :], cmap='gray', interpolation='none')
        plt.title("Region Map")
        plt.savefig(fname)
        plt.clf()

    def plot_path(self, fname):
        """
        Saves the current path of the drone to an image file
        :param fname: name of image to save
        :return: void
        """
        plt.imshow(self.__class__.visited[:, :], cmap='gray', interpolation='none')
        plt.title("Drone Path")
        plt.savefig(fname)
        plt.clf()

    def calculate_covered(self, size):
        """
        This function is used for us to observe performance via amount of area of interest covered.
        In the field, this function will not be very helpful because we calculate the percent by using
        the ground-truth data from ICRSsimulator.py. However, a similar function should probably be created as
        a metric of field-performance
        :param size: used to specify what kind of coverage we are calculating
        :return: percent covered of 'size' coverage
        """
        covered = 0

        if size == 'local':
            for i in range(25):
                for j in range(25):
                    if self.__class__.visited[i][j] < 1:
                        covered += 1

            percent_covered = covered / (25*25)

        elif size == 'region':
            for i in range(int(self.totalRows/3)):
                for j in range(int(self.totalCols/3)):
                    if self.__class__.visited[i+self.regions[self.current_target_index][0]][j+self.regions[self.current_target_index][1]] < 1:
                        covered += 1

            percent_covered = covered / (self.totalRows/3)**2

        elif size == 'mining':
            mining = 0
            for i in range(int(self.totalRows)):
                for j in range(int(self.totalCols)):
                    if self.sim.ICRSmap[i, j, 0] > 0:
                        if self.__class__.map[i][j] > 0:
                            covered += self.sim.ICRSmap[i, j, 0]
                        mining += self.sim.ICRSmap[i, j, 0]
            percent_covered = covered / mining

        else:
            for i in range(self.totalRows):
                for j in range(self.totalCols):
                    if self.__class__.visited[i][j] < 1:
                        covered += 1

            percent_covered = covered / (self.totalCols * self.totalRows)

        return percent_covered

    def update_target(self, next_target):
        """
        Sets the drone's current target equal to the one the target selector chooses
        :param next_target: whichever target is outputted by the target selector
        :return:
        """
        self.__class__.current_target_index = next_target

    def set_simulation_map(self):
        """
        sets the pixel value thresholds for ICRS classification
        :return: void
        """
        # Simulate classification of areas of interest
        lower = np.array([0, 0, 230]) #Was 0, 0, 230 for new maps (0, 0, 70) for old maps
        upper = np.array([160, 160, 400])
        interest_value = 1  # Mark these areas as being of highest interest
        self.sim.classify('Area of Interest', lower, upper, interest_value)
        """
        # Simulate classification of obstacles
        lower = np.array([200, 50, 50])
        upper = np.array([400, 200, 200])
        interest_value = 0  # Mark these areas as being of no interest
        self.sim.classify('Obstacle', lower, upper, interest_value)
        """

        self.sim.setMapSize(self.totalRows, self.totalCols)
        self.sim.createMap()