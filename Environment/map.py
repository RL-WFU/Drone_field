import numpy as np

class Map:
    """
    This class implements a map object, which contains the information the drone has gathered across an entire episode
    This map is numRows x numCols, and each cell contains the probability of mining in that cell, but
    only if the drone has seen that cell (the cell has been in the drone's 5x5 vision area at least once)
    Comparing this map with the ground-truth map has allowed us to see the area the drone is covering
    This map is also accessed to provide the drone with local map information in search and trace. To do so,
    we take the section of this map corresponding to the drone's current region, and pass that through the network
    as the local map. So, the drone has access to the information it has gathered in its current region at each
    timestep
    Like visited.py, there are two maps, the full one, and the search map. The full map is only reset at the start
    of each larger episode, while the search one is reset at the start of each search episode. This is because, like visited,
    search performs better when it is less overwhelmed with prior information, while trace can effectively use
    the prior information
    """
    def __init__(self, numRows, numCols):
        self.rows = numRows
        self.cols = numCols
        self.map = np.zeros([self.rows, self.cols])
        self.search_map = np.zeros([self.rows, self.cols])
        self.sight_distance = 2


    def reset_map(self):
        self.map = np.zeros([self.rows, self.cols])

    def reset_search_map(self):
        self.search_map = np.zeros([self.rows, self.cols])

    def update_map(self, image, row, col, search=False):
        """

        :param image: the 5x5 image the drone sees
        :param row: drone's row position
        :param col: drone's col position
        :param search: whether this update is for a search episode (true) or trace episode (false)
        :return:
        """
        for i in range(self.sight_distance*2):
            for j in range(self.sight_distance*2):
                self.map[row + i - self.sight_distance, col + j - self.sight_distance] = image[i, j]
                if search:
                    self.search_map[row + i - self.sight_distance, col + j - self.sight_distance] = image[i, j]