import numpy as np


class Visited:
    """
    This class implements a tracker for information of what cells the drone has visited.
    It creates a numRows x numCols map, where each cell is a 1, if it hasn't been visited, and a 0 if it has
    There are two separate visited info maps, a full one (self.visited), and one for search (self.search_visited)
    The full one only gets reset at the end of an entire episode (.7 mining cov).
    The search one gets reset at the start of every search episode. This is because search performs better
    without information on where it has been already, while trace performs better when it has that information
    """
    def __init__(self, numRows, numCols):
        self.rows = numRows
        self.cols = numCols
        self.visited = np.ones([numRows, numCols])
        self.search_visited = np.ones([numRows, numCols])

    def reset_visited(self):
        self.visited = np.ones([self.rows, self.cols])

    def reset_search_visited(self):
        self.search_visited = np.ones([self.rows, self.cols])

    def visited_position(self, row, col, search=False):
        self.visited[row, col] = 0
        if search:
            self.search_visited[row, col] = 0
