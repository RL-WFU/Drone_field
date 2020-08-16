from Environment.base_env import *


class SelectTarget(Env):
    """
    This environment is in charge of picking the next region the drone should navigate to
    It is called either at the end of a trace episode, or when trace is not successful within the current target
    It uses the discovered mining in each region, the percent of the region already covered, and the
    euclidean distance to each region to decide which region to navigate to
    """
    def __init__(self, visits):
        # Set simulation
        self.set_simulation_map()

        self.visits = visits
        # Define initial targets
        self.__class__.current_target_index = 0
        self.current_target = self.targets[self.__class__.current_target_index]

        # Set task-specific parameters
        self.num_actions = 9
        self.region_values = np.zeros([9, 3])

        # Set reward values
        self.MINING_REWARD = 100
        self.DISTANCE_PENALTY = -2
        self.COVERED_PENALTY = -1500
        self.HOVER_PENALTY = -100

        self.correct_target = False

    def set_target(self, next_target, row, col):
        """
        This function sets the chosen target as the current_target_index for the search and trace environments
        :param next_target: The chosen target
        :param row: current row position
        :param col: current col position
        :return:
        """
        self.__class__.row_position = row
        self.__class__.col_position = col

        self.current_target = self.targets[next_target]

        self.update_regions()

        #The reward was mostly used for when we were trying to train this as an RL agent. Since we found it
        #works better as an equation, the reward is now only for monitoring performance
        reward = self.get_reward(next_target)

        self.__class__.current_target_index = next_target

        state = self.region_values.reshape(1, 27)

        return next_target, state, reward

    def get_reward(self, next_target):

        hover = False
        if next_target == self.__class__.current_target_index:
            hover = True

        reward = self.region_values[next_target, 0]*self.MINING_REWARD + self.region_values[next_target, 1]*self.COVERED_PENALTY + \
            self.region_values[next_target, 2]*self.DISTANCE_PENALTY + hover*self.HOVER_PENALTY

        return reward

    def get_state(self):
        self.update_regions()
        state = self.region_values.reshape(1, 27)
        return state

    def update_regions(self):
        """
        This function calculates the mining and coverage that we found/covered from search and trace episodes,
        and updates self.region_values to include that information
        :return:
        """
        self.region_values[:, 0] = self.get_mining()
        self.region_values[:, 1] = self.get_covered()
        self.region_values[:, 2] = self.get_distance()

    def get_mining(self):
        """
        Calculates the percent of the region, out of what has been seen, that has been mining. We define mining
        as a probability >0, since our images are precise, but in the field this value should be higher, since most
        cells will not have exactly 0 probability of mining
        :return: num cells with >0 mining divided by the num cells that have been covered, for each region (array of length 9)
        """
        mining = np.zeros(9)

        for i in range(9):
            v = 0
            for j in range(int(self.totalRows/3)):
                for k in range(int(self.totalCols/3)):
                    if self.map[int(j+self.regions[i][0]), int(k+self.regions[i][1])] > 0:
                        mining[i] += 1
                    if self.visited[int(j+self.regions[i][0]), int(k+self.regions[i][1])] < 1:
                        v += 1
            mining[i] /= v+1

        return mining

    def get_covered(self):
        """

        :return: the percentage of each region that the drone has flown over (array of length 9)
        """
        covered = np.zeros(9)
        for i in range(9):
            for j in range(int(self.totalRows/3)):
                for k in range(int(self.totalRows/3)):
                    if self.visited[int(j+self.regions[i][0]), int(k+self.regions[i][1])] < 1:
                        covered[i] += 1
            covered[i] = covered[i]/((self.totalRows/3)**2)

        return covered

    def get_distance(self):
        """

        :return: euclidean distance from current position to position of each target (array of length 9)
        """
        distance = np.zeros(9)
        for i in range(9):
            distance[i] = math.sqrt((self.__class__.row_position - self.targets[i][0])**2 +
                                    (self.__class__.col_position - self.targets[i][1])**2)

        return distance

    def select_next_target(self, row, col):
        """
        This function is the formula for picking the next target. We use a weighted sum of mining, coverage, and distance,
        and pick the target for which that value is the highest
        :param row: current row position
        :param col: current col position
        :return: index of chosen target
        """
        self.__class__.row_position = row
        self.__class__.col_position = col
        self.update_regions()
        next_targets = np.zeros(9)
        for i in range(9):
            next_targets[i] = self.region_values[i, 0]*self.MINING_REWARD + self.region_values[i, 1]*self.COVERED_PENALTY + \
                self.region_values[i, 2]*self.DISTANCE_PENALTY
            if i == self.current_target_index:
                next_targets[i] += self.HOVER_PENALTY

        # print(self.region_values)
        self.target_value = np.amax(next_targets)
        return np.argmax(next_targets)

    def simple_select(self):
        next_target = self.current_target_index + 1
        return next_target
