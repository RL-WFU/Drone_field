from Environment.base_env import *
from copy import deepcopy


class Trace(Env):
    """
    This is the environment for the trace decision network. The goal of this agent is to discover mining, and cover
    all of it in the immediate area. The episode ends either at config.max_steps_trace steps, or when it has not been
    very successful in the region in the last 100 timesteps.
    Trace episodes start when the search episode ends, by reaching the region target, and when trace is over,
    the target selector picks a new target. Then search happens, and the cycle repeats
    Trace uses the 5x5 area of mining probabilities around it, the visited information around it, and the map of
    the region it has formed in order to make decisions
    """
    def __init__(self, visits, map_obj):
        # Set simulation
        self.set_simulation_map()

        self.visits = visits
        self.map_obj = map_obj
        # Set task-specific parameters
        self.num_actions = 5

        # Set reward values
        self.MINING_REWARD = 200
        self.COVERAGE_REWARD = 200
        self.VISITED_PENALTY = -10
        self.HOVER_PENALTY = -10

    def reset_tracing(self, row, col):
        """
        Called at the start of each trace episode. Retrieves the local map
        :param row: row position
        :param col: col position
        :return: the state and local map for initial decision-making
        """
        # Get new drone state
        state = self.get_classified_drone_image()
        state = self.flatten_state(state)
        state = np.append(state, 1)
        state = np.append(state, 1)
        state = np.append(state, 1)
        state = np.append(state, 1)
        # state = np.append(state, self.calculate_covered('region'))
        state = np.reshape(state, [1, 1, self.vision_size + 4])

        self.__class__.row_position = row
        self.__class__.col_position = col

        for region_index in range(9):
            if self.regions[region_index][0] < row < self.regions[region_index][0]+60 and self.regions[region_index][1] \
                    < col < self.regions[region_index][1]+60:
                self.__class__.current_target_index = region_index
        # Get new local map
        self.next_local_map()
        self.local_map = self.get_local_map()
        if self.local_map.shape != (25, 25):
            print(self.local_map.shape)
            print('position:', self.__class__.row_position, self.__class__.col_position)
            print('local map:', self.local_map_lower_row, self.local_map_upper_row, self.local_map_lower_col, self.local_map_upper_col)
            print('map shape:', self.map.__class__.shape)
        flattened_local_map = self.local_map.reshape(1, 1, 625)

        return state, flattened_local_map

    def step(self, action, time):
        """
        Step function. Takes the action chosen by the trace network, and returns the new state around it
        :param action: action the trace network chose
        :param time: current time step
        :return: state, flattened local map, reward, whether trace episode is done or not
        """
        self.done = False
        next_row = self.__class__.row_position
        next_col = self.__class__.col_position

        # Drone not allowed to move outside of the current region
        if action == 0:  # Forward one grid
            if self.__class__.row_position < (self.totalRows - self.sight_distance - 1) and self.__class__.row_position < self.regions[self.__class__.current_target_index][0] + 59:
                next_row = self.__class__.row_position + 1
                next_col = self.__class__.col_position
            else:
                action = 4
        elif action == 1:  # right one grid
            if self.__class__.col_position < (self.totalCols - self.sight_distance - 1) and self.__class__.col_position < self.regions[self.__class__.current_target_index][1] + 59:
                next_row = self.__class__.row_position
                next_col = self.__class__.col_position + 1
            else:
                action = 4
        elif action == 2:  # back one grid
            if self.__class__.row_position > self.sight_distance + 1 and self.__class__.row_position > self.regions[self.__class__.current_target_index][0]:
                next_row = self.__class__.row_position - 1
                next_col = self.__class__.col_position
            else:
                action = 4
        elif action == 3:  # left one grid
            if self.__class__.col_position > self.sight_distance + 1 and self.__class__.col_position > self.regions[self.__class__.current_target_index][1]:
                next_row = self.__class__.row_position
                next_col = self.__class__.col_position - 1
            else:
                action = 4

        self.__class__.row_position = next_row
        self.__class__.col_position = next_col
        # print(self.row_position, self.col_position)
        # print(self.local_target)

        image = self.get_classified_drone_image()

        if time > self.config.max_steps_trace:
            self.done = True

        reward = self.get_reward(image, action)

        self.visited_position()
        self.map_obj.update_map(image, self.__class__.row_position, self.__class__.col_position)
        self.__class__.map = self.map_obj.map
        #self.update_map(image)

        #Choose a new local map if it leaves the current one
        if (self.__class__.row_position < self.local_map_lower_row or self.__class__.col_position <
                self.local_map_lower_col or self.__class__.row_position > self.local_map_upper_row or
                self.__class__.col_position > self.local_map_upper_col) and 12 < self.__class__.row_position < \
                self.totalRows - 12 and 12 < self.__class__.col_position < self.totalCols - 12:
            self.next_local_map()

        self.local_map = self.get_local_map()
        flattened_local_map = self.local_map.reshape(1, 1, 625)

        #return the new state and local map
        state = self.flatten_state(image)
        state = np.append(state, self.__class__.visited[self.__class__.row_position + 1, self.__class__.col_position])
        state = np.append(state, self.__class__.visited[self.__class__.row_position, self.__class__.col_position + 1])
        state = np.append(state, self.__class__.visited[self.__class__.row_position - 1, self.__class__.col_position])
        state = np.append(state, self.__class__.visited[self.__class__.row_position, self.__class__.col_position + 1])
        # state = np.append(state, self.calculate_covered('region'))
        state = np.reshape(state, [1, 1, self.vision_size + 4])

        return state, flattened_local_map, reward, self.done

    def get_reward(self, image, action):
        """
        Calculates reward based on target, mining seen, and whether the current state has already been visited
        :param image: 2d array of mining probabilities within the drone's vision
        :return: reward value
        """
        mining_prob = 2*image[self.sight_distance, self.sight_distance]

        reward = mining_prob*self.MINING_REWARD*self.__class__.visited[self.__class__.row_position, self.__class__.col_position]

        if action == 4:
            reward += self.HOVER_PENALTY

        if self.__class__.visited[self.__class__.row_position, self.__class__.col_position] == 0:
            reward += self.VISITED_PENALTY

        if self.done:
            reward += self.calculate_covered('region')*self.COVERAGE_REWARD

        return reward

    def get_local_map(self):
        """
        Creates local map (shape: 25x25) of mining areas from the region map
        :return: local_map
        """
        local_map = deepcopy(self.map_obj.map[self.local_map_lower_row:self.local_map_upper_row + 1,
                             self.local_map_lower_col:self.local_map_upper_col + 1])
        #local_map = deepcopy(self.__class__.map[self.local_map_lower_row:self.local_map_upper_row+1, self.local_map_lower_col:self.local_map_upper_col+1])
        return local_map

    def next_local_map(self):
        """
        Sets boundaries on local map, placing the drone at the center of the new map
        :return: void
        """
        self.local_map_lower_row = self.__class__.row_position - 12
        self.local_map_upper_row = self.__class__.row_position + 12
        self.local_map_lower_col = self.__class__.col_position - 12
        self.local_map_upper_col = self.__class__.col_position + 12

        # Corrects for indexing error if drone is too close to a border
        if self.local_map_lower_row < 0:
            self.local_map_lower_row = 0
            self.local_map_upper_row = 24
        if self.local_map_lower_col < 0:
            self.local_map_lower_col = 0
            self.local_map_upper_col = 24
        if self.local_map_upper_row > 179:
            self.local_map_upper_row = 179
            self.local_map_lower_row = 155
        if self.local_map_upper_col > 179:
            self.local_map_upper_col = 179
            self.local_map_lower_col = 155

    def flatten_state(self, state):
        """
        :param state: 2d array of mining probabilities within the drone's vision
        :return: one dimensional array of the state information
        """
        flat_state = state.reshape(1, self.vision_size)
        return flat_state

    def visited_position(self):
        self.visits.visited_position(self.__class__.row_position, self.__class__.col_position)
        self.__class__.visited = self.visits.visited
