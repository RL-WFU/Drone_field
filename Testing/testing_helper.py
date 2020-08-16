from configurationSimple import ConfigSimple as config
import collections
import numpy as np
import matplotlib.pyplot as plt


def search_episode(search, searching_agent, row_position, col_position):
    """
    This function implements one episode of the search agent. The goal of this agent is to reach
    the specified target, out of the 9 region targets, which was picked by the target_selector_env
    It uses the last five "states" to make decisions on its actions. Refer to the search_env file for more
    information on these states

    :param search: object of search environment
    :param searching_agent: object of search RL agent
    :param row_position: starting row position of agent
    :param col_position: starting col position of agent
    :return: episode reward, number of steps taken, ending row position, ending col position
    """
    Transition = collections.namedtuple("Transition",
                                        ["state", "local_map", "action", "reward", "next_state", "next_local_map",
                                         "done"])
    episode = []
    total_reward = 0
    t = 0
    state, local_map = search.reset_search(row_position, col_position)


    for time in range(config.max_steps_search):
        if time < 5:
            action = np.random.randint(0, 5)
        else:
            states, local_maps = get_last_t_states(5, episode, search.vision_size + 6)
            action = searching_agent.act(states, local_maps)

        next_state, next_local_map, reward, done = search.step(action, time)

        total_reward += reward

        episode.append(Transition(
            state=state, local_map=local_map, action=action, reward=reward, next_state=next_state,
            next_local_map=next_local_map, done=done))

        state = next_state
        local_map = next_local_map

        if done:
            break

        t = time

    return total_reward, t, search.row_position, search.col_position


def trace_episode(trace, tracing_agent, row_position, col_position, target=None):
    """
    This function implements a single episode of trace. After reaching the target from search, an episode of trace
    is run, where the drone uses information it has gathered during search, and new information it encounters,
    to locate and cover areas of interest. If it finds little in its immediate area, we pick a new target and
    repeat the search-trace process

    :param trace: object of trace environment
    :param tracing_agent: object of trace RL agent
    :param row_position: starting row position of trace agent
    :param col_position: starting col position of trace agent
    :param target: object of target selection env. This should always be passed in testing
    :return: episode reward, time steps taken, ending row position, ending col position
    """
    Transition = collections.namedtuple("Transition",
                                        ["state", "local_map", "action", "reward", "next_state", "next_local_map",
                                         "done"])
    episode = []
    total_reward = 0
    t = 0
    state, local_map = trace.reset_tracing(row_position, col_position)
    coverage = trace.calculate_covered('mining')

    #print("Start Trace: {}".format(np.sum(trace.visited)))

    for time in range(config.max_steps_trace):
        if time < 5:
            action = np.random.randint(0, 5)
        else:
            states, local_maps = get_last_t_states(5, episode, trace.vision_size + 4)
            action = tracing_agent.act(states, local_maps)

        next_state, next_local_map, reward, done = trace.step(action, time)
        total_reward += reward

        episode.append(Transition(
            state=state, local_map=local_map, action=action, reward=reward, next_state=next_state,
            next_local_map=next_local_map, done=done))

        state = next_state
        local_map = next_local_map

        if done:
            break

        if target is not None:
            if (time + 1) % 100 == 0:
                new_coverage = trace.calculate_covered('mining')
                if new_coverage - coverage < .005:
                    next_target = target.select_next_target(trace.row_position, trace.col_position)
                    if next_target != trace.current_target_index:
                        break
                coverage = new_coverage

        t = time

    #print("End Trace: {}".format(np.sum(trace.visited)))

    return total_reward, t, trace.row_position, trace.col_position


def get_last_t_states(t, episode, size):
    states = []
    maps = []
    for i, transition in enumerate(episode[-t:]):
        states.append(transition.state)
        maps.append(transition.local_map)

    states = np.asarray(states)
    states = np.reshape(states, [1, t, size])

    maps = np.asarray(maps)
    maps = np.reshape(maps, [1, t, 25*25])

    return states, maps

def save_plots(num, agent, folder, average_rewards=None, episode_rewards=None, episode_covered=None, mining_coverage=None, map_obj=None):
    agent.save_local_map('Testing_results/ddrqn_local_map' + str(num) + '.jpg')
    agent.plot_path('Testing_results/ddrqn_drone_path' + str(num) + '.jpg')
    agent.save_map('Testing_results/ddrqn_map' + str(num) + '.jpg', map_obj)

    if average_rewards is not None:
        plt.plot(average_rewards)
        plt.ylabel('Averaged Episode reward')
        plt.xlabel('Episode')
        plt.savefig('Testing_results/' + str(folder) + '/ddrqn_average_reward.png')
        plt.clf()

    if episode_rewards is not None:
        plt.plot(episode_rewards)
        plt.ylabel('Episode reward')
        plt.xlabel('Episode')
        plt.savefig('Testing_results/' + str(folder) + '/ddrqn_reward.png')
        plt.clf()

    if episode_covered is not None:
        plt.plot(episode_covered)
        plt.ylabel('Percent Covered')
        plt.xlabel('Episode')
        plt.savefig('Testing_results/ddrqn_coverage.png')
        plt.clf()

    if mining_coverage is not None:
        plt.plot(mining_coverage)
        plt.xlabel('Iteration')
        plt.ylabel('Episode Mining Coverage')
        plt.savefig('Testing_results/mining_coverage.png')
        plt.clf()
