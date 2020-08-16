from ddrqn import *
from A2C import *
from Environment.search_env import *
from Environment.tracing_env import *
from Environment.target_selector_env import *
from Testing.testing_helper import *
from Environment.visited import *
from Environment.map import *

"""
Tests the full model

Saves plotting to Testing_results/Search and Testing_results/Trace
"""


def test_full_model(target_cost=False, search_weights=None, trace_weights=None, target_weights=None):
    """
    This is our function which tests the performance of the fully-trained drone. It runs a number of episodes
    (specified in config), where each episode consists of the drone starting in a randomized location within the
    first target (top left), and repeating the process - target selection, search, trace - until at least .7
    of the total mining in the map has been covered. We find that on randomly generated maps, the drone can cover
    this amount of area of interest in ~15000 avg steps, as opposed to the baseline of ~23000 when focusing solely
    on coverage.

    :param target_cost:
    :param search_weights:
    :param trace_weights:
    :param target_weights:
    :return:
    """
    # Initialize environment and ddrqn agents
    visited = Visited(config.total_rows, config.total_cols)
    map_obj = Map(config.total_rows, config.total_cols)
    search = Search(visited, map_obj)
    trace = Trace(visited, map_obj)
    target = SelectTarget(visited)
    action_size = search.num_actions
    sess = tf.Session()
    searching_agent = DDRQNAgent(search.vision_size+6, action_size, 'Search', sess)


    tracing_agent = A2CAgent(trace.vision_size + 4, action_size, 'Trace', sess)

    #initialize tensorflow parameters to be loaded
    sess.run(tf.global_variables_initializer())

    #These functions load our saved weights in the folder 'Weights'
    if search_weights is not None:
        searching_agent.load(search_weights + '_model', search_weights + '_target')
    if trace_weights is not None:
        tracing_agent.load(trace_weights + '_policy', trace_weights + '_value')


    done = False

    # Initialize episode logging
    search_rewards = []
    search_covered = []
    search_steps = []
    average_over = 10
    search_average_rewards = []
    search_average_r = deque(maxlen=average_over)
    search_episode_num = 0

    trace_rewards = []
    trace_covered = []
    trace_steps = []
    trace_average_rewards = []
    trace_average_r = deque(maxlen=average_over)
    trace_episode_num = 0

    total_steps = []


    for e in range(config.num_episodes):
        mining_coverage = []
        search.reset_env(visited, map_obj)
        trace.reset_env(visited, map_obj)
        target.reset_env(visited, map_obj)
        t = 0

        #Episode ends when we reach .7 mining coverage
        while target.calculate_covered('mining') < .7:
            mining = target.calculate_covered('mining')
            print('Mining Coverage:', mining)
            mining_coverage.append(mining)
            print('Total Steps:', t)

            # Complete one searching episode
            reward, steps, row_position, col_position = search_episode(search, searching_agent,
                                                                       trace.row_position, trace.col_position)
            search_rewards.append(reward)
            search_covered.append(search.calculate_covered('mining'))
            search_steps.append(steps)

            search_average_r.append(reward)

            if search_episode_num < average_over:
                r = 0
                for i in range(search_episode_num):
                    r += search_average_r[i]
                r /= (search_episode_num + 1)
                search_average_rewards.append(r)
            else:
                search_average_rewards.append(sum(search_average_r) / average_over)

            if search_episode_num % average_over == 0:
                save_plots(e + 1, search, 'Search', search_average_rewards, search_rewards,
                           mining_coverage=search_covered, map_obj=map_obj)

            #save_plots(e, search, 'Search')
            print("search episode: {} - {}, reward: {}, mining covered: {}, start position: {},{}, number of steps: {}"
                  .format((search_episode_num+1 % (e+1)), e+1, reward, search_covered[search_episode_num],
                          trace.row_position, trace.col_position, steps))

            search_episode_num += 1
            t += steps

            # Update all environments with the new information after the search episode
            trace.update_visited(search.visited)
            trace.transfer_map(search.map)
            target.update_visited(search.visited)
            target.transfer_map(search.map)

            # Complete one tracing episode
            reward, steps, row_position, col_position = trace_episode(trace, tracing_agent,
                                                                      search.row_position, search.col_position, target)
            trace_rewards.append(reward)
            trace_covered.append(trace.calculate_covered('mining'))
            trace_steps.append(steps)

            trace_average_r.append(reward)

            if trace_episode_num < average_over:
                r = 0
                for i in range(trace_episode_num):
                    r += trace_average_r[i]
                r /= (trace_episode_num + 1)
                trace_average_rewards.append(r)
            else:
                trace_average_rewards.append(sum(trace_average_r) / average_over)

            if trace_episode_num % average_over == 0:
                save_plots(e + 1, trace, 'Trace', trace_average_rewards, trace_rewards,
                           mining_coverage=trace_covered, map_obj=map_obj)

            #save_plots(e, trace, 'Trace')
            print("trace episode: {} - {}, reward: {}, mining covered: {}, start position: {},{}, number of steps: {}"
                  .format((trace_episode_num+1 % (e+1)), e+1, reward, trace_covered[trace_episode_num],
                          search.row_position, search.col_position, steps))

            trace_episode_num += 1
            t += steps

            # Update all environments with the new information after the trace episode
            search.update_visited(trace.visited)
            search.transfer_map(trace.map)
            target.update_visited(trace.visited)
            target.transfer_map(trace.map)

            #Pick a new target
            next_target = target.select_next_target(trace.row_position, trace.col_position)



            # Update all environments with the new target
            search.update_target(next_target)
            trace.update_target(next_target)
            target.update_target(next_target)
            print("Next target:", next_target)


        total_steps.append(t)
        print('***********')
        print("EPISODE {} COMPLETE: Steps -- {}, Mining Coverage -- {}, Total Coverage: {}"
              .format(e+1, t, target.calculate_covered('mining'), target.calculate_covered('map')))
        print('***********')

        plt.plot(total_steps)
        plt.xlabel('Episode')
        plt.ylabel('Total Steps Taken')
        plt.savefig('Testing_results/steps.png')
        plt.clf()

