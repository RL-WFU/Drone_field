import argparse
from Testing.full_testing import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--search_weights', default="search_weights", type=str, help='weights to load for search model')
    parser.add_argument('--trace_weights', default="trace_weights", type=str, help='weights to load for trace model')
    args = parser.parse_args()


    print('Testing...')
    test_full_model(True, args.search_weights,
                    args.trace_weights)
