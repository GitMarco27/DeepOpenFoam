import argparse
from collections import OrderedDict
from collections import namedtuple
from itertools import product


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training, validation and testing of AE and 3D regressor')

    # Required positional argument
    parser.add_argument('--data_path', type=str,
                        help='dataset path')

    parser.add_argument('--id', type=str,
                        help='trial name')

    args = parser.parse_args()

