import numpy as np


def concatenate_sides(suction_side: np.ndarray, pressure_side: np.ndarray):
    # A label is assigned to identify the side
    # 0: suction side
    # 1: pressure side
    suction_side = np.concatenate((suction_side, np.zeros([suction_side.shape[0], suction_side.shape[1], 1])), axis=2)
    pressure_side = np.concatenate((pressure_side, np.ones([pressure_side.shape[0], pressure_side.shape[1], 1])),
                                   axis=2)

    # the two sides are concatenated
    return np.concatenate((suction_side, pressure_side), axis=1)
