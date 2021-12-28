import tensorflow as tf


def chamfer_distance(x, y):
    x = tf.transpose(x, [0, 2, 1])
    y = tf.transpose(y, [0, 2, 1])

    x = tf.expand_dims(x, axis=3)  # shape [b, d, n, 1]
    # print(x.shape)
    y = tf.expand_dims(y, axis=2)  # shape [b, d, 1, m]
    # print(y.shape)

    d = tf.square(x - y)  # shape [b, d, n, m]
    # print(d.shape)
    d = tf.math.reduce_sum(d, axis=1)  # shape [b, n, m]
    # print(d.shape)

    min_for_each_x_i = tf.math.reduce_min(d, axis=2)  # shape [b, n]
    # print(min_for_each_x_i.shape)
    min_for_each_y_j = tf.math.reduce_min(d, axis=1)  # shape [b, m]
    # print(min_for_each_y_j.shape)

    distance = tf.math.reduce_sum(min_for_each_x_i, axis=1) + tf.math.reduce_sum(min_for_each_y_j, axis=1)
    # print(distance.shape)  # shape [b]

    distance = tf.reduce_mean(distance, axis=0)
    # print(distance)  # shape []
    return distance
