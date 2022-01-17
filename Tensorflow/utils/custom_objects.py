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

def r_squared(y, y_pred):
    """
    :param y: true valuse (tf.Tensor or np.ndarray)
    :param y_pred: predicted values (tf.Tensor or np.ndarray)
    :return:

    r2 score metric for tensorflow
    """
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    return r2

def euclidian_dist_loss(
        true,
        pred,
        a1: float = .5,
        a2: float = 1.,
        correction: bool = True):

    square_error = tf.math.square(true - pred)  # [batch, n_point, 3]
    e_distance = tf.math.sqrt(tf.math.reduce_sum(square_error, axis=-1))  # [batch, n_point]
    all_ed = tf.math.reduce_mean(e_distance)

    if not correction:
        return all_ed

    e_distance_se_row, _ = tf.math.top_k(e_distance, k=int(0.1 * pred.shape[1]))

    custom_ed = tf.math.reduce_mean(e_distance_se_row)

    loss = (a1 * all_ed + a2 * custom_ed) * 1 / 2

    return loss
