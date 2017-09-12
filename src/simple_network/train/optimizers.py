import tensorflow as tf


def adam_optimizer(loss, learning_rate, optimizer_data, name=""):
    # Define name
    if name:
        train_name = "train_{}".format(name)
    else:
        train_name = "train"
    # Define optimizer
    with tf.name_scope(train_name):
        beta_1 = optimizer_data.get("beta_1", 0.9)
        beta_2 = optimizer_data.get("beta_2", 0.999)
        epsilon = optimizer_data.get("epsilon", 1e-08)
        optimizer = tf.train.AdamOptimizer(learning_rate, beta_1, beta_2, epsilon)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(loss)
    return train_step


def momentum_optimizer(loss, learning_rate, optimizer_data, name=""):
        # Define name
    if name:
        train_name = "train_{}".format(name)
    else:
        train_name = "train"
    # Define optimizer
    with tf.name_scope(train_name):
        momentum = optimizer_data.get("momentum", 0.9)
        use_locking = optimizer_data.get("use_locking", False)
        use_nesterov = optimizer_data.get("use_nesterov", False)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_locking=use_locking,
                                               use_nesterov=use_nesterov)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(loss)
    return train_step


def rmsprop_optimizer(loss, learning_rate, optimizer_data, name=""):
    # Define name
    if name:
        train_name = "train_{}".format(name)
    else:
        train_name = "train"
    # Define optimizer
    with tf.name_scope(train_name):
        decay = optimizer_data.get("decay", 0.9)
        momentum = optimizer_data.get("momentum", 0.0)
        epsilon = optimizer_data.get("epsilon", 1e-10)
        use_locking = optimizer_data.get("use_locking", False)
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum,
                                              epsilon=epsilon, use_locking=use_locking)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(loss)
    return train_step