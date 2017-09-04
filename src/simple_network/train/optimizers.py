import tensorflow as tf


def adam_optimizer(loss, learning_rate, optimizer_data, name=""):
    # Define name
    if name:
        train_name = "train_{}".format(name)
    else:
        train_name = "train"
    # Define optimizer
    train_step = None
    with tf.name_scope(train_name):
        beta_1 = optimizer_data.get("beta_1", 0.9)
        beta_2 = optimizer_data.get("beta_2", 0.999)
        epsilon = optimizer_data.get("epsilon", 1e-08)
        optimizer = tf.train.AdamOptimizer(learning_rate, beta_1, beta_2, epsilon)
        train_step = optimizer.minimize(loss)
    return train_step
