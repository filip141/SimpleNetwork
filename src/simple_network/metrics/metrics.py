import tensorflow as tf
from simple_network.train.losses import cross_entropy as cross_ent
from simple_network.train.losses import mean_square as mse_loss
from simple_network.train.losses import mean_absolute as mae_loss
from simple_network.train.losses import mean_absolute_weight as mae_w_4


def cross_entropy(logits, labels):
    return cross_ent(logits=logits, labels=labels)


def cross_entropy_sigmoid(logits, labels):
    return cross_ent(logits=logits, labels=labels, loss_data={"activation": "sigmoid"})


def accuracy(logits, labels):
    with tf.name_scope('accuracy'):
        is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_val = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        tf.summary.scalar("accuracy", accuracy_val)
    return accuracy_val


def binary_accuracy(logits, labels):
    predicted_class = tf.greater(logits, 0.5)
    correct = tf.equal(predicted_class, tf.equal(labels,1.0))
    accuracy_bin = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.summary.scalar("binary_accuracy", accuracy_bin)
    return accuracy_bin


def mean_square(logits, labels):
    mse = mse_loss(logits=logits, labels=labels)
    return mse


def mean_absolute(logits, labels):
    mae = mae_loss(logits=logits, labels=labels)
    return mae


def mean_absolute_weighted_4(logits, labels):
    loss_data = {"nimages": 4}
    mae = mae_w_4(logits=logits, labels=labels, loss_data=loss_data)
    return mae
