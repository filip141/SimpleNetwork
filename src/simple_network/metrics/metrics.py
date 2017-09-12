import tensorflow as tf
from simple_network.train.losses import cross_entropy as cross_ent
from simple_network.train.losses import mean_square as mse_loss
from simple_network.train.losses import mean_absolute as mae_loss


def cross_entropy(logits, labels):
    return cross_ent(logits=logits, labels=labels)


def accuracy(logits, labels):
    with tf.name_scope('accuracy'):
        is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_val = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        tf.summary.scalar("accuracy", accuracy_val)
    return accuracy_val


def mean_square(logits, labels):
    mse = mse_loss(logits=logits, labels=labels)
    return mse


def mean_absolute(logits, labels):
    mae = mae_loss(logits=logits, labels=labels)
    return mae
