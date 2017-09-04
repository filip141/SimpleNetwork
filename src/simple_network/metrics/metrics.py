import tensorflow as tf
from simple_network.train.losses import cross_entropy as cross_ent


def cross_entropy(logits, labels, activation='softmax'):
    return cross_ent(logits=logits, labels=labels, activation=activation)


def accuracy(logits, labels, activation='softmax'):
    with tf.name_scope('accuracy'):
        is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_val = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        tf.summary.scalar("accuracy", accuracy_val)
    return accuracy_val
