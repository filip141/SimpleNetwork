import tensorflow as tf


def cross_entropy(logits, labels, activation="softmax"):
    cross_entr = None
    with tf.name_scope('cross_entropy'):
        if activation is None:
            cross_entr = -tf.reduce_sum(labels * tf.log(logits), 1)
        else:
            if activation == "softmax":
                cross_entr = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            elif activation == "sigmoid":
                cross_entr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
            else:
                raise AttributeError("Activation {} not defined.".format(activation))
        tf.summary.scalar("cross_entropy", cross_entr)
    return cross_entr


