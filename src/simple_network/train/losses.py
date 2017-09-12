import tensorflow as tf


def cross_entropy(logits, labels, loss_data=None):
    if loss_data is None:
        loss_data = {}
    activation = loss_data.get("activation", "softmax")
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


def mean_square(logits, labels, loss_data=None):
    with tf.name_scope('mean_square'):
        mse = tf.reduce_mean(tf.pow(logits - labels, 2))
        tf.summary.scalar("mean_square", mse)
    return mse


def mean_absolute(logits, labels, loss_data=None):
    with tf.name_scope('mean_absolute'):
        mae = tf.reduce_mean(tf.abs(logits - labels))
        tf.summary.scalar("mean_absolute", mae)
    return mae


def mean_absolute_weight(logits, labels, loss_data=None):
    if not isinstance(logits, list):
        raise AttributeError("Loss only available for Node output.")
    loss = 0
    nimages = loss_data.get("nimages", 2)
    reshape_weights = loss_data.get("reshape_weights", None)
    with tf.name_scope('mean_absolute_weight'):
        batch_images_w = tf.split(value=logits[1], num_or_size_splits=nimages, axis=0)
        batch_img_pred = tf.split(value=logits[0], num_or_size_splits=nimages, axis=0)
        batch_img_true_labels = tf.split(value=labels, num_or_size_splits=nimages, axis=0)
        for b_i_w, b_i_p, b_t_l in zip(batch_images_w, batch_img_pred, batch_img_true_labels):
            b_i_w = tf.reshape(b_i_w, [-1]) + 0.000001
            b_i_p = tf.reshape(b_i_p, [-1])
            b_t_l = tf.reshape(b_t_l, [-1])
            if reshape_weights is not None:
                w_img = tf.reshape(b_i_w, [1, reshape_weights[0], reshape_weights[1], 1])
                tf.summary.image("weight_img", w_img, 1)
            estimated_label = tf.reduce_sum(b_i_w * b_i_p) / tf.reduce_sum(b_i_w)
            loss += tf.reduce_mean(tf.abs(estimated_label - b_t_l[0]))
        tf.summary.scalar("mean_absolute_weight", loss)
    return loss

