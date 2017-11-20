import logging
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from tensorflow.python.training import moving_averages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def batch_norm(x, scope, is_training, epsilon=0.001, decay=0.99):
    """
    Returns a batch normalization layer that automatically switch between train and test phases based on the
    tensor is_training

    Args:
        x: input tensor
        scope: scope name
        is_training: boolean tensor or variable
        epsilon: epsilon parameter - see batch_norm_layer
        decay: epsilon parameter - see batch_norm_layer

    Returns:
        The correct batch normalization layer based on the value of is_training
    """
    return tf.cond(
        is_training,
        lambda: batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
        lambda: batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
    )


def batch_norm_layer(x, scope, is_training, epsilon=0.001, decay=0.99, reuse=None):
    """
    Performs a batch normalization layer

    Args:
        x: input tensor
        scope: scope name
        is_training: python boolean value
        epsilon: the variance epsilon - a small float number to avoid dividing by 0
        decay: the moving average decay

    Returns:
        The ops of a batch normalization layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        shape = x.get_shape().as_list()
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0),
                                     trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0),
                                     trainable=False)
        if is_training:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, range(len(shape)-1))
            update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    return output


class Layer(object):
    __metaclass__ = ABCMeta

    def __init__(self, layer_type, layer_name, default_name, save_summaries, reuse):
        # Define layer properties
        self.layer_type = layer_type

        # Names and summaries
        self.layer_name = layer_name
        self.default_name = default_name
        self.save_summaries = save_summaries
        self.reuse = reuse

    @abstractmethod
    def build_graph(self, layer_input):
        pass

    @staticmethod
    def build_layer(layer, layer_input, layer_idx, is_training, enable_log=True):
        if isinstance(layer, BatchNormalizationLayer):
            layer.set_training_indicator(is_training)
        elif isinstance(layer, DropoutLayer):
            layer.set_training_indicator(is_training)
        # Add number if default
        if layer.layer_name == layer.default_name:
            layer.layer_name = "{}_{}".format(layer.layer_name, layer_idx)
        layer_output = layer.build_graph(layer_input)
        if enable_log:
            logger.info("{} layer| Input shape: {}, Output shape: {}".format(layer.layer_type, layer.input_shape,
                                                                             layer.output_shape))
        return layer_output


class DropoutLayer(Layer):

    def __init__(self, percent=0.2, name='dropout', summaries=True, reuse=None):
        super(DropoutLayer, self).__init__("DropoutLayer", name, 'dropout', summaries, reuse)
        # Define layer properties
        self.layer_input = None
        self.input_shape = None
        self.output_shape = None
        self.output = None
        self.layer_size = None
        self.is_training = None
        self.percent = percent

    def set_training_indicator(self, is_training):
        self.is_training = is_training

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        self.layer_size = self.input_shape
        with tf.variable_scope(self.layer_name):
            self.output = tf.cond(self.is_training, lambda: tf.nn.dropout(self.layer_input, self.percent),
                                  lambda: tf.nn.dropout(self.layer_input, 1.0))
            self.output_shape = self.output.get_shape().as_list()[1:]
            tf.summary.histogram("dropout_output", self.output)
        return self.output


class BatchNormalizationLayer(Layer):

    def __init__(self, name='batch_normalization', summaries=True, reuse=None):
        super(BatchNormalizationLayer, self).__init__("BatchNormalizationLayer", name,
                                                      'batch_normalization', summaries, reuse)
        # Define layer properties
        self.layer_input = None
        self.input_shape = None
        self.output_shape = None
        self.output = None
        self.layer_size = None
        self.is_training = None

    def set_training_indicator(self, is_training):
        self.is_training = is_training

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        self.layer_size = self.input_shape
        with tf.name_scope(self.layer_name):
            self.output = batch_norm(self.layer_input, "bn_{}".format(self.layer_name), self.is_training)
            self.output_shape = self.output.get_shape().as_list()[1:]
            if self.save_summaries:
                with tf.variable_scope("bn_{}".format(self.layer_name), reuse=True):
                    beta = tf.get_variable("beta", [self.layer_size[-1]])
                    gamma = tf.get_variable("gamma", [self.layer_size[-1]])
                    tf.summary.scalar('bn_beta', beta[0])
                    tf.summary.scalar('bn_gamma', gamma[0])
                tf.summary.histogram("batch_normalization", self.output)
        return self.output


class LocalResponseNormalization(Layer):

    def __init__(self, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, name='local_response_normalization',
                 summaries=True, reuse=None):
        super(LocalResponseNormalization, self).__init__("LocalResponseNormalization", name,
                                                         'local_response_normalization', summaries, reuse)
        # Define layer properties
        self.layer_input = None
        self.input_shape = None
        self.output_shape = None
        self.output = None
        self.layer_size = None
        self.depth_radius = depth_radius
        self.alpha = alpha
        self.beta = beta
        self.bias = bias

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        self.layer_size = self.input_shape
        with tf.name_scope(self.layer_name):
            self.output = tf.nn.local_response_normalization(self.layer_input, depth_radius=self.depth_radius,
                                                             alpha=self.alpha, beta=self.beta, bias=self.bias,
                                                             name=self.layer_name)
            if self.save_summaries:
                self.output_shape = self.output.get_shape().as_list()[1:]
                tf.summary.histogram("local_response_normalization", self.output)
        return self.output


def get_initializer_by_name(initializer_name, stddev=0.1):
    if initializer_name == "zeros":
        return tf.zeros_initializer()
    elif initializer_name == "normal":
        return tf.random_normal_initializer(stddev=stddev)
    elif initializer_name == "xavier":
        return tf.contrib.layers.xavier_initializer()
