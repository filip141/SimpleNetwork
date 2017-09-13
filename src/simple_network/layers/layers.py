import tensorflow as tf
from abc import ABCMeta, abstractmethod


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
            self.output = tf.contrib.layers.batch_norm(self.layer_input, center=True, scale=True,
                                                       is_training=self.is_training, reuse=self.reuse,
                                                       scope="bn_{}".format(self.layer_name))
            if self.save_summaries:
                with tf.variable_scope("bn_{}".format(self.layer_name), reuse=True):
                    beta = tf.get_variable("beta", [self.layer_size[-1]])
                    gamma = tf.get_variable("gamma", [self.layer_size[-1]])
                    tf.summary.scalar('bn_beta', beta[0])
                    tf.summary.scalar('bn_gamma', gamma[0])
                self.output_shape = self.output.get_shape().as_list()[1:]
                tf.summary.histogram("batch_normalization", self.output)
        return self.output


def get_initializer_by_name(initializer_name, stddev=0.1):
    if initializer_name == "zeros":
        return tf.zeros_initializer()
    elif initializer_name == "normal":
        return tf.random_normal_initializer(stddev=stddev)
    elif initializer_name == "xavier":
        return tf.contrib.layers.xavier_initializer()
