import tensorflow as tf
from abc import ABCMeta, abstractmethod


class Layer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_graph(self, layer_input):
        pass


class DropoutLayer(Layer):

    def __init__(self, percent=0.2, name='dropout', summaries=True):
        # Define layer properties
        self.layer_type = "DropoutLayer"
        self.layer_input = None
        self.input_shape = None
        self.output_shape = None
        self.output = None
        self.layer_name = name
        self.layer_size = None
        self.percent = percent
        self.save_summaries = summaries

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        self.layer_size = self.input_shape
        with tf.name_scope(self.layer_name):
            self.output = tf.nn.dropout(self.layer_input, self.percent)
            self.output_shape = self.output.get_shape().as_list()[1:]
            tf.summary.histogram("dropout_output", self.output)
        return self.output


class BatchNormalizationLayer(Layer):

    def __init__(self, name='batch_normalization', summaries=True):
        # Define layer properties
        self.layer_type = "BatchNormalizationLayer"
        self.layer_input = None
        self.input_shape = None
        self.output_shape = None
        self.output = None
        self.layer_name = name
        self.layer_size = None
        self.is_training = None
        self.save_summaries = summaries

    def set_training_indicator(self, is_training):
        self.is_training = is_training

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        self.layer_size = self.input_shape
        with tf.name_scope(self.layer_name):
            self.output = tf.contrib.layers.batch_norm(self.layer_input, center=True, scale=True,
                                                       is_training=self.is_training)
            self.output_shape = self.output.get_shape().as_list()[1:]
            tf.summary.histogram("batch_normalization", self.output)
        return self.output


def get_initializer_by_name(initializer_name, l_size, stddev=0.1):
    if initializer_name == "zeros":
        return tf.zeros(l_size)
    elif initializer_name == "normal":
        return tf.random_normal(l_size, stddev=stddev)
    elif initializer_name == "xavier":
        xavier_init = tf.contrib.layers.xavier_initializer()
        return xavier_init(l_size)
