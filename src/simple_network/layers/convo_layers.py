import numpy as np
import tensorflow as tf
from simple_network.tools.utils import variable_summaries
from simple_network.layers.layers import Layer, get_initializer_by_name


class ConvolutionalLayer(Layer):

    def __init__(self, l_size, stddev=0.1, activation='linear', stride=1, padding='same', initializer="xavier",
                 summaries=True, name='convo_layer', reuse=None):
        super(ConvolutionalLayer, self).__init__("Convolutional", name, 'convo_layer', summaries)
        # Define variables
        self.weights = None
        self.bias = None
        self.not_activated = None
        self.activated_output = None

        # Define layer properties
        self.layer_input = None
        self.input_shape = None
        self.output_shape = None
        self.layer_size = l_size

        # initializer
        self.stddev = stddev
        self.initializer = initializer

        # Boarders
        self.stride = stride
        self.padding = padding.upper()
        self.activation = activation
        self.reuse = reuse

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        input_shape_filters = self.input_shape[-1]
        with tf.variable_scope(self.layer_name, reuse=self.reuse):
            # Define weights and biases
            self.weights = tf.get_variable("weights", [self.layer_size[0], self.layer_size[1],
                                                       input_shape_filters, self.layer_size[2]],
                                           initializer=get_initializer_by_name(self.initializer, stddev=self.stddev))
            self.bias = tf.get_variable("biases", [self.layer_size[2]], initializer=tf.constant_initializer(0.1))
            self.not_activated = tf.nn.conv2d(self.layer_input, self.weights,
                                              strides=[1, self.stride, self.stride, 1],
                                              padding=self.padding) + self.bias
            if self.activation != 'linear':
                self.activated_output = getattr(tf.nn, self.activation)(self.not_activated)
            else:
                self.activated_output = self.not_activated
            #  Get histograms
            self.output_shape = self.activated_output.get_shape().as_list()[1:]
            if self.save_summaries:
                variable_summaries(self.weights, "weights")
                variable_summaries(self.bias, "biases")
                tf.summary.image('convo_output',
                                 tf.reshape(self.activated_output[0], self.output_shape[::-1] + [1]),
                                 self.output_shape[-1])
                tf.summary.histogram("activations", self.activated_output)
            return self.activated_output


class MaxPoolingLayer(Layer):

    def __init__(self, pool_size, stride=1, padding='same', name='max_pooling', summaries=True):
        super(MaxPoolingLayer, self).__init__("MaxPooling", name, 'max_pooling', summaries)
        # Define variables
        self.weights = None
        self.bias = None
        self.not_activated = None
        self.output = None

        # Define layer properties
        self.layer_input = None
        self.input_shape = None
        self.output_shape = None
        self.layer_size = pool_size

        self.stride = stride
        self.padding = padding.upper()

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        with tf.variable_scope(self.layer_name):
            k_size = [1, ] + self.layer_size + [1, ]
            self.output = tf.nn.max_pool(self.layer_input, ksize=k_size, strides=[1, self.stride, self.stride, 1],
                                         padding=self.padding)
            #  Get histograms
            self.output_shape = self.output.get_shape().as_list()[1:]
            if self.save_summaries:
                tf.summary.image('max_pool_output',
                                 tf.reshape(self.output[0], self.output_shape[::-1] + [1]), self.output_shape[-1])
                tf.summary.histogram("max_pooling_histogram", self.output)
            return self.output


class Flatten(Layer):

    def __init__(self, name='flatten'):
        super(Flatten, self).__init__("Flatten", name, 'flatten', False)
        # Define layer properties
        self.layer_type = "Flatten"
        self.layer_input = None
        self.input_shape = None
        self.output_shape = None
        self.output = None

        # Names
        self.default_name = 'flatten'
        self.layer_name = name
        self.layer_size = None

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        self.layer_size = self.input_shape
        with tf.variable_scope(self.layer_name):
            self.output = tf.reshape(self.layer_input, [-1, np.prod(self.input_shape)])
        return self.output
