import numpy as np
import tensorflow as tf
from simple_network.tools.utils import variable_summaries
from simple_network.layers.layers import Layer, get_initializer_by_name


class ConvolutionalLayer(Layer):

    def __init__(self, l_size, stddev=0.1, activation='linear', stride=1, padding='same', initializer="xavier",
                 use_bias=True, summaries=True, name='convo_layer', reuse=None):
        super(ConvolutionalLayer, self).__init__("Convolutional", name, 'convo_layer', summaries, reuse)
        # Define variables
        self.use_bias = use_bias
        self.weights = None
        self.bias = None
        self.not_activated = None
        self.activated_output = None

        # Define layer properties
        self.layer_input = None
        self.layer_size = l_size

        # initializer
        self.stddev = stddev
        self.initializer = initializer

        # Boarders
        self.stride = stride
        self.padding = padding.upper()
        self.activation = activation

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        input_shape_filters = self.input_shape[-1]
        with tf.variable_scope(self.layer_name, reuse=self.reuse):
            # Define weights
            self.weights = tf.get_variable("weights", [self.layer_size[0], self.layer_size[1],
                                                       input_shape_filters, self.layer_size[2]],
                                           initializer=get_initializer_by_name(self.initializer, stddev=self.stddev),
                                           dtype=tf.float32, trainable=True)
            self.not_activated = tf.nn.conv2d(self.layer_input, self.weights,
                                              strides=[1, self.stride, self.stride, 1],
                                              padding=self.padding)

            # Define biases
            if self.use_bias:
                self.bias = tf.get_variable("biases", [self.layer_size[2]], initializer=tf.constant_initializer(0.1),
                                            dtype=tf.float32, trainable=True)
            else:
                self.bias = tf.zeros([self.layer_size[2]])

            self.not_activated = tf.nn.bias_add(self.not_activated, self.bias)

            # Use activation
            if self.activation != 'linear':
                self.activated_output = getattr(tf.nn, self.activation)(self.not_activated)
            else:
                self.activated_output = self.not_activated
            #  Get histograms
            self.output_shape = self.activated_output.get_shape().as_list()[1:]
            if self.save_summaries:
                variable_summaries(self.weights, "weights")
                variable_summaries(self.bias, "biases")
                tf.summary.histogram("activations", self.activated_output)
            return self.activated_output


class Convolutional3DLayer(Layer):

    def __init__(self, l_size, stddev=0.1, activation='linear', stride=1, stride_d=1, padding='same', initializer="xavier",
                 use_bias=True, summaries=True, name='convo3d_layer', reuse=None):
        super(Convolutional3DLayer, self).__init__("Convolutional3DLayer", name, 'convo3d_layer', summaries, reuse)
        # Define variables
        self.use_bias = use_bias
        self.weights = None
        self.bias = None
        self.not_activated = None
        self.activated_output = None

        # Define layer properties
        self.layer_input = None
        self.layer_size = l_size

        # initializer
        self.stddev = stddev
        self.initializer = initializer

        # Boarders
        self.stride_w = stride
        self.stride_d = stride_d
        self.padding = padding.upper()
        self.activation = activation

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        input_shape_filters = self.input_shape[-1]
        with tf.variable_scope(self.layer_name, reuse=self.reuse):
            # Define weights
            self.weights = tf.get_variable("weights", [self.layer_size[0], self.layer_size[1], self.layer_size[2],
                                                       input_shape_filters, self.layer_size[3]],
                                           initializer=get_initializer_by_name(self.initializer, stddev=self.stddev),
                                           dtype=tf.float32, trainable=True)
            self.not_activated = tf.nn.conv3d(self.layer_input, self.weights,
                                              strides=[1, self.stride_w, self.stride_w, self.stride_d, 1],
                                              padding=self.padding)

            # Define biases
            if self.use_bias:
                self.bias = tf.get_variable("biases", [self.layer_size[3]], initializer=tf.constant_initializer(0.1),
                                            dtype=tf.float32, trainable=True)
            else:
                self.bias = tf.zeros([self.layer_size[2]])

            self.not_activated = tf.nn.bias_add(self.not_activated, self.bias)

            # Use activation
            if self.activation != 'linear':
                self.activated_output = getattr(tf.nn, self.activation)(self.not_activated)
            else:
                self.activated_output = self.not_activated
            #  Get histograms
            self.output_shape = self.activated_output.get_shape().as_list()[1:]
            if self.save_summaries:
                variable_summaries(self.weights, "weights")
                variable_summaries(self.bias, "biases")
                tf.summary.histogram("activations", self.activated_output)
            return self.activated_output


class DeconvolutionLayer(Layer):

    def __init__(self, l_size, stddev=0.1, activation='linear', stride=1, batch_size=None, padding='same',
                 use_bias=True, initializer="xavier", output_shape=None, summaries=True,
                 name='deconv_layer', reuse=None):
        super(DeconvolutionLayer, self).__init__("DeconvolutionLayer", name, 'deconv_layer', summaries, reuse)
        # Define variables
        self.use_bias = use_bias
        self.weights = None
        self.bias = None
        self.not_activated = None
        self.activated_output = None

        # Define layer properties
        self.batch_size = batch_size
        self.layer_input = None
        self.input_shape = None
        self.layer_size = l_size
        self.output_shape = output_shape

        # initializer
        self.stddev = stddev
        self.initializer = initializer

        # Boarders
        self.stride = stride
        self.padding = padding.upper()
        self.activation = activation

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        input_shape_filters = self.input_shape[-1]
        with tf.variable_scope(self.layer_name, reuse=self.reuse):
            # Define weights
            self.weights = tf.get_variable("weights", [self.layer_size[0], self.layer_size[1],
                                                       self.layer_size[2], input_shape_filters],
                                           initializer=get_initializer_by_name(self.initializer, stddev=self.stddev),
                                           dtype=tf.float32, trainable=True)
            # Define biases
            if self.use_bias:
                self.bias = tf.get_variable("biases", [self.layer_size[2]], initializer=tf.constant_initializer(0.1),
                                            dtype=tf.float32, trainable=True)
            else:
                self.bias = tf.zeros([self.layer_size[2]])

            dyn_input_shape = tf.shape(self.layer_input)
            if not isinstance(self.batch_size, int):
                self.batch_size = dyn_input_shape[0]
            tf_output = tf.stack([self.batch_size, self.output_shape[0], self.output_shape[1], self.output_shape[2]])
            self.not_activated = tf.nn.conv2d_transpose(self.layer_input, self.weights, tf_output,
                                                        strides=[1, self.stride, self.stride, 1],
                                                        padding=self.padding)
            self.not_activated = tf.nn.bias_add(self.not_activated, self.bias)
            if self.activation != 'linear':
                self.activated_output = getattr(tf.nn, self.activation)(self.not_activated)
            else:
                self.activated_output = self.not_activated
            #  Get histograms
            if self.save_summaries:
                variable_summaries(self.weights, "weights")
                variable_summaries(self.bias, "biases")
                tf.summary.histogram("activations", self.activated_output)
            return self.activated_output


class Deconvolution3DLayer(Layer):

    def __init__(self, l_size, stddev=0.1, activation='linear', stride=1, stride_d=1, batch_size=None,
                 padding='same', use_bias=True, initializer="xavier", output_shape=None, summaries=True,
                 name='deconv3d_layer', reuse=None):
        super(Deconvolution3DLayer, self).__init__("Deconvolution3DLayer", name, 'deconv3d_layer', summaries, reuse)
        # Define variables
        self.use_bias = use_bias
        self.weights = None
        self.bias = None
        self.not_activated = None
        self.activated_output = None

        # Define layer properties
        self.batch_size = batch_size
        self.layer_input = None
        self.input_shape = None
        self.layer_size = l_size
        self.output_shape = output_shape

        # initializer
        self.stddev = stddev
        self.initializer = initializer

        # Boarders
        self.stride_w = stride
        self.stride_d = stride_d
        self.padding = padding.upper()
        self.activation = activation

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        input_shape_filters = self.input_shape[-1]
        with tf.variable_scope(self.layer_name, reuse=self.reuse):
            # Define weights
            self.weights = tf.get_variable("weights", [self.layer_size[0], self.layer_size[1],
                                                       self.layer_size[2], self.layer_size[3], input_shape_filters],
                                           initializer=get_initializer_by_name(self.initializer, stddev=self.stddev),
                                           dtype=tf.float32, trainable=True)
            # Define biases
            if self.use_bias:
                self.bias = tf.get_variable("biases", [self.layer_size[3]], initializer=tf.constant_initializer(0.1),
                                            dtype=tf.float32, trainable=True)
            else:
                self.bias = tf.zeros([self.layer_size[2]])

            dyn_input_shape = tf.shape(self.layer_input)
            if not isinstance(self.batch_size, int):
                self.batch_size = dyn_input_shape[0]
            tf_output = tf.stack([self.batch_size, self.output_shape[0], self.output_shape[1], self.output_shape[2],
                                  self.output_shape[3]])
            self.not_activated = tf.nn.conv3d_transpose(self.layer_input, self.weights, tf_output,
                                                        strides=[1, self.stride_w, self.stride_w, self.stride_d, 1],
                                                        padding=self.padding)
            self.not_activated = tf.nn.bias_add(self.not_activated, self.bias)
            if self.activation != 'linear':
                self.activated_output = getattr(tf.nn, self.activation)(self.not_activated)
            else:
                self.activated_output = self.not_activated
            #  Get histograms
            if self.save_summaries:
                variable_summaries(self.weights, "weights")
                variable_summaries(self.bias, "biases")
                tf.summary.histogram("activations", self.activated_output)
            return self.activated_output


class MaxPoolingLayer(Layer):

    def __init__(self, pool_size, stride=1, padding='same', name='max_pooling', summaries=False, reuse=None):
        super(MaxPoolingLayer, self).__init__("MaxPooling", name, 'max_pooling', summaries, reuse)
        # Define variables
        self.weights = None
        self.bias = None
        self.not_activated = None
        self.output = None

        # Define layer properties
        self.layer_input = None
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


class MaxPooling3DLayer(Layer):

    def __init__(self, pool_size, stride=1, padding='same', name='max_pooling3d', summaries=False, reuse=None):
        super(MaxPooling3DLayer, self).__init__("MaxPooling3DLayer", name, 'max_pooling3d', summaries, reuse)
        # Define variables
        self.weights = None
        self.bias = None
        self.not_activated = None
        self.output = None

        # Define layer properties
        self.layer_input = None
        self.layer_size = pool_size

        self.stride = stride
        self.padding = padding.upper()

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        with tf.variable_scope(self.layer_name):
            k_size = [1, ] + self.layer_size + [1, ]
            self.output = tf.nn.max_pool3d(self.layer_input, ksize=k_size,
                                           strides=[1, self.stride, self.stride, self.stride, 1],
                                           padding=self.padding)
            #  Get histograms
            self.output_shape = self.output.get_shape().as_list()[1:]
            if self.save_summaries:
                tf.summary.histogram("max_pooling_histogram", self.output)
            return self.output


class GlobalAveragePoolingLayer(Layer):

    def __init__(self, name='global_average_pooling', summaries=True, reuse=None):
        super(GlobalAveragePoolingLayer, self).__init__("GlobalAveragePoolingLayer", name,
                                                        'global_average_pooling', summaries, reuse)
        # Define variables
        self.weights = None
        self.bias = None
        self.not_activated = None
        self.output = None

        # Define layer properties
        self.layer_input = None

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        with tf.variable_scope(self.layer_name):
            self.output = tf.reduce_mean(self.layer_input, [1, 2])
            #  Get histograms
            self.output_shape = self.output.get_shape().as_list()[1:]
            if self.save_summaries:
                tf.summary.histogram("global_average_pooling_histogram", self.output)
            return self.output


class Flatten(Layer):

    def __init__(self, name='flatten', reuse=None):
        super(Flatten, self).__init__("Flatten", name, 'flatten', False, reuse)
        # Define layer properties
        self.layer_type = "Flatten"
        self.layer_input = None
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
