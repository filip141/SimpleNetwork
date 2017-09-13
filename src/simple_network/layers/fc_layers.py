import tensorflow as tf
from simple_network.tools.utils import variable_summaries
from simple_network.layers.layers import Layer, get_initializer_by_name


class FullyConnectedLayer(Layer):

    def __init__(self, l_size, stddev=0.1, activation='linear', name='fc_layer', initializer="xavier",
                 summaries=True, reuse=None):
        super(FullyConnectedLayer, self).__init__("FullyConnected", name, 'fc_layer', summaries, reuse)
        # Define variables
        self.weights = None
        self.bias = None
        self.not_activated = None
        self.activated_output = None

        # Define layer properties
        self.layer_input = None
        self.output_shape = None
        self.input_shape = None
        self.layer_size = l_size

        # initializer
        self.stddev = stddev
        self.initializer = initializer
        self.activation = activation

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        with tf.variable_scope(self.layer_name, reuse=self.reuse):
            # Define weights and biases
            self.weights = tf.get_variable("weights", [self.layer_size[0], self.layer_size[1]],
                                           initializer=get_initializer_by_name(self.initializer, stddev=self.stddev))
            self.bias = tf.get_variable("biases", [self.layer_size[1]], initializer=tf.zeros_initializer())
            self.not_activated = tf.matmul(self.layer_input, self.weights) + self.bias

            #  Get histograms
            if self.save_summaries:
                variable_summaries(self.weights, "weights")
                variable_summaries(self.bias, "biases")
            if self.activation != 'linear':
                self.activated_output = getattr(tf.nn, self.activation)(self.not_activated)
            else:
                self.activated_output = self.not_activated
            self.output_shape = self.activated_output.get_shape().as_list()[1:]
            tf.summary.histogram("activations", self.activated_output)
        return self.activated_output
