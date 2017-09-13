import tensorflow as tf
from simple_network.layers.layers import Layer


class ReluLayer(Layer):

    def __init__(self, name='relu', summaries=True, reuse=None):
        super(ReluLayer, self).__init__("ReluLayer", name, 'relu', summaries, reuse)
        # Define layer properties
        self.layer_input = None
        self.input_shape = None
        self.output_shape = None
        self.output = None
        self.layer_size = None

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        self.layer_size = self.input_shape
        with tf.variable_scope(self.layer_name):
            self.output = tf.nn.relu(self.layer_input, name="relu_activation")
            self.output_shape = self.output.get_shape().as_list()[1:]
            tf.summary.histogram("relu_activation", self.output)
        return self.output


class LeakyReluLayer(Layer):

    def __init__(self, alpha=0.1, name='leaky_relu', summaries=True, reuse=None):
        super(LeakyReluLayer, self).__init__("LeakyReluLayer", name, 'leaky_relu', summaries, reuse)
        # Define layer properties
        self.layer_input = None
        self.input_shape = None
        self.output_shape = None
        self.output = None
        self.layer_size = None
        self.alpha = alpha

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        self.layer_size = self.input_shape
        with tf.variable_scope(self.layer_name):
            self.output = tf.maximum(self.layer_input, self.alpha * self.layer_input, name="leaky_relu_activation")
            self.output_shape = self.output.get_shape().as_list()[1:]
            tf.summary.histogram("leaky_relu_activation", self.output)
        return self.output


class SoftmaxLayer(Layer):

    def __init__(self, name='softmax', summaries=True, reuse=None):
        super(SoftmaxLayer, self).__init__("SoftmaxLayer", name, 'softmax', summaries, reuse)
        # Define layer properties
        self.layer_input = None
        self.input_shape = None
        self.output_shape = None
        self.output = None
        self.layer_size = None

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        self.layer_size = self.input_shape
        with tf.variable_scope(self.layer_name):
            self.output = tf.nn.softmax(self.layer_input, name="softmax_activation")
            self.output_shape = self.output.get_shape().as_list()[1:]
            tf.summary.histogram("softmax_activation", self.output)
        return self.output
