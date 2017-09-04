import tensorflow as tf
from simple_network.layers.layers import Layer


class ReluLayer(Layer):

    def __init__(self, name='relu', summaries=True):
        # Define layer properties
        self.layer_type = "ReluLayer"
        self.layer_input = None
        self.input_shape = None
        self.output_shape = None
        self.output = None
        self.layer_name = name
        self.layer_size = None
        self.save_summaries = summaries

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        self.layer_size = self.input_shape
        with tf.name_scope(self.layer_name):
            self.output = tf.nn.relu(self.layer_input, name="relu_activation")
            self.output_shape = self.output.get_shape().as_list()[1:]
            tf.summary.histogram("relu_activation", self.output)
        return self.output


class LeakyReluLayer(Layer):

    def __init__(self, alpha=0.1, name='leaky_relu', summaries=True):
        # Define layer properties
        self.layer_type = "LeakyReluLayer"
        self.layer_input = None
        self.input_shape = None
        self.output_shape = None
        self.output = None
        self.layer_name = name
        self.layer_size = None
        self.alpha = alpha
        self.save_summaries = summaries

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        self.layer_size = self.input_shape
        with tf.name_scope(self.layer_name):
            self.output = tf.maximum(self.layer_input, self.alpha * self.layer_input, name="leaky_relu_activation")
            self.output_shape = self.output.get_shape().as_list()[1:]
            tf.summary.histogram("leaky_relu_activation", self.output)
        return self.output


class SoftmaxLayer(Layer):

    def __init__(self, name='softmax', summaries=True):
        # Define layer properties
        self.layer_type = "SoftmaxLayer"
        self.layer_input = None
        self.input_shape = None
        self.output_shape = None
        self.output = None
        self.layer_name = name
        self.layer_size = None
        self.save_summaries = summaries

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        self.layer_size = self.input_shape
        with tf.name_scope(self.layer_name):
            self.output = tf.nn.softmax(self.layer_input, name="softmax_activation")
            self.output_shape = self.output.get_shape().as_list()[1:]
            tf.summary.histogram("softmax_activation", self.output)
        return self.output
