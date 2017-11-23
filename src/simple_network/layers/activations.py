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
        with tf.variable_scope(self.layer_name, reuse=self.reuse):
            self.output = tf.nn.relu(self.layer_input, name="relu_activation")
            self.output_shape = self.output.get_shape().as_list()[1:]
            if self.save_summaries:
                tf.summary.histogram("relu_activation", self.output)
        return self.output


class SwishLayer(Layer):

    def __init__(self, name='swish', summaries=True, reuse=None):
        super(SwishLayer, self).__init__("SwishLayer", name, 'swish', summaries, reuse)
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
            self.output = self.layer_input * tf.nn.sigmoid(self.layer_input)
            self.output_shape = self.output.get_shape().as_list()[1:]
            tf.summary.histogram("swish_activation", self.output)
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


class TanhLayer(Layer):

    def __init__(self, name='tanh', summaries=True, reuse=None):
        super(TanhLayer, self).__init__("TanhLayer", name, 'tanh', summaries, reuse)
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
            self.output = tf.nn.tanh(self.layer_input, name="tanh_activation")
            self.output_shape = self.output.get_shape().as_list()[1:]
            tf.summary.histogram("tanh_activation", self.output)
        return self.output


class SigmoidLayer(Layer):

    def __init__(self, name='sigmoid', summaries=True, reuse=None):
        super(SigmoidLayer, self).__init__("SigmoidLayer", name, 'sigmoid', summaries, reuse)
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
            self.output = tf.nn.sigmoid(self.layer_input, name="sigmoid_activation")
            self.output_shape = self.output.get_shape().as_list()[1:]
            tf.summary.histogram("sigmoid_activation", self.output)
        return self.output


class LinearLayer(Layer):

    def __init__(self, name='linear', summaries=True, reuse=None):
        super(LinearLayer, self).__init__("LinearLayer", name, 'linear', summaries, reuse)
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
            self.output = self.layer_input
            self.output_shape = self.output.get_shape().as_list()[1:]
            tf.summary.histogram("linear_activation", self.output)
        return self.output
