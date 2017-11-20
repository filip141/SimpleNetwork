import copy
import logging
import tensorflow as tf
from simple_network.layers.layers import Layer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkNode(object):
    """
        Network node makes available to define multiple streams in neural network.
        By adding network to node we are defining independent network streams.
    """

    def __init__(self, name="node_layer", reduce_output=None):
        if reduce_output is None:
            reduce_output = ""
        self.layer_size = None
        self.node_layers = []
        self.layer_name = name
        self.reduce_output = reduce_output

    def add(self, layer):
        self.node_layers.append(layer)

    def nodes_num(self):
        return len(self.node_layers)

    def get_output_info(self):
        return self.reduce_output.lower()

    def add_many(self, layer, ntimes=1, reuse_weights=True):
        f_layer = copy.deepcopy(layer)
        f_layer.reuse = False
        self.add(f_layer)
        for _ in range(1, ntimes):
            n_layer = copy.deepcopy(layer)
            n_layer.reuse = reuse_weights
            self.add(n_layer)

    @staticmethod
    def build_node(node, layer_input, layer_idx, is_training):
        n_layers = node.nodes_num()
        outputs_list = []
        reduce_output = node.get_output_info()
        with tf.name_scope(node.layer_name):
            if not isinstance(layer_input, list):
                layer_input = [layer_input for _ in range(0, n_layers)]
            for l_in, l_ll, l_idx in zip(layer_input, node.node_layers, range(n_layers)):
                # Build layer
                l_out = Layer.build_layer(l_ll, l_in, "{}_{}".format(layer_idx, l_idx), is_training, enable_log=False)
                # Save layer size as node size and add output to list
                node.layer_size = l_ll.layer_size
                outputs_list.append(l_out)
                logger.info("Node {} | {} layer| Input shape: {}, Output shape: {}"
                            .format(node.layer_name, l_ll.layer_type, l_ll.input_shape, l_ll.output_shape))
            if reduce_output == 'mean':
                output_val = tf.reduce_mean(tf.stack(outputs_list), axis=0)
            elif reduce_output == "concat":
                output_val = tf.concat(axis=3, values=outputs_list)
            else:
                output_val = outputs_list
        return output_val


class ResidualNode(object):

    def __init__(self, name="residual_node", ntimes=1):
        self.ntimes = ntimes
        self.n_layers = None
        self.node_layers = []
        self.node_layers_outputs = []
        self.layer_name = name

    def add(self, layer):
        self.node_layers.append(layer)

    def nodes_num(self):
        return len(self.node_layers)

    def get_layer(self):
        for _ in range(0, self.ntimes):
            for layer in self.node_layers:
                n_layer = copy.deepcopy(layer)
                yield n_layer

    @staticmethod
    def build_node(node, layer_input, layer_idx, is_training):
        first_layer_input = layer_input
        node.n_layers = node.nodes_num() * node.ntimes
        l_out = None
        with tf.name_scope(node.layer_name):
            for l_idx, layer in enumerate(node.get_layer()):
                layer.layer_name += "_{}".format(l_idx + 1)
                l_out = Layer.build_layer(layer, layer_input, "{}_{}".format(layer_idx, l_idx), is_training,
                                          enable_log=False)
                layer_input = l_out
                node.node_layers_outputs.append(l_out)
                logger.info("Residual Node {} | {} layer| Input shape: {}, Output shape: {}"
                            .format(node.layer_name, layer.layer_type, layer.input_shape, layer.output_shape))
        input_shape = first_layer_input.get_shape().as_list()
        output_shape = l_out.get_shape().as_list()
        if not input_shape == output_shape:
            raise ValueError("Input Shape and Output Shape does not match. Can't add both.")
        node.node_layers_outputs.append(l_out + first_layer_input)
        return l_out + first_layer_input
