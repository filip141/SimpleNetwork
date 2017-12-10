import copy
import logging
import tensorflow as tf
from simple_network.layers.layers import Layer
from simple_network.layers import ConvolutionalLayer

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
        self.residual_layers = []
        self.residual_layers_outputs = []
        self.node_layers_outputs = []
        self.last_activation_block = None
        self.layer_name = name

    def add_act_layer(self, layer):
        self.last_activation_block = layer

    def add(self, layer):
        self.node_layers.append(layer)

    def add_residual(self, layer):
        self.residual_layers.append(layer)

    def nodes_num(self):
        return len(self.node_layers)

    def residual_layers_num(self):
        return len(self.residual_layers)

    @staticmethod
    def build_node(node, layer_input, layer_idx, is_training):
        # Define number of layers for node
        node.n_layers = node.nodes_num() * node.ntimes

        # Build node
        with tf.name_scope(node.layer_name):
            for t_idx in range(0, node.ntimes):
                block_in = layer_input
                # Build layers for residual node
                with tf.name_scope("residual_node_layers_{}".format(t_idx)):
                    for l_idx, layer in enumerate(node.node_layers):
                        n_layer = copy.deepcopy(layer)
                        l_full_id = t_idx * node.nodes_num() + l_idx + 1
                        n_layer.layer_name += "_{}".format(l_full_id)
                        l_out = Layer.build_layer(n_layer, layer_input, "{}_{}".format(layer_idx, l_full_id),
                                                  is_training, enable_log=False)
                        layer_input = l_out
                        node.node_layers_outputs.append(l_out)
                        logger_string = "Residual Node {} | {} layer| Input shape: {}, Output shape: {}"\
                            .format(node.layer_name, n_layer.layer_type, n_layer.input_shape,n_layer.output_shape)
                        if isinstance(n_layer, ConvolutionalLayer):
                            logger_string += ", Kernel Size: {}".format(n_layer.layer_size)
                        logger.info(logger_string)

                # Add additional layers to residual connection
                layer_shortcut_input = block_in
                with tf.name_scope("residual_shortcut_layers_{}".format(t_idx)):
                    for lr_idx, r_layer in enumerate(node.residual_layers):
                        n_r_layer = copy.deepcopy(r_layer)
                        lr_full_id = t_idx * node.residual_layers_num() + lr_idx + 1
                        n_r_layer.layer_name += "_{}".format(lr_full_id)
                        lr_out = Layer.build_layer(n_r_layer, layer_shortcut_input,
                                                   "{}_{}_short".format(layer_idx, lr_full_id),
                                                   is_training, enable_log=False)
                        layer_shortcut_input = lr_out
                        node.residual_layers_outputs.append(lr_out)
                        logger_string = "Residual Node {} | {} shortcut-layer| Input shape: {}, Output shape: {}"\
                            .format(node.layer_name, n_r_layer.layer_type, n_r_layer.input_shape,
                                    n_r_layer.output_shape)
                        if isinstance(n_r_layer, ConvolutionalLayer):
                            logger_string += ", Kernel Size: {}".format(n_r_layer.layer_size)
                        logger.info(logger_string)

                node.node_layers_outputs[-1] += layer_shortcut_input
                layer_input += layer_shortcut_input
                # if last activation layer defined
                if node.last_activation_block is not None:
                    layer_input = Layer.build_layer(node.last_activation_block, layer_input,
                                                    "{}_final_act_{}".format(layer_idx, t_idx),
                                                    is_training, enable_log=False)
                    node.node_layers_outputs.append(layer_input)
        return layer_input
