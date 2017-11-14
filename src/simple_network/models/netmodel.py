import os
import copy
import time
import logging
import tempfile
import tensorflow as tf
from simple_network.tools.utils import ModelLogger
from simple_network.tools.utils import create_sprite_image
from simple_network.layers import BatchNormalizationLayer, DropoutLayer
from simple_network.metrics import cross_entropy, accuracy, mean_square, mean_absolute, \
    mean_absolute_weighted_4, cross_entropy_sigmoid, binary_accuracy
from simple_network.train.losses import custom_loss
from simple_network.train.losses import cross_entropy as cross_entropy_loss
from simple_network.train.losses import mean_square as mean_square_loss
from simple_network.train.losses import mean_absolute as mean_absolute_loss
from simple_network.train.losses import mean_absolute_weight as mean_absolute_weight_loss
from simple_network.train.optimizers import adam_optimizer, momentum_optimizer, rmsprop_optimizer, sgd_optimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkNode(object):

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

    def add_many(self, layer, ntimes=1):
        f_layer = copy.deepcopy(layer)
        f_layer.reuse = False
        self.add(f_layer)
        for _ in range(1, ntimes):
            n_layer = copy.deepcopy(layer)
            n_layer.reuse = True
            self.add(n_layer)


class SNModel(object):

    def __init__(self, input_size, summary_path=None, metric=None, input_summary=None, input_placeholder=None,
                 session=None):
        # If summary_path is None set tempdir
        if summary_path is None:
            self.summary_path = tempfile.gettempdir()
        if not os.path.isdir(summary_path):
            os.mkdir(summary_path)
        current_time_str = time.strftime("%Y%m%d%H%M%S")
        self.summary_path = os.path.join(summary_path, "logs_{}".format(current_time_str))
        self.model_info_path = os.path.join(summary_path, "info")
        if not os.path.isdir(self.summary_path):
            os.mkdir(self.summary_path)
        if not os.path.isdir(self.model_info_path):
            os.mkdir(self.model_info_path)

        if metric is None:
            metric = []

        self.layers = []
        self.metric = metric
        self.input_size = input_size
        self.input_summary = input_summary
        self.layer_outputs = []
        self.last_layer_prediction = None
        self.input_layer_placeholder = input_placeholder
        self.output_labels_placeholder = None
        self.is_training_placeholder = None
        self.optimizer = None
        self.optimizer_data = {}
        self.learning_rate = None
        self.loss = None
        self.loss_data = {}
        self.metric_list_func = []
        self.optimizer_func = None
        self.loss_func = None
        self.saver = None
        self.embedding_handler = None
        self.save_path = os.path.join(summary_path, "model")
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        if session is None:
            self.sess = tf.Session()
        else:
            self.sess = session
        self.train_writer = tf.summary.FileWriter(os.path.join(self.summary_path, "train"))
        self.test_writer = tf.summary.FileWriter(os.path.join(self.summary_path, "test"))

    def save_model_info(self):
        model_logger = ModelLogger(self.model_info_path)
        model_logger.title("Model Information")
        model_logger.add_property("Input layer shape", self.input_size)
        model_logger.add_to_json("input_shape", self.input_size)

        model_logger.add_property("Optimizer", self.optimizer)
        model_logger.add_to_json("optimizer", self.optimizer)

        model_logger.add_property("Learning Rate", self.learning_rate)
        model_logger.add_to_json("learning_rate", self.learning_rate)

        model_logger.add_property("Loss", self.loss)
        model_logger.add_to_json("loss", self.loss)

        model_logger.add_property("Metrics", self.metric)
        model_logger.add_to_json("metrics", self.metric)
        model_logger.title("Layers")
        layers_data = []
        for layer in self.layers:
            if isinstance(layer, NetworkNode):
                model_logger.info("Network node")
                node_layers = []
                for inside_layer in layer.node_layers:
                    node_layers.append({"name": inside_layer.layer_type, "input_shape": inside_layer.input_shape,
                                        "output_shape": inside_layer.output_shape})
                    model_logger.info("----{} layer| Input shape: {}, Output shape: {}"
                                      .format(inside_layer.layer_type, inside_layer.input_shape,
                                              inside_layer.output_shape))
                layers_data.append({"name": "network_node", "node_layers": node_layers})
            else:
                layers_data.append({"name": layer.layer_type, "input_shape": layer.input_shape,
                                    "output_shape": layer.output_shape})
                model_logger.info("{} layer| Input shape: {}, Output shape: {}".format(layer.layer_type,
                                                                                       layer.input_shape,
                                                                                       layer.output_shape))
        model_logger.add_to_json("layers", layers_data)
        model_logger.save()

    def save(self, global_step=None):
        self.saver.save(self.sess, os.path.join(self.save_path, "tensorflow_model"),
                        global_step=global_step)

    def restore(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_path))

    def add(self, new_layer):
        self.layers.append(new_layer)

    def build_layer(self, layer, layer_input, layer_idx, enable_log=True):
        if isinstance(layer, BatchNormalizationLayer):
            layer.set_training_indicator(self.is_training_placeholder)
        elif isinstance(layer, DropoutLayer):
            layer.set_training_indicator(self.is_training_placeholder)
        # Add number if default
        if layer.layer_name == layer.default_name:
            layer.layer_name = "{}_{}".format(layer.layer_name, layer_idx)
        layer_output = layer.build_graph(layer_input)
        if enable_log:
            logger.info("{} layer| Input shape: {}, Output shape: {}".format(layer.layer_type, layer.input_shape,
                                                                             layer.output_shape))
        return layer_output

    def set_optimizer(self, optimizer, **kwargs):
        self.optimizer = optimizer
        self.optimizer_data = kwargs

    def set_loss(self, loss, **kwargs):
        self.loss = loss
        self.loss_data = kwargs

    def get_loss_by_name(self):
        # Check loss
        if self.loss is None:
            return None
        if self.loss == "cross_entropy":
            return cross_entropy_loss
        if self.loss == "mse":
            return mean_square_loss
        if self.loss == "mae":
            return mean_absolute_loss
        if self.loss == "mae_weight":
            return mean_absolute_weight_loss
        if self.loss == "custom_loss":
            return custom_loss
        else:
            raise AttributeError("Loss {} not defined.".format(self.loss))

    def get_optimizer_by_name(self):
        # Check optimizer
        if self.optimizer is None:
            return None
        if self.optimizer == "Adam":
            return adam_optimizer
        elif self.optimizer == "Momentum":
            return momentum_optimizer
        elif self.optimizer == "RMSprop":
            return rmsprop_optimizer
        elif self.optimizer == "SGD":
            return sgd_optimizer
        else:
            raise AttributeError("Optimizer {} not defined.".format(self.optimizer))

    def get_metric_by_name(self):
        metric_list = []
        for metric_item in self.metric:
            if metric_item == "accuracy":
                metric_list.append(accuracy)
            elif metric_item == "binary_accuracy":
                metric_list.append(binary_accuracy)
            elif metric_item == "cross_entropy":
                metric_list.append(cross_entropy)
            elif metric_item == "cross_entropy_sigmoid":
                metric_list.append(cross_entropy_sigmoid)
            elif metric_item == "mse":
                metric_list.append(mean_square)
            elif metric_item == "mae":
                metric_list.append(mean_absolute)
            elif metric_item == "mae_weighted_4":
                metric_list.append(mean_absolute_weighted_4)
            else:
                raise AttributeError("Metric {} not defined.".format(metric_item))
        return metric_list

    def prepare_input(self):
        model_size = [None, ] + self.input_size
        if self.input_layer_placeholder is None:
            self.input_layer_placeholder = tf.placeholder(tf.float32, model_size, name='x')
        # Save input summary
        if isinstance(self.input_summary, dict):
            summary_input_data = self.input_layer_placeholder
            reshape_b = self.input_summary.get('reshape', False)
            if reshape_b:
                reshape_size = self.input_summary.get('reshape_size', None)
                if reshape_size is None:
                    raise ValueError("Reshape size should be specified with reshape flag.")
                reshape_size = [-1] + reshape_size
                summary_input_data = tf.reshape(summary_input_data, reshape_size)
            number_of_img = self.input_summary.get('img_number', 1)
            tf.summary.image('input', summary_input_data, number_of_img)

    def get_last_layer_prediction(self):
        return self.last_layer_prediction

    def add_embedding_monitor(self, em_iterator, em_num, embedding_input, img_res, log_dir):
        log_path = os.path.join(log_dir, "train")
        img_path, meta_path, images, labels = create_sprite_image(em_iterator, log_path, img_num=em_num)
        layer_size = embedding_input.get_shape().as_list()[-1]
        embedding = tf.Variable(tf.zeros([em_num, layer_size]), name="test_embedding")
        self.embedding_handler = embedding.assign(embedding_input)

        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = embedding.name
        embedding_config.sprite.image_path = img_path
        embedding_config.metadata_path = meta_path
        embedding_config.sprite.single_image_dim.extend(img_res)
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(self.train_writer, config)
        return images, labels

    def get_layer_by_name(self, layer_name):
        for layer in self.layers:
            if isinstance(layer, NetworkNode):
                for d_layer in layer.node_layers:
                    if d_layer.layer_name == layer_name:
                        return d_layer
            if layer.layer_name == layer_name:
                return layer
