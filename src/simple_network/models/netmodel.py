import os
import logging
import tempfile
import tensorflow as tf
from simple_network.layers import BatchNormalizationLayer, DropoutLayer
from simple_network.metrics import cross_entropy, accuracy, mean_square, mean_absolute
from simple_network.train.losses import cross_entropy as cross_entropy_loss
from simple_network.train.losses import mean_square as mean_square_loss
from simple_network.train.losses import mean_absolute as mean_absolute_loss
from simple_network.train.losses import mean_absolute_weight as mean_absolute_weight_loss
from simple_network.train.optimizers import adam_optimizer, momentum_optimizer, rmsprop_optimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SNModel(object):

    def __init__(self, input_size, summary_path=None, metric=None, input_summary=None):
        # If summary_path is None set tempdir
        if summary_path is None:
            summary_path = tempfile.gettempdir()
        if metric is None:
            metric = []

        self.layers = []
        self.metric = metric
        self.input_size = input_size
        self.input_summary = input_summary
        self.input_layer_placeholder = None
        self.output_labels_placeholder = None
        self.is_training_placeholder = None
        self.optimizer = None
        self.optimizer_data = {}
        self.loss = None
        self.loss_data = {}
        self.metric_list_func = []
        self.optimizer_func = None
        self.loss_func = None
        self.sess = tf.InteractiveSession()
        self.saver = None
        self.summary_path = summary_path
        self.writer = tf.summary.FileWriter(summary_path)

    def save(self, global_step=None):
        self.saver.save(self.sess, os.path.join(self.summary_path, "tensorflow_model"),
                        global_step=global_step)

    def restore(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.summary_path))

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
            raise AttributeError("Loss function not set. Look into simple_network/train/losses to "
                                 "find appropriate loss function.")
        if self.loss == "cross_entropy":
            return cross_entropy_loss
        if self.loss == "mse":
            return mean_square_loss
        if self.loss == "mae":
            return mean_absolute_loss
        if self.loss == "mae_weight":
            return mean_absolute_weight_loss
        else:
            raise AttributeError("Loss {} not defined.".format(self.loss))

    def get_optimizer_by_name(self):
        # Check optimizer
        if self.optimizer is None:
            raise AttributeError("Optimizer not set. Look into simple_network/train/losses to find appropriate o"
                                 "optimizer.")
        if self.optimizer == "Adam":
            return adam_optimizer
        elif self.optimizer == "Momentum":
            return momentum_optimizer
        elif self.optimizer == "RMSprop":
            return rmsprop_optimizer
        else:
            raise AttributeError("Optimizer {} not defined.".format(self.loss))

    def get_metric_by_name(self):
        metric_list = []
        for metric_item in self.metric:
            if metric_item == "accuracy":
                metric_list.append(accuracy)
            elif metric_item == "cross_entropy":
                metric_list.append(cross_entropy)
            elif metric_item == "mse":
                metric_list.append(mean_square)
            elif metric_item == "mae":
                metric_list.append(mean_absolute)
            else:
                raise AttributeError("Metric {} not defined.".format(self.loss))
        return metric_list

    def prepare_input(self):
        model_size = [None, ] + self.input_size
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