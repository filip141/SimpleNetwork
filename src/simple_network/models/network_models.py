import time
import logging
import tempfile
import tensorflow as tf
from simple_network.train.optimizers import adam_optimizer
from simple_network.metrics import cross_entropy, accuracy
from simple_network.train.losses import cross_entropy as cross_entropy_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkModel(object):

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
        self.__metric_list_func = []
        self.__optimizer_func = None
        self.__loss_func = None
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(summary_path)

    def add(self, new_layer):
        self.layers.append(new_layer)

    def set_optimizer(self, optimizer, **kwargs):
        self.optimizer = optimizer
        self.optimizer_data = kwargs

    def set_loss(self, loss):
        self.loss = loss

    def build_model(self, learning_rate):
        # Define input
        logger.info("-" * 90)
        model_size = [None, ] + self.input_size
        with tf.name_scope("input_layer"):
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
        logger.info("Input layer| shape: {}".format(self.input_size))

        # build model
        layer_input = self.input_layer_placeholder
        self.is_training_placeholder = tf.placeholder(tf.bool, name='is_training')
        for layer in self.layers:
            if isinstance(layer, BatchNormalizationLayer):
                layer.set_training_indicator(self.is_training_placeholder)
            layer_output = layer.build_graph(layer_input)
            layer_input = layer_output
            logger.info("{} layer| Input shape: {}, Output shape: {}".format(layer.layer_type, layer.input_shape,
                                                                             layer.output_shape))
        # Output placeholder
        self.output_labels_placeholder = tf.placeholder(tf.float32, [None, self.layers[-1].layer_size[-1]],
                                                        name='labels')
        y_labels = self.output_labels_placeholder

        # Define loss and optimizer
        self.__loss_func = self.get_loss_by_name()(logits=layer_output, labels=y_labels)
        logger.info("Loss function: {}".format(self.loss))
        self.__optimizer_func = self.get_optimizer_by_name()(self.__loss_func, learning_rate, self.optimizer_data)
        logger.info("Optimizer: {}".format(self.optimizer))

        # Initialize metrics
        metric_list_func = self.get_metric_by_name()
        for metric_item in metric_list_func:
            self.__metric_list_func.append(metric_item(logits=layer_output, labels=y_labels))
        logger.info("Metric: {}".format(self.metric))
        logger.info("-" * 90)

        # Initialize all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self, train_iter, test_iter, train_step=100, test_step=100, epochs=1000, sample_per_epoch=1000,
              learning_rate=0.01, summary_step=5, reshape_input=None):
        # Build model
        self.build_model(learning_rate=learning_rate)
        self.writer.add_graph(self.sess.graph)
        # Train
        start_time = time.time()
        merged_summary = tf.summary.merge_all()
        for epoch_idx in range(epochs):
            # samples in epoch
            logger.info("Training: Epoch number {}".format(epoch_idx))
            for sample_iter in range(sample_per_epoch):
                # Load Training batch
                batch_x, batch_y = train_iter.next_batch(train_step)
                # reshape train input if defined
                if reshape_input is not None:
                    batch_x = batch_x.reshape([train_step, ] + reshape_input)

                # Load Test batch
                test_batch_x, test_batch_y = test_iter.next_batch(test_step)
                # reshape test input if defined
                if reshape_input is not None:
                    test_batch_x = test_batch_x.reshape([train_step, ] + reshape_input)
                train_data = {self.input_layer_placeholder: batch_x, self.output_labels_placeholder: batch_y,
                              self.is_training_placeholder: True}

                # Train
                self.sess.run(self.__optimizer_func, feed_dict=train_data)
                if self.metric is not None:
                    train_metrics_result = self.sess.run(self.__metric_list_func, feed_dict=train_data)
                    train_metric_result_dict = dict(zip(self.metric, train_metrics_result))
                    train_info_str = "Training set metrics: {} | ".format(", ".join(
                        [": ".join((tm_k.title(), str(tm_v))) for tm_k, tm_v in train_metric_result_dict.items()]))

                    test_data = {self.input_layer_placeholder: test_batch_x, self.output_labels_placeholder: test_batch_y,
                                 self.is_training_placeholder: False}
                    test_metrics_result = self.sess.run(self.__metric_list_func, feed_dict=test_data)
                    test_metric_result_dict = dict(zip(self.metric, test_metrics_result))
                    test_info_str = "Test set metrics: {}".format(", ".join(
                        [": ".join((tm_k.title(), str(tm_v))) for tm_k, tm_v in test_metric_result_dict.items()]))
                    logger.info(train_info_str + test_info_str + " | Sample number: {} | Time: {}"
                                .format(sample_iter, time.time() - start_time))

                # Save summary
                if sample_iter % summary_step == 0:
                    sum_res = self.sess.run(merged_summary, train_data)
                    self.writer.add_summary(sum_res, epoch_idx * sample_per_epoch + sample_iter)

    def get_loss_by_name(self):
        # Check loss
        if self.loss is None:
            raise AttributeError("Loss function not set. Look into simple_network/train/losses to "
                                 "find appropriate loss function.")
        if self.loss == "cross_entropy":
            return cross_entropy_loss
        else:
            raise AttributeError("Loss {} not defined.".format(self.loss))

    def get_optimizer_by_name(self):
        # Check optimizer
        if self.optimizer is None:
            raise AttributeError("Optimizer not set. Look into simple_network/train/losses to find appropriate o"
                                 "optimizer.")
        if self.optimizer == "Adam":
            return adam_optimizer
        else:
            raise AttributeError("Optimizer {} not defined.".format(self.loss))

    def get_metric_by_name(self):
        metric_list = []
        for metric_item in self.metric:
            if metric_item == "accuracy":
                metric_list.append(accuracy)
            elif metric_item == "cross_entropy":
                metric_list.append(cross_entropy)
            else:
                raise AttributeError("Metric {} not defined.".format(self.loss))
        return metric_list

if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    import numpy as np
    from tensorflow.examples.tutorials.mnist import input_data
    import matplotlib.pyplot as plt
    import os
    from simple_network.layers import ConvolutionalLayer, MaxPoolingLayer, ReluLayer, \
        FullyConnectedLayer, Flatten, DropoutLayer, BatchNormalizationLayer, LeakyReluLayer
    from simple_network.tools.utils import CIFARDataset

    # Model params
    K = 200
    L = 300
    M = 60
    N = 30

    input_summary = {"img_number": 30}

    # Load mnist data
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    cifar_train = CIFARDataset(data_path="/home/phoenix/PycharmProjects/simple_network/datasets/train")
    cifar_test = CIFARDataset(data_path="/home/phoenix/PycharmProjects/simple_network/datasets/test")
    summary_path = "/home/phoenix/tensor_logs"
    files_in_dir = os.listdir(summary_path)
    for s_file in files_in_dir:
        os.remove(os.path.join(summary_path, s_file))
    nm = NetworkModel([32, 32, 3], metric=["accuracy", "cross_entropy"], input_summary=input_summary,
                      summary_path=summary_path)

    nm.add(ConvolutionalLayer([3, 3, 48], initializer="xavier", name='convo_layer_1_1'))
    nm.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_1_1"))
    nm.add(ConvolutionalLayer([3, 3, 48], initializer="xavier", name='convo_layer_1_2'))
    nm.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_1_2"))
    nm.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="same", name="pooling_1_1"))
    nm.add(DropoutLayer(percent=0.25))

    nm.add(ConvolutionalLayer([3, 3, 96], initializer="xavier", name='convo_layer_2_1'))
    nm.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_2_1"))
    nm.add(ConvolutionalLayer([3, 3, 96], initializer="xavier", name='convo_layer_2_2'))
    nm.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_2_2"))
    nm.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="same", name="pooling_2_1"))
    nm.add(DropoutLayer(percent=0.25))

    nm.add(ConvolutionalLayer([3, 3, 192], initializer="xavier", name='convo_layer_3_1'))
    nm.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_3_1"))
    nm.add(ConvolutionalLayer([3, 3, 192], initializer="xavier", name='convo_layer_3_2'))
    nm.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_3_2"))
    nm.add(MaxPoolingLayer(pool_size=[2, 2], stride=2, padding="same", name="pooling_3_1"))
    nm.add(DropoutLayer(percent=0.25))

    nm.add(Flatten(name='flatten_1'))

    nm.add(FullyConnectedLayer([3072, 512], initializer="xavier", name='fully_connected_4_1'))
    nm.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_4_1"))
    nm.add(DropoutLayer(percent=0.5))

    nm.add(FullyConnectedLayer([512, 256], initializer="xavier", name='fully_connected_5_1'))
    nm.add(LeakyReluLayer(alpha=0.1, name="leaky_relu_5_1"))
    nm.add(DropoutLayer(percent=0.5))

    nm.add(FullyConnectedLayer([256, 10], initializer="xavier", name='fully_connected_6_1'))
    nm.set_optimizer("Adam", beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    nm.set_loss("cross_entropy")
    nm.train(train_iter=cifar_train, train_step=128, test_iter=cifar_test, test_step=128,
             learning_rate=0.001, sample_per_epoch=391, epochs=100)