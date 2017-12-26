import os
import time
import logging
import numpy as np
import tensorflow as tf
from simple_network.layers.layers import Layer
from simple_network.models.netmodel import SNModel
from simple_network.models.additional_nodes import NetworkNode, ResidualNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkParallel(SNModel):

    def __init__(self, input_size, summary_path=None, metric=None, input_summary=None, input_placeholder=None,
                 session=None):
        # If summary_path is None set tempdir
        self.model_build = False
        super(NetworkParallel, self).__init__(input_size, summary_path, metric, input_summary,
                                              input_placeholder, session)
        handler = logging.FileHandler(os.path.join(self.summary_path, "log.log"))
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    def build_model(self, learning_rate=0.01, decay=None, decay_steps=100000, out_placeholder=True):
        # Define input
        logger.info("-" * 90)
        with tf.name_scope("input_layer"):
            self.prepare_input()
        logger.info("Input layer| shape: {}".format(self.input_size))

        # build model
        layer_output = None
        layer_input = self.input_layer_placeholder
        self.is_training_placeholder = tf.placeholder(tf.bool, name='is_training')
        for l_idx, layer in enumerate(self.layers):
            if isinstance(layer, Layer):
                # Define building function for layer
                layer_output = Layer.build_layer(layer, layer_input, l_idx, self.is_training_placeholder)
            elif isinstance(layer, NetworkNode):
                # Define build function for NetworkNode for multiple stream networks
                layer_output = NetworkNode.build_node(layer, layer_input, l_idx, self.is_training_placeholder)
            elif isinstance(layer, ResidualNode):
                # Define build function for NetworkNode for multiple stream networks
                layer_output = ResidualNode.build_node(layer, layer_input, l_idx, self.is_training_placeholder)
            else:
                raise AttributeError("Object not defined {}".format(type(layer)))
            self.layer_outputs.append(layer_output)
            layer_input = layer_output
        self.last_layer_prediction = layer_output
        # Output placeholder
        if out_placeholder:
            self.output_labels_placeholder = tf.placeholder(tf.float32, [None, self.layers[-1].layer_size[-1]],
                                                            name='labels')
            y_labels = self.output_labels_placeholder

        # Define loss
        loss_function = self.get_loss_by_name()
        if loss_function is not None:
            self.loss_func = loss_function(logits=layer_output, labels=y_labels, loss_data=self.loss_data)
            logger.info("Loss function: {}".format(self.loss))

        # Define decay
        self.learning_rate = learning_rate
        if decay is not None:
            global_step = tf.Variable(0, trainable=False)
            self.optimizer_data["global_step"] = global_step
            learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)

        # Define optimizer
        optimizer_function = self.get_optimizer_by_name()
        if optimizer_function is not None:
            self.optimizer_func = optimizer_function(self.loss_func, learning_rate, self.optimizer_data)
            logger.info("Optimizer: {}".format(self.optimizer))

        # Initialize metrics
        metric_list_func = self.get_metric_by_name()
        for metric_item in metric_list_func:
            self.metric_list_func.append(metric_item(logits=layer_output, labels=y_labels))
        logger.info("Metric: {}".format(self.metric))
        logger.info("-" * 90)

        # Initialize all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # Initialize saver for future saving weights
        self.model_build = True
        self.saver = tf.train.Saver()
        self.save_model_info()

    def moving_average(self, moving_avg, t_results, phase="Train"):
        moving_avg = [x_it[-9:] + [train_met] for x_it, train_met in zip(moving_avg, t_results)]
        moving_avg_mean = [np.mean(x_it_n) for x_it_n in moving_avg]
        train_metric_result_dict = dict(zip(self.metric, moving_avg_mean))
        train_info_str = "{} set metrics: {} | ".format(phase, ", ".join(
            [": ".join((tm_k.title(), str(tm_v))) for tm_k, tm_v in train_metric_result_dict.items()]))
        return moving_avg, train_info_str, moving_avg_mean

    def prepare_batch(self, batch_x, batch_y, reshape_input, is_train=True):
        batch_size = batch_x.shape[0]

        # reshape input if defined
        if reshape_input is not None:
            batch_x = batch_x.reshape([batch_size, ] + reshape_input)

        nn_data = {self.input_layer_placeholder: batch_x, self.output_labels_placeholder: batch_y,
                   self.is_training_placeholder: is_train}
        return nn_data

    def run_optimize(self, batch_x, batch_y, reshape_input):
        train_data = self.prepare_batch(batch_x, batch_y, reshape_input)
        self.sess.run(self.optimizer_func, feed_dict=train_data)

    def run_eval(self, batch_x, batch_y, reshape_input, is_train=True):
        eval_data = self.prepare_batch(batch_x, batch_y, reshape_input, is_train)
        eval_metrics_result = self.sess.run(self.metric_list_func, feed_dict=eval_data)
        return eval_metrics_result

    def summary_save(self, tr_bx, tr_by, ts_bx, ts_by, reshape_input, merged_sum, idx):
        train_data = self.prepare_batch(tr_bx, tr_by, reshape_input, True)
        test_data = self.prepare_batch(ts_bx, ts_by, reshape_input, False)
        sum_res_train = self.sess.run(merged_sum, train_data)
        sum_res_test = self.sess.run(merged_sum, test_data)
        self.train_writer.add_summary(sum_res_train, idx)
        self.test_writer.add_summary(sum_res_test, idx)

    def train(self, train_iter, test_iter, train_step=100, test_step=100, epochs=1000, sample_per_epoch=1000,
              summary_step=5, reshape_input=None, embedding_num=None, save_model=True, early_stop=None,
              early_stop_lower=False, test_update=10):
        # Check Build model
        if not self.model_build:
            raise AttributeError("Model should be build before training it.")
        if early_stop is None:
            early_stop = {}
        self.train_writer.add_graph(self.sess.graph)
        self.test_writer.add_graph(self.sess.graph)

        # Embeddings
        start_time = time.time()
        merged_summary = tf.summary.merge_all()
        embedd_img, embedd_labels = None, None
        if embedding_num is not None:
            embedd_img, embedd_labels = self.add_embedding_monitor(em_iterator=train_iter, em_num=embedding_num,
                                                                   embedding_input=self.layer_outputs[-5],
                                                                   img_res=self.input_size,
                                                                   log_dir=self.summary_path)

        test_metrics_result = 0
        moving_avg_train = [[] for _ in range(len(self.metric_list_func))]
        moving_avg_test = [[] for _ in range(len(self.metric_list_func))]
        for epoch_idx in range(epochs):

            # samples in epoch
            logger.info("Training: Epoch number {}".format(epoch_idx))
            for sample_iter in range(sample_per_epoch):
                # Load Training batch
                batch_x, batch_y = train_iter.next_batch(train_step)

                # Load Test batch
                test_batch_x, test_batch_y = test_iter.next_batch(test_step)

                # Train
                self.run_optimize(batch_x, batch_y, reshape_input)

                if self.metric is not None:
                    train_metrics_result = self.run_eval(batch_x, batch_y, reshape_input)
                    moving_avg_train, train_info_str, _ = self.moving_average(moving_avg_train, train_metrics_result,
                                                                              "Train")

                    # Update test statistics every n iterations
                    if sample_iter % test_update == 0:
                        test_metrics_result = self.run_eval(test_batch_x, test_batch_y, reshape_input, is_train=False)
                    moving_avg_test, test_info_str, moving_avg_m = self.moving_average(moving_avg_test,
                                                                                       test_metrics_result, "Test")
                    # Define early stopping
                    if early_stop:
                        for metric_name, metric_score in zip(self.metric, moving_avg_m):
                            if not early_stop_lower:
                                early_get = early_stop.get(metric_name, np.inf)
                                if metric_score > early_get:
                                    logger.info("Early stopping: Test metric score {}, "
                                                "Expected score {} [{}]".format(metric_score, early_get, metric_name))
                                    return
                            else:
                                if metric_score < -early_get:
                                    logger.info("Early stopping: Test metric score {}, "
                                                "Expected score {} [{}]".format(metric_score, early_get, metric_name))
                                    return

                    # Log progress
                    logger.info(train_info_str + test_info_str + "Sample number: {} | Time: {}"
                                .format(sample_iter, time.time() - start_time))

                # Save summary
                if sample_iter % summary_step == 0:
                    self.summary_save(batch_x, batch_y, test_batch_x, test_batch_y, reshape_input,
                                      merged_summary, epoch_idx * sample_per_epoch + sample_iter)
                # Add embeddings
                if (epoch_idx * sample_per_epoch + sample_iter) % 10 == 0:
                    if self.embedding_handler is not None:
                        embedding_data = {self.input_layer_placeholder: embedd_img,
                                          self.output_labels_placeholder: embedd_labels,
                                          self.is_training_placeholder: False}
                        self.sess.run(self.embedding_handler, feed_dict=embedding_data)
                        log_path = os.path.join(self.summary_path, "train")
                        self.saver.save(self.sess, os.path.join(log_path, "model.ckpt"),
                                        epoch_idx * sample_per_epoch + sample_iter)
            if save_model:
                self.save(global_step=epoch_idx * sample_per_epoch)
