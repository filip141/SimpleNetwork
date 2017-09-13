import time
import logging
import tensorflow as tf
from simple_network.models.netmodel import SNModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkModel(SNModel):

    def __init__(self, input_size, summary_path=None, metric=None, input_summary=None):
        # If summary_path is None set tempdir
        super(NetworkModel, self).__init__(input_size, summary_path, metric, input_summary)

    def build_model(self, learning_rate):
        # Define input
        logger.info("-" * 90)
        with tf.name_scope("input_layer"):
            self.prepare_input()
        logger.info("Input layer| shape: {}".format(self.input_size))

        # build model
        layer_input = self.input_layer_placeholder
        self.is_training_placeholder = tf.placeholder(tf.bool, name='is_training')
        for l_idx, layer in enumerate(self.layers):
            layer_output = self.build_layer(layer, layer_input, l_idx)
            layer_input = layer_output
        # Output placeholder
        self.output_labels_placeholder = tf.placeholder(tf.float32, [None, self.layers[-1].layer_size[-1]],
                                                        name='labels')
        y_labels = self.output_labels_placeholder

        # Define loss and optimizer
        self.loss_func = self.get_loss_by_name()(logits=layer_output, labels=y_labels, loss_data=self.loss_data)
        logger.info("Loss function: {}".format(self.loss))
        self.optimizer_func = self.get_optimizer_by_name()(self.loss_func, learning_rate, self.optimizer_data)
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
        self.saver = tf.train.Saver()

    def train(self, train_iter, test_iter, train_step=100, test_step=100, epochs=1000, sample_per_epoch=1000,
              learning_rate=0.001, summary_step=5, reshape_input=None, save_model=True):
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
                self.sess.run(self.optimizer_func, feed_dict=train_data)
                if self.metric is not None:
                    train_metrics_result = self.sess.run(self.metric_list_func, feed_dict=train_data)
                    train_metric_result_dict = dict(zip(self.metric, train_metrics_result))
                    train_info_str = "Training set metrics: {} | ".format(", ".join(
                        [": ".join((tm_k.title(), str(tm_v))) for tm_k, tm_v in train_metric_result_dict.items()]))

                    test_data = {self.input_layer_placeholder: test_batch_x, self.output_labels_placeholder: test_batch_y,
                                 self.is_training_placeholder: False}
                    test_metrics_result = self.sess.run(self.metric_list_func, feed_dict=test_data)
                    test_metric_result_dict = dict(zip(self.metric, test_metrics_result))
                    test_info_str = "Test set metrics: {}".format(", ".join(
                        [": ".join((tm_k.title(), str(tm_v))) for tm_k, tm_v in test_metric_result_dict.items()]))
                    logger.info(train_info_str + test_info_str + " | Sample number: {} | Time: {}"
                                .format(sample_iter, time.time() - start_time))

                # Save summary
                if sample_iter % summary_step == 0:
                    sum_res = self.sess.run(merged_summary, train_data)
                    self.writer.add_summary(sum_res, epoch_idx * sample_per_epoch + sample_iter)
            if save_model:
                self.save(global_step=epoch_idx * sample_per_epoch)