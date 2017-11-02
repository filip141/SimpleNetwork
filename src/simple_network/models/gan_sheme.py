import os
import time
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from simple_network.tools.utils import Messenger
from simple_network.models.network_parallel import NetworkParallel


class GANScheme(object):
    __metaclass__ = ABCMeta

    def __init__(self, generator_input_size, discriminator_input_size, log_path):
        self.input_summary = {"img_number": 30}
        self.generator_input_size = generator_input_size
        self.discriminator_input_size = discriminator_input_size
        self.generator_learning_rate = None
        self.discriminator_learning_rate = None
        self.session = tf.Session()

        # Define paths
        generator_path = os.path.join(log_path, "generator")
        self.generator_path = generator_path
        if not os.path.isdir(generator_path):
            os.mkdir(generator_path)
        discriminator_path = os.path.join(log_path, "discriminator")
        self. discriminator_path = discriminator_path
        if not os.path.isdir(discriminator_path):
            os.mkdir(discriminator_path)

        # Define Fake Discriminator model
        self.discriminator_fake = None
        # Define Discriminator model
        self.discriminator = NetworkParallel(discriminator_input_size, input_summary=self.input_summary,
                                             summary_path=discriminator_path, session=self.session)
        # Define Generator model
        self.generator = NetworkParallel(generator_input_size, summary_path=generator_path, session=self.session)

    @abstractmethod
    def build_generator(self, generator):
        pass

    @abstractmethod
    def build_discriminator(self, discriminator):
        pass

    @staticmethod
    def generator_loss(logits, targets):
        with tf.name_scope("Generator_loss"):
            loss_comp = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
            red_mean = tf.reduce_mean(loss_comp)
            tf.summary.scalar("Generator_loss", red_mean)
        return red_mean

    @staticmethod
    def discriminator_loss(logits_1, targets_1, logits_2, targets_2):
        with tf.name_scope("Discriminator_loss"):
            loss_1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_1, labels=targets_1)
            red_mean_1 = tf.reduce_mean(loss_1)
            loss_2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_2, labels=targets_2)
            red_mean_2 = tf.reduce_mean(loss_2)
            red_mean_overall = red_mean_1 + red_mean_2
            tf.summary.scalar("Discriminator_loss", red_mean_overall)
            tf.summary.scalar("Discriminator_loss_real", red_mean_2)
            tf.summary.scalar("Discriminator_loss_fake", red_mean_1)
        return red_mean_overall

    def build_model(self, discriminator_learning_rate, generator_learning_rate):
        # Build generator layers
        with tf.variable_scope("generator"):
            self.build_generator(self.generator)
            Messenger.title_message("Building Generator Network")
            self.generator.build_model(generator_learning_rate, out_placeholder=False)
        # Define fake generator
        self.discriminator_fake = NetworkParallel(self.discriminator_input_size, input_summary=self.input_summary,
                                                  summary_path=self.discriminator_path,
                                                  input_placeholder=self.generator.layer_outputs[-1],
                                                  session=self.session)
        with tf.variable_scope("discriminator"):
            self.build_discriminator(self.discriminator)
            self.build_discriminator(self.discriminator_fake)
            Messenger.title_message("Building Discriminator Network")
            # build real discriminator
            with tf.name_scope("discriminator_real"):
                self.discriminator.build_model(discriminator_learning_rate, out_placeholder=False)
            tf.get_variable_scope().reuse_variables()
            Messenger.title_message("Building Discriminator Fake Network")
            # Build fake discriminator
            with tf.name_scope("discriminator_fake"):
                self.discriminator_fake.build_model(discriminator_learning_rate, out_placeholder=False)
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate

    def train(self, train_iter, generator_steps=10, train_step=100, epochs=1000, sample_per_epoch=1000,
              summary_step=5, reshape_input=None, save_model=True):
        # Check Build model
        if not self.discriminator.model_build:
            raise AttributeError("Discriminator Model should be build before training it.")
        if not self.generator.model_build:
            raise AttributeError("Generator Model should be build before training it.")

        # Create losses for both networks
        fake_image = self.discriminator_fake.layer_outputs[-1]
        real_image = self.discriminator.layer_outputs[-1]
        generator_loss = self.generator_loss(fake_image, tf.ones_like(fake_image))
        discriminator_loss = self.discriminator_loss(fake_image, tf.zeros_like(fake_image),
                                                     real_image, tf.ones_like(real_image))

        # Used trick acquired from github page
        g_vars = [var for var in tf.trainable_variables() if "discriminator" in var.name]
        d_vars = [var for var in tf.trainable_variables() if "generator" in var.name]

        # Define optimizers Generator
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer_generator = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=0.5)
        with tf.control_dependencies(update_ops):
            optimizer_generator = optimizer_generator.minimize(generator_loss, var_list=g_vars)

        # Define optimizers Discriminator
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer_discriminator = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.5)
        with tf.control_dependencies(update_ops):
            optimizer_discriminator = optimizer_discriminator.minimize(discriminator_loss, var_list=d_vars)

        self.discriminator.optimizer_func = optimizer_discriminator
        self.generator.optimizer_func = optimizer_generator

        # Save time
        start_time = time.time()

        # Define writers for discriminator
        self.discriminator.train_writer.add_graph(self.discriminator.sess.graph)

        # Define writers for generator
        self.generator.train_writer.add_graph(self.generator.sess.graph)

        # Merge summaries
        merged_summary = tf.summary.merge_all()
        self.session.run(tf.global_variables_initializer())

        # Set Moving Average for both models
        gen_moving_avg_train = []
        dsc_moving_avg_train = []
        for epoch_idx in range(epochs):
            # samples in epoch
            Messenger.text("Training: Epoch number {}".format(epoch_idx))
            for sample_iter in range(sample_per_epoch):
                # Load Training batch
                batch_x, _ = train_iter.next_batch(train_step)
                # reshape train input if defined
                if reshape_input is not None:
                    batch_x = batch_x.reshape([train_step, ] + reshape_input)

                vec_ref = np.random.normal(size=[train_step, ] + self.generator.input_size)
                dsc_train_data = {self.discriminator.input_layer_placeholder: batch_x,
                                  self.generator.input_layer_placeholder: vec_ref,
                                  self.discriminator.is_training_placeholder: True,
                                  self.generator.is_training_placeholder: False,
                                  self.discriminator_fake.is_training_placeholder: True}
                gen_train_data = {self.discriminator.input_layer_placeholder: batch_x,
                                  self.generator.input_layer_placeholder: vec_ref,
                                  self.discriminator.is_training_placeholder: False,
                                  self.generator.is_training_placeholder: True,
                                  self.discriminator_fake.is_training_placeholder: False}
                test_train_data = {self.discriminator.input_layer_placeholder: batch_x,
                                   self.generator.input_layer_placeholder: vec_ref,
                                   self.discriminator.is_training_placeholder: False,
                                   self.generator.is_training_placeholder: False,
                                   self.discriminator_fake.is_training_placeholder: False}

                # Train
                self.session.run(self.discriminator.optimizer_func, feed_dict=dsc_train_data)
                if sample_iter % generator_steps == 0:
                    self.session.run(self.generator.optimizer_func, feed_dict=gen_train_data)

                err_generator = self.session.run(generator_loss, feed_dict=test_train_data)
                err_discriminator = self.session.run(discriminator_loss, feed_dict=test_train_data)
                gen_moving_avg_train = gen_moving_avg_train[-9:] + [err_generator]
                dsc_moving_avg_train = dsc_moving_avg_train[-9:] + [err_discriminator]
                train_info_str = "Discriminator loss: {} | Generator loss: {} | Sample number: {} | Time: {}"\
                    .format(np.mean(dsc_moving_avg_train), np.mean(gen_moving_avg_train),
                            sample_iter, time.time() - start_time)
                # Show message
                Messenger.text(train_info_str)

                # Save summary
                if sample_iter % summary_step == 0:
                    sum_res_discriminator = self.session.run(merged_summary, test_train_data)
                    sum_res_generator = self.session.run(merged_summary, test_train_data)
                    self.discriminator.train_writer.add_summary(
                        sum_res_discriminator, epoch_idx * sample_per_epoch + sample_iter)
                    self.generator.train_writer.add_summary(
                        sum_res_generator, epoch_idx * sample_per_epoch + sample_iter)
            if save_model:
                self.discriminator.save(global_step=epoch_idx * sample_per_epoch)
                self.generator.save(global_step=epoch_idx * sample_per_epoch)

