import time
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from simple_network.tools.utils import Messenger
from simple_network.models.gan_sheme import GANScheme


class WasserstainGANScheme(GANScheme):
    __metaclass__ = ABCMeta

    def __init__(self, generator_input_size, discriminator_input_size, log_path, batch_size, labels='none',
                 labels_size=10):
        super(WasserstainGANScheme, self).__init__(generator_input_size, discriminator_input_size, log_path,
                                                   batch_size, labels, labels_size)

    @abstractmethod
    def build_generator(self, generator):
        pass

    @abstractmethod
    def build_discriminator(self, discriminator):
        pass

    def wasserstain_gan_loss(self):
        fake_image = self.discriminator_fake.layer_outputs[-1]
        real_image = self.discriminator.layer_outputs[-1]

        # Create losses for both networks
        generator_loss = tf.reduce_mean(fake_image)
        critic_loss = tf.reduce_mean(real_image) - tf.reduce_mean(fake_image)
        return generator_loss, critic_loss

    @staticmethod
    def clip_weights(d_vars):
        clip_values = [-0.01, 0.01]
        clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for
                                     var in d_vars]
        return clip_discriminator_var_op

    def prepare_input(self, batch_x, batch_y, batch_size, reshape_input):
        # reshape train input if defined
        if reshape_input is not None:
            batch_x = batch_x.reshape([batch_size, ] + reshape_input)

        # Create Feed dict for tensorflow
        vec_ref = np.random.normal(0.0, 1.0, size=[batch_size, ] + self.generator_true_size)
        dsc_train_data = {self.discriminator_in: batch_x,
                          self.generator_in: vec_ref,
                          self.discriminator.is_training_placeholder: True,
                          self.generator.is_training_placeholder: True,
                          self.discriminator_fake.is_training_placeholder: True}
        # Add labels for semi-supervised learning
        if self.labels == "convo-semi-supervised":
            dsc_train_data[self.labels_placeholder] = batch_y
        return dsc_train_data

    @staticmethod
    def moving_average(moving_avg_list, new_sample):
        moving_avg_list = moving_avg_list[-9:] + [new_sample]
        moving_mean = np.mean(moving_avg_list)
        return moving_mean

    def train(self, train_iter, generator_steps=1, discriminator_steps=1, train_step=100, epochs=1000,
              sample_per_epoch=1000, summary_step=5, reshape_input=None, save_model=True, restore_model=True):
        # Check Build model
        if not self.discriminator.model_build:
            raise AttributeError("Discriminator Model should be build before training it.")
        if not self.generator.model_build:
            raise AttributeError("Generator Model should be build before training it.")
        self.save_model_info()

        # GAN model chosen loss
        Messenger.text("WGAN Loss: {}".format(self.gan_model_loss))
        generator_loss, discriminator_loss = self.wasserstain_gan_loss()

        # Used trick acquired from github page
        d_vars = [var for var in tf.trainable_variables() if "discriminator" in var.name]
        g_vars = [var for var in tf.trainable_variables() if "generator" in var.name]

        # clip weights in D
        clip_w = self.clip_weights(d_vars)

        # Define optimizers Generator and Discriminator
        with tf.name_scope("train"):
            optimizer_generator = self.get_optimizer_by_name(self.generator_optimizer)(
                generator_loss, self.generator_learning_rate, self.generator_optimizer_data, g_vars)
            optimizer_discriminator = self.get_optimizer_by_name(self.discriminator_optimizer)(
                discriminator_loss,  self.discriminator_learning_rate, self.discriminator_optimizer_data, d_vars)

        # Information about optimizers
        Messenger.fancy_message("Generator optimizer: {}".format(self.generator_optimizer))
        Messenger.fancy_message("Critic optimizer: {}".format(self.discriminator_optimizer))

        # Save optimizer functions
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

        # Restore model
        if restore_model:
            self.restore()

        # Set Moving Average for both models
        gen_moving_avg_train = []
        dsc_moving_avg_train = []
        for epoch_idx in range(epochs):
            # samples in epoch
            Messenger.text("Training: Epoch number {}".format(epoch_idx))
            for sample_iter in range(sample_per_epoch):

                # Load Training batch
                batch_x, batch_y = train_iter.next_batch(train_step)

                dsc_train_data = self.prepare_input(batch_x, batch_y, train_step, reshape_input)

                # Train network
                update_gen = ""
                update_dsc = ""
                if sample_iter % discriminator_steps == 0:
                    update_dsc = "Critic"
                    self.session.run(self.discriminator.optimizer_func, feed_dict=dsc_train_data)
                    self.session.run(clip_w)
                if sample_iter % generator_steps == 0:
                    update_gen = "Generator"
                    self.session.run(self.generator.optimizer_func, feed_dict=dsc_train_data)
                updated_network = "{} {}".format(update_gen, update_dsc)

                # Calculate loss and moving mean for it
                err_generator = self.session.run(generator_loss, feed_dict=dsc_train_data)
                err_discriminator = self.session.run(discriminator_loss, feed_dict=dsc_train_data)
                gen_moving_mean = self.moving_average(gen_moving_avg_train, err_generator)
                dsc_moving_mean = self.moving_average(dsc_moving_avg_train, err_discriminator)

                train_info_str = "Discriminator loss: {} | Generator loss: {} | Updated {} | Sample number: {} | " \
                                 "Time: {}"\
                    .format(dsc_moving_mean, gen_moving_mean, updated_network, sample_iter,
                            time.time() - start_time)
                # Show message
                Messenger.text(train_info_str)

                # Save summary
                if sample_iter % summary_step == 0:
                    sum_res_discriminator = self.session.run(merged_summary, dsc_train_data)
                    sum_res_generator = self.session.run(merged_summary, dsc_train_data)
                    self.discriminator.train_writer.add_summary(
                        sum_res_discriminator, epoch_idx * sample_per_epoch + sample_iter)
                    self.generator.train_writer.add_summary(
                        sum_res_generator, epoch_idx * sample_per_epoch + sample_iter)
            if save_model:
                self.discriminator.save(global_step=epoch_idx * sample_per_epoch)
                self.generator.save(global_step=epoch_idx * sample_per_epoch)
