import time
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from simple_network.tools.utils import Messenger
from simple_network.models.gan_sheme import GANScheme


class MMGANScheme(GANScheme):
    __metaclass__ = ABCMeta

    def __init__(self, generator_input_size, discriminator_input_size, log_path, batch_size, labels='none',
                 labels_size=10):
        super(MMGANScheme, self).__init__(generator_input_size, discriminator_input_size, log_path, batch_size, labels,
                                          labels_size)

    @abstractmethod
    def build_generator(self, generator):
        pass

    @abstractmethod
    def build_discriminator(self, discriminator):
        pass

    @staticmethod
    def generator_js_non_saturation_loss(logits, targets):
        with tf.name_scope("Generator_loss"):
            loss_comp = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
            red_mean = tf.reduce_mean(loss_comp)
            tf.summary.scalar("Generator_loss", red_mean)
        return red_mean

    @staticmethod
    def generator_js_saturation_loss(logits, targets):
        with tf.name_scope("Generator_loss"):
            layer_activation = tf.nn.sigmoid(logits)
            loss_comp = targets * tf.log(1 - layer_activation)
            red_mean = tf.reduce_mean(loss_comp)
            tf.summary.scalar("Generator_loss", red_mean)
        return red_mean

    @staticmethod
    def generator_feature_matching_loss(dsc_fake_act, dsc_legit_act):
        with tf.name_scope("Generator_loss"):
            m_legit = tf.reduce_mean(dsc_legit_act, axis=0)
            m_fake = tf.reduce_mean(dsc_fake_act, axis=0)
            loss_gen = tf.reduce_mean(tf.abs(m_legit - m_fake))
            tf.summary.scalar("Generator_loss", loss_gen)
        return loss_gen

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

    def js_non_saturation_gan_loss(self):
        lb_smooth = self.gan_model_loss_data.get("label_smooth", False)
        fake_image = self.discriminator_fake.layer_outputs[-1]
        real_image = self.discriminator.layer_outputs[-1]

        # Add label smoothing
        Messenger.text("Label-Smoothing {}".format(lb_smooth))
        if lb_smooth:
            legit_dsc_targets = 0.9 * tf.ones_like(real_image)
        else:
            legit_dsc_targets = tf.ones_like(real_image)
        fake_dsc_targets = tf.zeros_like(real_image)
        # Create losses for both networks
        generator_loss = self.generator_js_non_saturation_loss(fake_image, tf.ones_like(fake_image))
        discriminator_loss = self.discriminator_loss(fake_image, fake_dsc_targets,
                                                     real_image, legit_dsc_targets)
        return generator_loss, discriminator_loss

    def js_saturation_gan_loss(self):
        lb_smooth = self.gan_model_loss_data.get("label_smooth", False)
        fake_image = self.discriminator_fake.layer_outputs[-1]
        real_image = self.discriminator.layer_outputs[-1]

        # Add label smoothing
        Messenger.text("Label-Smoothing {}".format(lb_smooth))
        if lb_smooth:
            legit_dsc_targets = 0.9 * tf.ones_like(real_image)
        else:
            legit_dsc_targets = tf.ones_like(real_image)
        fake_dsc_targets = tf.zeros_like(real_image)
        # Create losses for both networks
        generator_loss = self.generator_js_saturation_loss(fake_image, tf.ones_like(fake_image))
        discriminator_loss = self.discriminator_loss(fake_image, fake_dsc_targets,
                                                     real_image, legit_dsc_targets)
        return generator_loss, discriminator_loss

    def feature_matching_gan_loss(self):
        Messenger.text("Using Feature-Matching Cost")
        layer_num = self.gan_model_loss_data.get("no_layer", -2)
        lb_smooth = self.gan_model_loss_data.get("label_smooth", False)

        dsc_legit_act = self.discriminator.layer_outputs[layer_num]
        dsc_fake_act = self.discriminator_fake.layer_outputs[layer_num]
        fake_image = self.discriminator_fake.layer_outputs[-1]
        real_image = self.discriminator.layer_outputs[-1]

        # Add label smoothing
        Messenger.text("Label-Smoothing {}".format(lb_smooth))
        if lb_smooth:
            legit_dsc_targets = 0.9 * tf.ones_like(real_image)
        else:
            legit_dsc_targets = tf.ones_like(real_image)
        fake_dsc_targets = tf.zeros_like(real_image)
        # Create losses for both networks
        generator_loss = self.generator_feature_matching_loss(dsc_fake_act, dsc_legit_act)
        discriminator_loss = self.discriminator_loss(fake_image, fake_dsc_targets,
                                                     real_image, legit_dsc_targets)
        return generator_loss, discriminator_loss

    def get_gan_loss_func(self):
        if self.gan_model_loss == 'js-non-saturation':
            generator_loss, discriminator_loss = self.js_non_saturation_gan_loss()
        elif self.gan_model_loss == 'js-saturation':
            generator_loss, discriminator_loss = self.js_saturation_gan_loss()
        elif self.gan_model_loss == 'feature-matching':
            generator_loss, discriminator_loss = self.feature_matching_gan_loss()
        else:
            raise ValueError("Loss function not defined")
        return generator_loss, discriminator_loss

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
        Messenger.text("Gan Loss: {}".format(self.gan_model_loss))
        generator_loss, discriminator_loss = self.get_gan_loss_func()

        # Used trick acquired from github page
        d_vars = [var for var in tf.trainable_variables() if "discriminator" in var.name]
        g_vars = [var for var in tf.trainable_variables() if "generator" in var.name]

        # Define optimizers Generator and Discriminator
        with tf.name_scope("train"):
            optimizer_generator = self.get_optimizer_by_name(self.generator_optimizer)(
                generator_loss, self.generator_learning_rate, self.generator_optimizer_data, g_vars)
            optimizer_discriminator = self.get_optimizer_by_name(self.discriminator_optimizer)(
                discriminator_loss,  self.discriminator_learning_rate, self.discriminator_optimizer_data, d_vars)

        # Information about optimizers
        Messenger.fancy_message("Generator optimizer: {}".format(self.generator_optimizer))
        Messenger.fancy_message("Discriminator optimizer: {}".format(self.discriminator_optimizer))

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
                    update_dsc = "Discriminator"
                    self.session.run(self.discriminator.optimizer_func, feed_dict=dsc_train_data)
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

