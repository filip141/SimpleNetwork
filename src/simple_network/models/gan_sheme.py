import os
import time
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from simple_network.tools.utils import Messenger, ModelLogger
from simple_network.models.network_parallel import NetworkParallel
from simple_network.train.optimizers import adam_optimizer, momentum_optimizer, rmsprop_optimizer, sgd_optimizer


class GANScheme(object):
    __metaclass__ = ABCMeta

    def __init__(self, generator_input_size, discriminator_input_size, log_path, batch_size,
                 labels='none', labels_size=10):
        self.input_summary = {"img_number": 5}
        self.batch_size = batch_size

        # Information about labels
        self.labels = labels
        self.labels_size = labels_size
        self.labels_placeholder = None

        # Save Generator and Discriminator input size
        self.generator_in = None
        self.discriminator_in = None
        self.generator_true_size = [generator_input_size, ]
        self.generator_input_size = [generator_input_size, ]
        self.discriminator_input_size = discriminator_input_size

        # Set information about loss used in training
        self.gan_model_loss = 'js-non-saturation'
        self.gan_model_loss_data = None

        # Save information about optimizers
        self.generator_optimizer = "Adam"
        self.generator_optimizer_data = None
        self.discriminator_optimizer = "Adam"
        self.discriminator_optimizer_data = None

        # Create variables for learning rate and initialize
        # Tensorflow session
        self.generator_learning_rate = None
        self.discriminator_learning_rate = None
        self.session = tf.Session()

        # Define paths and create them if not exist
        self.log_path = log_path
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        self.model_gan_path = os.path.join(log_path, "info")
        if not os.path.isdir(self.model_gan_path):
            os.mkdir(self.model_gan_path)

        generator_path = os.path.join(log_path, "generator")
        self.generator_path = generator_path
        if not os.path.isdir(generator_path):
            os.mkdir(generator_path)
        discriminator_path = os.path.join(log_path, "discriminator")
        self. discriminator_path = discriminator_path
        if not os.path.isdir(discriminator_path):
            os.mkdir(discriminator_path)
        Messenger.set_logger_path(os.path.join(self.log_path, "log.log"))

        if labels == 'convo-semi-supervised':
            # Define input placeholder for generator
            gen_input_size_final = [self.generator_input_size[0] + labels_size, ]
            input_gen_size = [None, ] + self.generator_input_size
            input_gen_placeholder = tf.placeholder(tf.float32, input_gen_size, name='gen_x')

            # define placeholder for discriminator
            dsc_input_size_final = [discriminator_input_size[0], discriminator_input_size[1],
                                    discriminator_input_size[2] + labels_size]
            input_dsc_size = [None, ] + self.discriminator_input_size
            input_dsc_placeholder = tf.placeholder(tf.float32, input_dsc_size, name="dsc_x")

            # Define placeholder for labels
            labels_size_t = [None, labels_size]
            labels_placeholder = tf.placeholder(tf.float32, labels_size_t, name='conditional_placeholder')

            # Concat labels with generator input
            new_gen_placeholder = tf.concat(axis=1, values=[input_gen_placeholder, labels_placeholder])

            # Concat labels with discriminator input
            labels_y1 = tf.reshape(labels_placeholder, shape=[self.batch_size, 1, 1, labels_size])
            new_dsc_placeholder = tf.concat(axis=3, values=[input_dsc_placeholder,
                                                            labels_y1*tf.ones(
                                                                [self.batch_size,
                                                                 input_dsc_size[1],
                                                                 input_dsc_size[2], labels_size])])

            # Save labels and input images placeholder also construct input size vector for encoder
            self.labels_placeholder = labels_placeholder
            self.generator_in = input_gen_placeholder
            self.discriminator_in = input_dsc_placeholder
            # Modify generator input
            self.generator_input_size = [self.generator_input_size[0] + labels_size]
        else:
            # Vanilla GAN mode, in this option labels placeholder is not merged with input placeholder
            input_gen_size = [None, ] + self.generator_input_size
            new_gen_placeholder = tf.placeholder(tf.float32, input_gen_size, name='gen_x')

            # Define placeholder for discriminator
            input_dsc_size = [None, ] + self.discriminator_input_size
            new_dsc_placeholder = tf.placeholder(tf.float32, input_dsc_size, name='dsc_x')

            # Define input size vectors for vanilla VAE
            dsc_input_size_final = self.discriminator_input_size
            gen_input_size_final = self.generator_input_size
            self.generator_in = new_gen_placeholder
            self.discriminator_in = new_dsc_placeholder

        # Define Fake Discriminator model
        self.discriminator_fake = None
        # Define Discriminator model
        self.generator_output_size = self.discriminator_input_size
        self.discriminator = NetworkParallel(dsc_input_size_final, input_summary=self.input_summary,
                                             summary_path=discriminator_path, session=self.session,
                                             input_placeholder=new_dsc_placeholder)
        # Define Generator model
        self.generator = NetworkParallel(gen_input_size_final, summary_path=generator_path, session=self.session,
                                         input_placeholder=new_gen_placeholder)

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
    def generator_kl_loss(logits, targets):
        with tf.name_scope("Generator_loss"):
            d_g_z = tf.nn.sigmoid(logits)
            loss_comp = - targets * tf.log(d_g_z / (1 - d_g_z))
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

    def save_model_info(self):
        model_logger = ModelLogger(self.model_gan_path)
        model_logger.title("Discriminator")
        model_logger.add_property("Discriminator Input", self.discriminator_input_size)
        model_logger.add_to_json("discriminator_input", self.discriminator_input_size)

        model_logger.add_property("Discriminator Optimizer", self.discriminator_optimizer)
        model_logger.add_to_json("discriminator_optimizer", self.discriminator_optimizer)

        model_logger.add_property("Discriminator Learning Rate", self.discriminator_learning_rate)
        model_logger.add_to_json("learning_rate", self.discriminator_learning_rate)

        model_logger.add_property("Discriminator Path", self.discriminator_path)
        model_logger.add_to_json("discriminator_path", self.discriminator_path)

        model_logger.title("Generator")
        model_logger.add_property("Generator Input", self.generator_input_size)
        model_logger.add_to_json("generator_input", self.generator_input_size)

        model_logger.add_property("Generator Optimizer", self.generator_optimizer)
        model_logger.add_to_json("generator_optimizer", self.generator_optimizer)

        model_logger.add_property("Generator Learning Rate", self.generator_learning_rate)
        model_logger.add_to_json("learning_rate", self.generator_learning_rate)

        model_logger.add_property("Generator Path", self.generator_path)
        model_logger.add_to_json("discriminator_path", self.generator_path)

        model_logger.title("GAN")
        model_logger.add_property("GAN Loss", self.gan_model_loss)
        model_logger.add_to_json("gan_loss", self.gan_model_loss)

        model_logger.add_property("GAN Loss Data", self.gan_model_loss_data)
        model_logger.add_to_json("gan_loss_data", self.gan_model_loss_data)

        model_logger.add_property("GAN Labels Type", self.labels)
        model_logger.add_to_json("gan_labels_type", self.labels)
        model_logger.save()

    def model_compile(self, discriminator_learning_rate, generator_learning_rate):
        # Build generator layers
        with tf.variable_scope("generator"):
            self.build_generator(self.generator)
            Messenger.title_message("Building Generator Network")
            self.generator.build_model(generator_learning_rate, out_placeholder=False)

        # Define fake generator
        dsc_final_shape = self.discriminator_input_size
        dsc_fake_input_placeholder = self.generator.layer_outputs[-1]
        if self.labels == "convo-semi-supervised":
            labels_y1 = tf.reshape(self.labels_placeholder, shape=[-1, 1, 1, self.labels_size])
            dsc_fake_input_placeholder = tf.concat(axis=3, values=[dsc_fake_input_placeholder,
                                                                   labels_y1*tf.ones(
                                                                       [self.batch_size,
                                                                        self.discriminator_input_size[0],
                                                                        self.discriminator_input_size[1],
                                                                        self.labels_size])])
            dsc_final_shape = [self.discriminator_input_size[0], self.discriminator_input_size[1],
                               self.discriminator_input_size[2] + self.labels_size]
        self.discriminator_fake = NetworkParallel(dsc_final_shape, input_summary=self.input_summary,
                                                  summary_path=self.discriminator_path,
                                                  input_placeholder=dsc_fake_input_placeholder,
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

    def set_generator_optimizer(self, optimizer, **kwargs):
        self.generator_optimizer = optimizer
        self.generator_optimizer_data = kwargs

    def set_discriminator_optimizer(self, optimizer, **kwargs):
        self.discriminator_optimizer = optimizer
        self.discriminator_optimizer_data = kwargs

    def get_optimizer_by_name(self, optimizer_name):
        # Check optimizer
        if optimizer_name is None:
            return None
        if optimizer_name == "Adam":
            return adam_optimizer
        elif optimizer_name == "Momentum":
            return momentum_optimizer
        elif optimizer_name == "RMSprop":
            return rmsprop_optimizer
        elif optimizer_name == "SGD":
            return sgd_optimizer
        else:
            raise AttributeError("Optimizer {} not defined.".format(self.optimizer))

    def set_loss(self, loss, **kwargs):
        self.gan_model_loss = loss
        self.gan_model_loss_data = kwargs

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

    def kl_gan_loss(self):
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
        generator_loss = self.generator_kl_loss(fake_image, tf.ones_like(fake_image))
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

    def restore(self):
        try:
            self.discriminator_fake.restore()
            self.discriminator.restore()
            self.generator.restore()
            Messenger.text("Model successful restored.")
        except Exception:
            Messenger.text("No restore points in {}".format(self.log_path))

    def train(self, train_iter, generator_steps=1, discriminator_steps=1, train_step=100, epochs=1000,
              sample_per_epoch=1000, summary_step=5, reshape_input=None, save_model=True, restore_model=True):
        # Check Build model
        if not self.discriminator.model_build:
            raise AttributeError("Discriminator Model should be build before training it.")
        if not self.generator.model_build:
            raise AttributeError("Generator Model should be build before training it.")
        self.save_model_info()

        # GAN model chosen loss
        if self.gan_model_loss == 'js-non-saturation':
            generator_loss, discriminator_loss = self.js_non_saturation_gan_loss()
        elif self.gan_model_loss == 'js-saturation':
            generator_loss, discriminator_loss = self.js_saturation_gan_loss()
        elif self.gan_model_loss == 'kl-qp-loss':
            generator_loss, discriminator_loss = self.kl_gan_loss()
        elif self.gan_model_loss == 'feature-matching':
            generator_loss, discriminator_loss = self.feature_matching_gan_loss()
        else:
            raise ValueError("Loss function not defined")

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

                # reshape train input if defined
                if reshape_input is not None:
                    batch_x = batch_x.reshape([train_step, ] + reshape_input)

                # Create Feed dict for tensorflow
                vec_ref = np.random.normal(0.0, 1.0, size=[train_step, ] + self.generator_true_size)
                dsc_train_data = {self.discriminator_in: batch_x,
                                  self.generator_in: vec_ref,
                                  self.discriminator.is_training_placeholder: True,
                                  self.generator.is_training_placeholder: True,
                                  self.discriminator_fake.is_training_placeholder: True}
                # Add labels for semi-supervised learning
                if self.labels == "convo-semi-supervised":
                    dsc_train_data[self.labels_placeholder] = batch_y

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
                gen_moving_avg_train = gen_moving_avg_train[-9:] + [err_generator]
                dsc_moving_avg_train = dsc_moving_avg_train[-9:] + [err_discriminator]
                dsc_moving_mean = np.mean(dsc_moving_avg_train)
                gen_moving_mean = np.mean(gen_moving_avg_train)

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

