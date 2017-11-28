import os
import time
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from simple_network.tools.utils import Messenger
from simple_network.models.network_parallel import NetworkParallel
from simple_network.train.optimizers import adam_optimizer, momentum_optimizer, rmsprop_optimizer, sgd_optimizer


class VAEScheme(object):
    __metaclass__ = ABCMeta

    def __init__(self, encoder_input_size, decoder_input_size, batch_size, log_path, labels='none',
                 labels_size=10):
        self.input_summary = {"img_number": 5}
        self.batch_size = batch_size
        self.labels = labels
        self.labels_size = labels_size

        self.encoder_in = None
        self.encoder_input_size = encoder_input_size
        self.decoder_input_size = [decoder_input_size, ]

        # Optimizer
        self.loss_func = None
        self.optimizer = "Adam"
        self.optimizer_data = None

        # Learning rate and tensorflow session
        self.learning_rate = None
        self.session = tf.Session()

        # Encoder variables
        self.encoder_labels = None
        self.decoder_labels = None
        self.encoder_z_mean = None
        self.encoder_z_logstd = None

        # Define paths and create them if not exist
        self.log_path = log_path
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        self.model_gan_path = os.path.join(log_path, "info")
        if not os.path.isdir(self.model_gan_path):
            os.mkdir(self.model_gan_path)

        encoder_path = os.path.join(log_path, "encoder")
        self.encoder_path = encoder_path
        if not os.path.isdir(encoder_path):
            os.mkdir(encoder_path)
        decoder_path = os.path.join(log_path, "decoder")
        self.decoder_path = decoder_path
        if not os.path.isdir(decoder_path):
            os.mkdir(decoder_path)

        # Define Encoder and Decoder model
        # For conditional [CVAE] define labels input for encoder,
        if labels == 'convo-semi-supervised':
            # In this mode labels are merged with input placeholder
            # Define
            labels_size_t = [None, 1, labels_size]
            decoder_input_size_final = [self.decoder_input_size[0] + labels_size]
            input_enc_size = [batch_size, ] + self.encoder_input_size
            input_layer_placeholder = tf.placeholder(tf.float32, input_enc_size, name='x')
            labels_placeholder = tf.placeholder(tf.float32, labels_size_t, name='conditional_placeholder')

            # Define labels and concat them
            labels_y1 = tf.reshape(labels_placeholder, shape=[batch_size, 1, 1, labels_size])
            new_input_placeholder = tf.concat(axis=3, values=[input_layer_placeholder,
                                                              labels_y1*tf.ones(
                                                                  [input_enc_size[0], input_enc_size[1],
                                                                   input_enc_size[2], labels_size])])
            # Save labels and input images placeholder also construct input size vector for encoder
            self.encoder_labels = labels_placeholder
            enc_input_size_final = [self.encoder_input_size[0], self.encoder_input_size[1],
                                    self.encoder_input_size[2] + labels_size]
            self.encoder_in = input_layer_placeholder
        else:
            # Vanilla VAE mode, in this option labels placeholder is not merged with input placeholder
            input_enc_size = [None, ] + self.encoder_input_size
            new_input_placeholder = tf.placeholder(tf.float32, input_enc_size, name='x')

            # Define input size vectors for vanilla VAE
            decoder_input_size_final = self.decoder_input_size
            enc_input_size_final = self.encoder_input_size
            self.encoder_in = new_input_placeholder

        # Define network objects for decoder [train], decoder [predict] and encoder
        # Both decoders sharing weights together.
        self.decoder_network_train = None
        self.decoder_output_size = self.encoder_input_size
        self.encoder_network = NetworkParallel(enc_input_size_final, input_summary=self.input_summary,
                                               summary_path=self.encoder_path, session=self.session,
                                               input_placeholder=new_input_placeholder)
        self.decoder_network = NetworkParallel(decoder_input_size_final, input_summary=None,
                                               summary_path=self.decoder_path, session=self.session)

    @abstractmethod
    def build_encoder(self, encoder):
        pass

    @abstractmethod
    def build_decoder(self, decoder):
        pass

    def model_compile(self, learning_rate):
        # Build encoder layers
        with tf.variable_scope("encoder"):
            self.build_encoder(self.encoder_network)
            Messenger.title_message("Building Encoder Network")
            self.encoder_network.build_model(learning_rate, out_placeholder=False)

        # Generate input for decoder
        noise_samples = tf.random_normal([self.batch_size, self.decoder_input_size[0]], 0, 1, dtype=tf.float32)
        self.encoder_z_mean, self.encoder_z_logstd = self.encoder_network.layer_outputs[-1]
        guessed_z = self.encoder_z_mean + (tf.exp(.5 * self.encoder_z_logstd) * noise_samples)

        # Define decoder input for conditional form [CVAE]
        if self.labels == 'convo-semi-supervised':
            labels_size_t = [None, 1, self.labels_size]
            self.decoder_input_size = [self.decoder_input_size[0] + self.labels_size]
            self.decoder_labels = tf.placeholder(tf.float32, labels_size_t, name='decoder_cond_input')
            dec_lab_reshape = tf.reshape(self.decoder_labels, [-1, self.labels_size])
            guessed_z = tf.concat(axis=1, values=[guessed_z, dec_lab_reshape])
        self.decoder_network_train = NetworkParallel(self.decoder_input_size, input_summary=None,
                                                     summary_path=self.decoder_path, session=self.session,
                                                     input_placeholder=guessed_z)
        with tf.variable_scope("decoder"):
            self.build_decoder(self.decoder_network)
            self.build_decoder(self.decoder_network_train)
            # build decoder train
            Messenger.title_message("Building Decoder Network Train")
            with tf.name_scope("decoder_train"):
                self.decoder_network_train.build_model(learning_rate, out_placeholder=False)
            tf.get_variable_scope().reuse_variables()
            Messenger.title_message("Building Decoder Network")
            with tf.name_scope("decoder"):
                self.decoder_network.build_model(learning_rate, out_placeholder=False)
        self.learning_rate = learning_rate

    def set_optimizer(self, optimizer, **kwargs):
        self.optimizer = optimizer
        self.optimizer_data = kwargs

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

    def vae_loss(self, decoder_output, z_mean, z_log_std):
        decoder_out_loss_in = tf.reshape(decoder_output, [-1, np.prod(self.decoder_output_size)])
        encoder_in_loss_in = tf.reshape(self.encoder_in, [-1, np.prod(self.encoder_input_size)])
        generation_loss = tf.reduce_sum(encoder_in_loss_in * tf.log(decoder_out_loss_in + 1e-9) +
                                        (1 - encoder_in_loss_in) *
                                        tf.log(1 - decoder_out_loss_in + 1e-9), axis=1)
        kl_term = -0.5 * tf.reduce_sum(1 + 2 * z_log_std - tf.pow(z_mean, 2) - tf.exp(2 * z_log_std), axis=1)
        vae_loss = -tf.reduce_mean(generation_loss - kl_term)
        tf.summary.scalar("generation_loss", tf.reduce_mean(generation_loss))
        tf.summary.scalar("kl_term", tf.reduce_mean(kl_term))
        tf.summary.scalar("VAE_loss", vae_loss)

        summary_output_data = decoder_output
        summary_shape = decoder_output.get_shape().as_list()
        output_last_dim = summary_shape[-1]
        if len(summary_shape) == 4:
            if output_last_dim != 2 and output_last_dim != 3:
                summary_output_data = tf.expand_dims(summary_output_data[:, :, :, 0], axis=3)
            tf.summary.image('output', summary_output_data, 5)
        return vae_loss

    def restore(self):
        try:
            self.encoder_network.restore()
            self.decoder_network_train.restore()
            self.decoder_network.restore()
            Messenger.text("Model successful restored.")
        except Exception:
            Messenger.text("No restore points in {}".format(self.log_path))

    def train(self, train_iter, train_step=100, epochs=1000, sample_per_epoch=1000, summary_step=5,
              reshape_input=None, save_model=True, restore_model=True):
        # Check Build model
        if not self.encoder_network.model_build:
            raise AttributeError("Encoder Model should be build before training it.")
        if not self.decoder_network_train.model_build:
            raise AttributeError("Decoder Model should be build before training it.")

        # Create losses for vae autoencoder
        decoded_images = self.decoder_network_train.layer_outputs[-1]

        self.loss_func = self.vae_loss(decoded_images, self.encoder_z_mean, self.encoder_z_logstd)

        # Set network optimizer
        with tf.name_scope("train"):
            optimizer_vae = self.get_optimizer_by_name(self.optimizer)(self.loss_func, self.learning_rate,
                                                                       self.optimizer_data)
            Messenger.fancy_message("VAE optimizer: {}".format(self.optimizer))
        self.decoder_network_train.optimizer_func = optimizer_vae

        # Save time
        start_time = time.time()

        # Define writers for decoder and encoder
        self.decoder_network_train.train_writer.add_graph(self.decoder_network_train.sess.graph)
        self.encoder_network.train_writer.add_graph(self.encoder_network.sess.graph)

        # Merge summaries
        merged_summary = tf.summary.merge_all()
        self.session.run(tf.global_variables_initializer())

        # Restore model
        if restore_model:
            self.restore()

        # Set Moving Average for VAE model
        dec_moving_avg_train = []
        for epoch_idx in range(epochs):
            # samples in epoch
            Messenger.text("Training: Epoch number {}".format(epoch_idx))
            for sample_iter in range(sample_per_epoch):
                # Load Training batch
                batch_x, batch_y = train_iter.next_batch(train_step)
                # reshape train input if defined
                if reshape_input is not None:
                    batch_x = batch_x.reshape([train_step, ] + reshape_input)

                dsc_train_data = {}
                # Reshape input and add labels for convo-semi-supervised CVAE
                if self.labels == 'convo-semi-supervised':
                    dsc_train_data[self.encoder_labels] = batch_y[:, np.newaxis]
                    dsc_train_data[self.decoder_labels] = batch_y[:, np.newaxis]
                # Set feed dict data
                dsc_train_data[self.encoder_in] = batch_x
                dsc_train_data[self.encoder_network.is_training_placeholder] = True
                dsc_train_data[self.decoder_network_train.is_training_placeholder] = True

                # Train
                self.session.run(self.decoder_network_train.optimizer_func, feed_dict=dsc_train_data)

                err_decoder = self.session.run(self.loss_func, feed_dict=dsc_train_data)
                dec_moving_avg_train = dec_moving_avg_train[-9:] + [err_decoder]
                train_info_str = "VAE loss: {} | Sample number: {} | Time: {}"\
                    .format(np.mean(dec_moving_avg_train), sample_iter, time.time() - start_time)
                # Show message
                Messenger.text(train_info_str)

                # Save summary
                if sample_iter % summary_step == 0:
                    vec_ref = np.random.uniform(0.0, 1.0, size=[train_step, ] + self.decoder_network.input_size)
                    summary_data = {self.encoder_in: batch_x,
                                    self.decoder_network.input_layer_placeholder: vec_ref,
                                    self.encoder_network.is_training_placeholder: False,
                                    self.decoder_network_train.is_training_placeholder: False,
                                    self.decoder_network.is_training_placeholder: False}

                    if self.labels == 'convo-semi-supervised':
                        summary_data[self.encoder_labels] = batch_y[:, np.newaxis]
                        summary_data[self.decoder_labels] = batch_y[:, np.newaxis]

                    sum_res_encoder = self.session.run(merged_summary, summary_data)
                    sum_res_decoder = self.session.run(merged_summary, summary_data)
                    self.encoder_network.train_writer.add_summary(
                        sum_res_encoder, epoch_idx * sample_per_epoch + sample_iter)
                    self.decoder_network_train.train_writer.add_summary(
                        sum_res_decoder, epoch_idx * sample_per_epoch + sample_iter)
            if save_model:
                self.encoder_network.save(global_step=epoch_idx * sample_per_epoch)
                self.decoder_network_train.save(global_step=epoch_idx * sample_per_epoch)