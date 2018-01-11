import os
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
            dsc_input_size_final = list(discriminator_input_size)
            dsc_input_size_final[-1] += labels_size
            input_dsc_size = [None, ] + self.discriminator_input_size
            input_dsc_placeholder = tf.placeholder(tf.float32, input_dsc_size, name="dsc_x")

            # Define placeholder for labels
            labels_size_t = [None, labels_size]
            labels_placeholder = tf.placeholder(tf.float32, labels_size_t, name='conditional_placeholder')

            # Concat labels with generator input
            new_gen_placeholder = tf.concat(axis=1, values=[input_gen_placeholder, labels_placeholder])

            # Concat labels with discriminator input
            labels_y1_size = [self.batch_size] + [1 for _ in range(len(input_dsc_size) - 2)] + [labels_size]
            labels_y1 = tf.reshape(labels_placeholder, shape=labels_y1_size)
            ones_size = [self.batch_size] + list(input_dsc_size)[1:-1] + [labels_size]
            new_dsc_placeholder = tf.concat(axis=-1, values=[input_dsc_placeholder,
                                                             labels_y1*tf.ones(ones_size)])

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
        dsc_final_shape = list(self.discriminator_input_size)
        dsc_fake_input_placeholder = self.generator.layer_outputs[-1]
        if self.labels == "convo-semi-supervised":
            labels_y1_size = [-1] + [1 for _ in range(len(dsc_final_shape) - 1)] + [self.labels_size]
            labels_y1 = tf.reshape(self.labels_placeholder, shape=labels_y1_size)
            ones_size = [self.batch_size] + list(dsc_final_shape)[:-1] + [self.labels_size]
            dsc_fake_input_placeholder = tf.concat(axis=-1, values=[dsc_fake_input_placeholder,
                                                                    labels_y1*tf.ones(ones_size)])
            dsc_final_shape = list(self.discriminator_input_size)
            dsc_final_shape[-1] += self.labels_size
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
        elif optimizer_name == "RMSProp":
            return rmsprop_optimizer
        elif optimizer_name == "SGD":
            return sgd_optimizer
        else:
            raise AttributeError("Optimizer {}, {} not defined.".format(self.generator_optimizer,
                                                                        self.discriminator_optimizer))

    def set_loss(self, loss, **kwargs):
        self.gan_model_loss = loss
        self.gan_model_loss_data = kwargs

    def restore(self):
        try:
            self.discriminator_fake.restore()
            self.discriminator.restore()
            self.generator.restore()
            Messenger.text("Model successful restored.")
        except Exception:
            Messenger.text("No restore points in {}".format(self.log_path))

    @abstractmethod
    def train(self, train_iter, generator_steps=1, discriminator_steps=1, train_step=100, epochs=1000,
              sample_per_epoch=1000, summary_step=50, reshape_input=None, save_model=True, restore_model=True):
        pass
