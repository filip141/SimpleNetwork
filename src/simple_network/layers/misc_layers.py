import tensorflow as tf
from simple_network.layers.layers import Layer


class ImageSplitterLayer(Layer):

    def __init__(self, name='image_splitter', ksize=160, summaries=True, reuse=None):
        super(ImageSplitterLayer, self).__init__("ImageSplitterLayer", name, 'image_splitter', summaries, reuse)
        # Define layer properties
        self.layer_input = None
        self.output = None
        self.ksize = ksize
        self.layer_size = None

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        self.layer_size = self.input_shape
        img_channels = self.layer_input.get_shape().as_list()[-1]
        with tf.variable_scope(self.layer_name):
            img_patches = tf.extract_image_patches(images=self.layer_input, ksizes=[1, self.ksize, self.ksize, 1],
                                                   strides=[1, self.ksize, self.ksize, 1],
                                                   rates=[1, 1, 1, 1], padding='SAME')
            patches_shape = img_patches.get_shape().as_list()
            patches_list = []
            for wp_idx in range(0, patches_shape[1]):
                for hp_idx in range(0, patches_shape[2]):
                    tmp_img = tf.reshape(img_patches[:, wp_idx, hp_idx, :], (-1, self.ksize, self.ksize, img_channels))
                    patches_list.append(tmp_img)
            if self.save_summaries:
                for path_idx, patch in enumerate(patches_list):
                    tf.summary.image('split_im_{}'.format(path_idx), patch, 1)
            self.output = patches_list
            self.output_shape = patches_list[0].get_shape().as_list()
        return self.output


class SplitterLayer(Layer):

    def __init__(self, name='splitter', num_split=2, summaries=True, reuse=None):
        super(SplitterLayer, self).__init__("SplitterLayer", name, 'splitter', summaries, reuse)
        # Define layer properties
        self.layer_input = None
        self.output = None
        self.num_split = num_split
        self.layer_size = None

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        self.layer_size = self.input_shape
        with tf.variable_scope(self.layer_name):
            self.output = tf.split(axis=3, num_or_size_splits=self.num_split, value=self.layer_input)
            self.output_shape = self.output[0].get_shape().as_list()
        return self.output


class ReshapeLayer(Layer):

    def __init__(self, output_shape, name='reshaper', summaries=True, reuse=None):
        super(ReshapeLayer, self).__init__("ReshapeLayer", name, 'reshaper', summaries, reuse)
        # Define layer properties
        self.layer_input = None
        self.input_shape = None
        self.output_shape = output_shape
        self.output = None
        self.layer_size = None

    def build_graph(self, layer_input):
        self.layer_input = layer_input
        self.input_shape = self.layer_input.get_shape().as_list()[1:]
        self.layer_size = self.input_shape
        with tf.variable_scope(self.layer_name):
            self.output = tf.reshape(self.layer_input, [-1, ] + self.output_shape)
        return self.output