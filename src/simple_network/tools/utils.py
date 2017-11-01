import os
import cv2
import random
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def variable_summaries(var, name):
    with tf.name_scope('summaries_{}'.format(name)):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram(name, var)


def img_patch_spliter(im_path, patch_num=32, patch_res="32x32", patches_method="split"):
    patch_res_tpl = [int(x) for x in patch_res.split("x")]
    batch_matrix = np.zeros((patch_num, patch_res_tpl[0], patch_res_tpl[1], 3))

    # Read image from patch
    live_img = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    img_shape = live_img.shape

    # Return image or patches
    n_patches_w = int(img_shape[1] / float(patch_res_tpl[0]))
    n_patches_h = int(img_shape[0] / float(patch_res_tpl[1]))
    # iterate over elements
    img_idx = 0
    while img_idx < patch_num:
        if patches_method == 'split':
            # Create patches
            for im_x in range(n_patches_h):
                for im_y in range(n_patches_w):
                    img = live_img[
                          im_x * patch_res_tpl[0]: (im_x + 1) * patch_res_tpl[0],
                          im_y * patch_res_tpl[1]: (im_y + 1) * patch_res_tpl[1]
                          ]
                    batch_matrix[img_idx] = img
                    img_idx += 1
                    if img_idx >= patch_num:
                        return batch_matrix
        elif patches_method == 'random':
            for p_idx in range(0, patch_num):
                w_pos = random.randint(0, img_shape[1] - patch_res_tpl[0])
                h_pos = random.randint(0, img_shape[0] - patch_res_tpl[1])
                img = live_img[h_pos:h_pos + patch_res_tpl[1], w_pos:w_pos + patch_res_tpl[0]]
                batch_matrix[img_idx] = img
                img_idx += 1
                if img_idx >= patch_num:
                    return batch_matrix
        else:
            raise AttributeError("Method for splitting patches not defined.")
    return batch_matrix


def create_sprite_image(dataset_iterator, log_path, img_num):
    images, labels = dataset_iterator.next_batch(img_num)
    img_h = images.shape[1]
    img_w = images.shape[2]
    img_c = images.shape[3]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    sprite_image = np.ones((img_h * n_plots, img_w * n_plots, img_c))
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                sprite_image[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w] = this_img

    # Create metadata file
    meta_temp_path = os.path.join(log_path, "labels.tsv")
    with open(meta_temp_path, 'w') as meta_f:
        for index, label in enumerate(labels):
            new_label = np.argmax(label)
            meta_f.write('{}\n'.format(new_label))
    img_temp_path = os.path.join(log_path, "sprite.png")

    plt.imsave(img_temp_path, sprite_image)
    return img_temp_path, meta_temp_path, images, labels


class Messenger(object):

    def __init__(self):
        pass

    @staticmethod
    def text(message):
        logger.info(message)

    @staticmethod
    def section_message(message):
        logger.info(message)
        logger.info("-" * 90)

    @staticmethod
    def title_message(message):
        logger.info("=" * 90)
        logger.info(message)
        logger.info("=" * 90)