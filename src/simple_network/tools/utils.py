import os
import cv2
import random
import tensorflow as tf
import numpy as np

CIFAR10_LABELS = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5,
                  "frog": 6, "horse": 7, "ship": 8, "truck": 9}


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


class CIFARDataset(object):

    def __init__(self, data_path="./train"):
        self.data_path = data_path
        self.position = 0
        self.files_list = os.listdir(data_path)
        random.shuffle(self.files_list)

    def next_batch(self, number):
        first_file_name = self.files_list[self.position]
        _, first_label = os.path.split(first_file_name)[1].split(".")[0].split("_")
        first_path = os.path.join(self.data_path, first_file_name)

        #  Read image
        first_img = cv2.cvtColor(cv2.imread(first_path), cv2.COLOR_BGR2RGB)
        first_img_shape = first_img.shape
        first_one_hot_labels = np.zeros((10, ))
        batch_labels = np.zeros((number, 10))
        batch_matrix = np.zeros((number, first_img_shape[0], first_img_shape[1], first_img_shape[2]))
        first_one_hot_labels[CIFAR10_LABELS[first_label]] = 1.0
        batch_matrix[0] = (first_img.astype('float32') - np.min(first_img)) / np.std(first_img)
        batch_labels[0] = first_one_hot_labels
        self.position += 1
        self.position = self.position if self.position < len(self.files_list) else 0

        # iterate over elements
        for img_idx in range(1, number):
            img_name = self.files_list[self.position]
            _, img_label = os.path.split(img_name)[1].split(".")[0].split("_")
            img_path = os.path.join(self.data_path, img_name)
            ds_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            one_hot_labels = np.zeros((10, ))
            one_hot_labels[CIFAR10_LABELS[img_label]] = 1.0
            batch_matrix[img_idx] = (ds_img.astype('float32') - np.min(ds_img)) / \
                                    (np.max(ds_img) - np.min(ds_img))
            batch_labels[img_idx] = one_hot_labels
            self.position += 1
            self.position = self.position if self.position < len(self.files_list) else 0
        return batch_matrix, batch_labels


class LIVEDataset(object):

    def __init__(self, data_path="./train", new_resolution="320x320", patches=None,
                 patches_method='split', no_patches=32, is_train=True):
        self.data_path = data_path
        self.images_mos = []
        self.position = 0
        self.no_patches = no_patches
        self.pathes_method = patches_method
        self.is_train = is_train
        self.patches = patches
        self.new_resolution = new_resolution
        self.fastfading_path = os.path.join(data_path, "fastfading")
        self.gblur_path = os.path.join(data_path, "gblur")
        self.j2k_path = os.path.join(data_path, "jp2k")
        self.jpeg_path = os.path.join(data_path, "jpeg")
        self.wn_path = os.path.join(data_path, "wn")

        # Define resolution for image
        if new_resolution is None:
            self.res_tuple = None
        else:
            self.res_tuple = [int(x) for x in new_resolution.split("x")]

        # Define resolution for patches
        if patches is None:
            self.patch_res = None
        else:
            self.patch_res = [int(x) for x in patches.split("x")]

        # Fast fading
        fast_fading_info = open(os.path.join(self.fastfading_path, "info.txt"), 'r')
        for ff_line in fast_fading_info:
            splt_line = ff_line.split(" ")
            if len(splt_line) < 3:
                continue
            self.images_mos.append((os.path.join(self.fastfading_path, splt_line[1]), float(splt_line[2].rstrip())))
        # G-Blur
        gblur_info = open(os.path.join(self.gblur_path, "info.txt"), 'r')
        for gb_line in gblur_info:
            splt_line = gb_line.split(" ")
            if len(splt_line) < 3:
                continue
            self.images_mos.append((os.path.join(self.gblur_path, splt_line[1]), float(splt_line[2].rstrip())))
        # J2K Path
        j2k_info = open(os.path.join(self.j2k_path, "info.txt"), 'r')
        for j2k_line in j2k_info:
            splt_line = j2k_line.split(" ")
            if len(splt_line) < 3:
                continue
            self.images_mos.append((os.path.join(self.j2k_path, splt_line[1]), float(splt_line[2].rstrip())))
        # J2K Path
        jpeg_info = open(os.path.join(self.jpeg_path, "info.txt"), 'r')
        for jpeg_line in jpeg_info:
            splt_line = jpeg_line.split(" ")
            if len(splt_line) < 3:
                continue
            self.images_mos.append((os.path.join(self.jpeg_path, splt_line[1]), float(splt_line[2].rstrip())))
        # J2K Path
        wn_info = open(os.path.join(self.wn_path, "info.txt"), 'r')
        for wn_line in wn_info:
            splt_line = wn_line.split(" ")
            if len(splt_line) < 3:
                continue
            self.images_mos.append((os.path.join(self.wn_path, splt_line[1]), float(splt_line[2].rstrip())))
        random.shuffle(self.images_mos)

        # Divide into test and train
        test_len = int(0.2 * len(self.images_mos))
        self.train_mos = self.images_mos[test_len:]
        self.test_mos = self.images_mos[:test_len]

    def next_batch(self, number):
        if self.is_train:
            img_base = self.train_mos
        else:
            img_base = self.test_mos
        batch_labels = np.zeros((number, 1))
        if self.patches is None:
            batch_matrix = np.zeros((number, self.res_tuple[0], self.res_tuple[1], 3))
        else:
            batch_matrix = np.zeros((number, self.patch_res[0], self.patch_res[1], 3))

        # iterate over elements
        img_idx = 0
        while img_idx < number:
            im_path, im_mos = img_base[self.position]
            live_img = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
            if self.new_resolution is None:
                live_img_trans = live_img
            else:
                live_img_trans = cv2.resize(live_img, tuple(self.res_tuple), interpolation=cv2.INTER_AREA)
            img_shape = live_img_trans.shape

            # Return image or patches
            if self.patches is None:
                batch_matrix[img_idx] = live_img_trans
                batch_labels[img_idx] = im_mos
                img_idx += 1
            else:
                n_patches_w = int(img_shape[1] / float(self.patch_res[0]))
                n_patches_h = int(img_shape[0] / float(self.patch_res[1]))
                if self.pathes_method == 'split':
                    # Create patches
                    for im_x in range(n_patches_h):
                        for im_y in range(n_patches_w):
                            img = live_img_trans[
                                  im_x * self.patch_res[0]: (im_x + 1) * self.patch_res[0],
                                  im_y * self.patch_res[1]: (im_y + 1) * self.patch_res[1]
                                  ]
                            batch_matrix[img_idx] = img
                            batch_labels[img_idx] = im_mos
                            img_idx += 1
                            if img_idx > number:
                                return batch_matrix, batch_labels
                elif self.pathes_method == 'random':
                    for p_idx in range(0, self.no_patches):
                        w_pos = random.randint(0, img_shape[1] - self.patch_res[0])
                        h_pos = random.randint(0, img_shape[0] - self.patch_res[1])
                        img = live_img_trans[h_pos:h_pos + self.patch_res[1], w_pos:w_pos + self.patch_res[0]]
                        batch_matrix[img_idx] = img
                        batch_labels[img_idx] = im_mos
                        img_idx += 1
                        if img_idx >= number:
                            return batch_matrix, batch_labels
                else:
                    raise AttributeError("Method for splitting patches not defined.")
            self.position += 1
            self.position = self.position if self.position < len(img_base) else 0
        return batch_matrix, batch_labels