import os
import sys
import argparse

from glob import glob
import scipy.io as sio
from skimage.transform import resize
import imageio
from PIL import Image

import h5py
import numpy as np
import re

from lab_segmentation.configSegmenter import flying_objects_config


def image_dir_to_h5(dir_path, set, output_file):

    images = glob(os.path.join(dir_path, 'image', '*.png'))
    n_image = len(images)
    labels = {
        re.sub(r'gt_', '', os.path.basename(path)): path
        for path in glob(os.path.join(dir_path, 'gt_image', 'gt_*.png'))}
    n_labels = len(labels)
    assert n_labels == n_image, "numbers of image and ground truth labels are not the same>> image nbr: %d gt nbr: %d"  % (n_image, n_labels)


    cfg = flying_objects_config()


    output_file.create_dataset(set + '_x', (n_image, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.IMAGE_CHANNEL), np.float)
    output_file.create_dataset(set + '_y', (n_image, len(cfg.defaultClasses)), np.int)
    output_file.create_dataset(set + '_y_finegrained', (n_image, len(cfg.fineGrainedClasses)), np.int)
    output_file.create_dataset(set + '_y_segmentation', (n_image, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 4), np.float)

    for i, image_path in enumerate(images):
        sys.stdout.write("\rProcessing %i" % i)
        sys.stdout.flush()

        image = Image.open(image_path)
        image = image.resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH))
        image = np.asarray(image) / 255


        path, img_name = os.path.split(image_path)
        fn, ext = img_name.split(".")
        names = fn.split("_")

        labels_finegrained = np.zeros(shape=(len(cfg.fineGrainedClasses)), dtype=np.int)
        labels_default = np.zeros(shape=(len(cfg.defaultClasses)), dtype=np.int)

        label_finegrained = names[1] + "_" + names[2]
        label_default = names[1]

        if np.isin(label_finegrained, cfg.fineGrainedClasses):
            loc = cfg.fineGrainedClasses.index(label_finegrained)
            labels_finegrained[loc] = 1
        else:
            print("ERROR: Label " + str(label_finegrained) + " is not defined!")

        if np.isin(label_default, cfg.defaultClasses):
            loc = cfg.defaultClasses.index(label_default)
            labels_default[loc] = 1
        else:
            print("ERROR: Label " + str(label_default) + " is not defined!")

        output_file[set + '_x'][i, ...] = image
        output_file[set + '_y'][i, ...] = labels_default
        output_file[set + '_y_finegrained'][i, ...] = labels_finegrained

        #segmentation
        # read labels
        gt_image_file = labels[os.path.basename(image_path)]
        gt_image = Image.open(gt_image_file)
        gt_image = gt_image.resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH))
        gt_image = np.asarray(gt_image)

        # create background image
        bkgnd_image = 255 * np.ones((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH))
        bkgnd_image = bkgnd_image - gt_image[:, :, 0]
        bkgnd_image = bkgnd_image - gt_image[:, :, 1]
        bkgnd_image = bkgnd_image - gt_image[:, :, 2]
        gt_image = np.dstack((gt_image, bkgnd_image))
        output_file[set + '_y_segmentation'][i, ...] = gt_image


if __name__ == '__main__':
    training_data_dir = "data/FlyingObjectDataset_10K/training"
    validation_data_dir = "data/FlyingObjectDataset_10K/validation"
    testing_data_dir = "data/FlyingObjectDataset_10K/testing"

    output_file = h5py.File('data_new_no.hdf5', 'w')

    print("\n\n")
    image_dir_to_h5(training_data_dir, 'train', output_file)
    print("\n\n")
    image_dir_to_h5(validation_data_dir, 'val', output_file)
    print("\n\n")
    image_dir_to_h5(testing_data_dir, 'test', output_file)

    output_file.close()
