import os
import argparse

from glob import glob
import scipy.io as sio
from skimage.transform import resize
import imageio

import h5py
import numpy as np

from lab_classification.configClassifier import flying_objects_config


def image_dir_to_h5(dir_path, set, output_file):
    images = glob(os.path.join(dir_path, 'image', '*.png'))
    n_image = len(images)
    cfg = flying_objects_config()
    set_shape = (n_image, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.IMAGE_CHANNEL)


    output_file.create_dataset(set + '_x', set_shape, np.float)
    output_file.create_dataset(set + '_y', (n_image, len(cfg.CLASSES)), np.float)


    for i, image_path in enumerate(images):
        print("\rProcessing %i" % i)
        image = resize(imageio.imread(image_path), set_shape[1:])


        labels = np.zeros(shape=(len(cfg.CLASSES)), dtype=np.float32)
        path, img_name = os.path.split(image_path)
        fn, ext = img_name.split(".")
        names = fn.split("_")
        currLabel = names[1] + "_" + names[2]
        if np.isin(currLabel, cfg.CLASSES):
            loc = cfg.CLASSES.index(currLabel)
            labels[loc] = 1
        else:
            print("ERROR: Label " + str(currLabel) + " is not defined!")

        output_file[set + '_x'][i, ...] = image
        output_file[set + '_y'][i, ...] = labels



if __name__ == '__main__':
    training_data_dir = "data/FlyingObjectDataset_10K/training"
    validation_data_dir = "data/FlyingObjectDataset_10K/validation"
    testing_data_dir = "data/FlyingObjectDataset_10K/testing"

    output_file = h5py.File('data.hdf5', 'w')

    image_dir_to_h5(training_data_dir, 'train', output_file)
    image_dir_to_h5(validation_data_dir, 'val', output_file)
    image_dir_to_h5(testing_data_dir, 'test', output_file)


    output_file.close()