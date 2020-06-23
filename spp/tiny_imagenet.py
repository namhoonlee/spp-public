import os
import glob
import cv2

import numpy as np


def read_data_sets(path_dataset):
    # global label txt
    label_txt = os.path.join(path_dataset, 'wnids.txt')
    with open(label_txt, 'r') as f:
        labels = f.readlines()
    labels = [label.rstrip('\n') for label in labels]
    labels_dict = {}
    for i, label in enumerate(labels):
        labels_dict[label] = i

    # train images and labels
    train_images, train_labels = [], []
    path_dataset_train = os.path.join(path_dataset, 'train')
    for labelname in os.listdir(path_dataset_train):
        files = glob.glob(os.path.join(*[path_dataset_train, labelname, 'images', '*.JPEG']))
        for f in files:
            train_images.append(cv2.imread(f))
            train_labels.append(labels_dict[labelname])
    train_images = np.asarray(train_images)
    train_labels = np.asarray(train_labels)

    # read val annotation file
    path_dataset_val = os.path.join(path_dataset, 'val')
    val_annotations_dict = {}
    with open(os.path.join(path_dataset_val, 'val_annotations.txt'), 'r') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            val_annotations_dict[fields[0]] = fields[1]

    # test images and labels
    test_images, test_labels = [], []
    files = glob.glob(os.path.join(*[path_dataset_val, 'images', '*.JPEG']))
    for f in files:
        test_images.append(cv2.imread(f))
        path, f_nopath = os.path.split(f)
        test_labels.append(labels_dict[val_annotations_dict[f_nopath]])
    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)

    # Pre-processing (normalize)
    train_images = np.divide(train_images, 255, dtype=np.float32)
    test_images = np.divide(test_images, 255, dtype=np.float32)
    channel_mean = np.mean(train_images, axis=(0,1,2), dtype=np.float32, keepdims=True)
    channel_std = np.std(train_images, axis=(0,1,2), dtype=np.float32, keepdims=True)
    train_images = (train_images - channel_mean) / channel_std
    test_images = (test_images - channel_mean) / channel_std

    dataset = {
        'train': {'image': train_images, 'label': train_labels},
        'test': {'image': test_images, 'label': test_labels},
    }
    return dataset
