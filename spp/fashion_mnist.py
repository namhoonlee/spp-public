import os
import numpy as np
import gzip


def read_data_sets(path_dataset, one_hot=False):
    TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
    TRAIN_LABELS = "train-labels-idx1-ubyte.gz"
    TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
    TEST_LABELS = "t10k-labels-idx1-ubyte.gz"
    
    train_label_path = os.path.join(path_dataset, TRAIN_LABELS)
    with gzip.open(train_label_path, 'rb') as path:
        train_labels = np.frombuffer(path.read(), dtype=np.uint8, offset=8)

    train_image_path = os.path.join(path_dataset, TRAIN_IMAGES)
    with gzip.open(train_image_path, 'rb') as path:
        train_images = np.frombuffer(path.read(), dtype=np.uint8, offset=16).reshape(
            len(train_labels), 28, 28, 1)

    test_label_path = os.path.join(path_dataset, TEST_LABELS)
    with gzip.open(test_label_path, 'rb') as path:
        test_labels = np.frombuffer(path.read(), dtype=np.uint8, offset=8)

    test_image_path = os.path.join(path_dataset, TEST_IMAGES)
    with gzip.open(test_image_path, 'rb') as path:
        test_images = np.frombuffer(path.read(), dtype=np.uint8, offset=16).reshape(
            len(test_labels), 28, 28, 1)

    # Pre-processing (normalize)
    train_images = np.divide(train_images, 255, dtype=np.float32)
    test_images = np.divide(test_images, 255, dtype=np.float32)
    mean = np.mean(train_images)
    std = np.std(train_images)
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std

    dataset = {
        'train': {'image': train_images, 'label': train_labels},
        'test': {'image': test_images, 'label': test_labels},
    }
    return dataset
