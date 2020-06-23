import os
import itertools
import numpy as np

import mnist
import fashion_mnist
import cifar
import tiny_imagenet


class Dataset(object):
    def __init__(self, datasource, path_data, **kwargs):
        self.datasource = datasource
        self.path_data = path_data
        self.rand = np.random.RandomState(seed=0)
        if self.datasource == 'mnist':
            self.dataset = mnist.read_data_sets(
                os.path.join(self.path_data, 'MNIST'),
            )
        elif self.datasource == 'fashion-mnist':
            self.dataset = fashion_mnist.read_data_sets(
                os.path.join(self.path_data, 'Fashion-MNIST'),
            )
        elif self.datasource == 'cifar-10':
            self.dataset = cifar.read_data_sets(
                os.path.join(self.path_data, 'cifar-10-batches-py'),
            )
        elif self.datasource == 'tiny-imagenet':
            self.dataset = tiny_imagenet.read_data_sets(
                os.path.join(self.path_data, 'tiny-imagenet-200'),
            )
        else:
            raise NotImplementedError
        self.split_dataset('train', 'val', int(self.dataset['train']['image'].shape[0] * 0.1))
        self.num_example = {k: self.dataset[k]['image'].shape[0] for k in self.dataset.keys()}

    def split_dataset(self, source, target, number):
        keys = ['image', 'label']
        indices = list(range(self.dataset[source]['image'].shape[0]))
        self.rand.shuffle(indices)
        ind_target = indices[:number]
        ind_remain = indices[number:]
        self.dataset[target] = {k: self.dataset[source][k][ind_target] for k in keys}
        self.dataset[source] = {k: self.dataset[source][k][ind_remain] for k in keys}

    def get_generator(self, mode, split, shuffle):
        if mode == 'epoch':
            epochs = range(1)
        elif mode == 'unlimited':
            epochs = itertools.count()
        else:
            raise ValueError
        for i in epochs:
            example_ids = list(range(self.num_example[split]))
            if shuffle:
                self.rand.shuffle(example_ids)
            for example_id in example_ids:
                yield {
                    'image': self.dataset[split]['image'][example_id],
                    'label': self.dataset[split]['label'][example_id],
                    'id': example_id,
                }

    def get_next_batch(self, batch_size, generator):
        images, labels, ids = [], [], []
        for i in range(batch_size):
            try:
                example = next(generator)
                images.append(example['image'])
                labels.append(example['label'])
                ids.append(example['id'])
            except StopIteration:
                break
        batch = None
        if images:
            batch = {
                'image': np.asarray(images),
                'label': np.asarray(labels),
                'id': np.asarray(ids),
            }
        return batch
