import os
import sys
import argparse
import tensorflow as tf
import json

from dataset import Dataset
from model import Model
import prune
import check
import train
import test
import approximate_isometry


def parse_arguments():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--nruns', type=int, default=1, help='the number of times to run the program')
    # Data
    parser.add_argument('--path_data', type=str, default='path_to_datasets', help='location of data sets')
    parser.add_argument('--datasource', type=str, default='mnist', help='data set to use')
    parser.add_argument('--aug_kinds', nargs='+', type=str, default=[], help='augmentation kinds to perform')
    # Model
    parser.add_argument('--arch', type=str, default='mlp-7-linear', help='model architecture')
    # Initialization
    parser.add_argument('--init_w', type=json.loads, default={'kind': 'orthogonal'}, help='initializer for w')
    parser.add_argument('--init_b', type=json.loads, default={'kind': 'zeros'}, help='initializer for b')
    # Pruning
    parser.add_argument('--target_sparsity', type=float, default=0.9, help='target sparsity')
    parser.add_argument('--datasource_pruning', type=str, default='mnist', help='data set to use for transfer pruning')
    parser.add_argument('--transfer_pruning', action='store_true', help='use separate datasource for pruning')
    # Train
    parser.add_argument('--batch_size', type=int, default=100, help='number of examples in the mini-batch')
    parser.add_argument('--train_iterations', type=int, default=10000, help='number of training iterations')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer of choice')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('--decay_type', type=str, default='constant', help='learning rate decay type')
    parser.add_argument('--decay_boundaries', nargs='+', type=int, default=[], help='boundaries for piecewise_constant decay')
    parser.add_argument('--decay_values', nargs='+', type=float, default=[], help='values for piecewise_constant decay')
    parser.add_argument('--check_interval', type=int, default=100, help='check interval during training')
    parser.add_argument('--save_interval', type=int, default=1000, help='save interval during training')
    # Test
    parser.add_argument('--num_eval_checkpoints', type=int, default=20, help='number of checkpoints to evaluate')
    parser.add_argument('--no_load_cache', action='store_true', help='do not allow loading cache in test mode')
    # Dynamical isometry
    parser.add_argument('--check_jsv', action='store_true', help='check jacobian singular values')
    parser.add_argument('--enforce_isometry', action='store_true', help='enforce approximate dynamical isometry')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # Multiple runs
    for run in range(args.nruns):

        # Start
        print('--\nStart run ({})'.format(run))

        # Set paths
        path_save = 'run-{}'.format(run)
        path_keys = ['model', 'log', 'assess']
        args.path = {key: os.path.join(path_save, key) for key in path_keys}

        # Reset the default graph and set a graph-level seed
        tf.reset_default_graph()
        tf.set_random_seed(seed=run)

        # Dataset
        dataset = Dataset(**vars(args))
        if args.transfer_pruning:
            dataset_pruning = Dataset(args.datasource_pruning, args.path_data)

        # Model
        model = Model(**vars(args))
        model.construct_model()

        # Session
        sess = tf.InteractiveSession()

        # Initialization
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        _ = sess.run([model.weights_init], {model.init: True})
        if args.check_jsv:
            check.jacobian_singular_value(args, model, sess, dataset, 'after-init')

        # Prune
        prune.prune(args, model, sess, dataset_pruning if args.transfer_pruning else dataset)
        if args.check_jsv:
            check.jacobian_singular_value(args, model, sess, dataset, 'after-prune')

        # Enforce approximate dynamical isometry in the sparse network
        if args.enforce_isometry:
            approximate_isometry.optimize(args, model, sess, dataset)
            if args.check_jsv:
                check.jacobian_singular_value(args, model, sess, dataset, 'after-isometry')

        # Train and test
        train.train(args, model, sess, dataset)
        test.test(args, model, sess, dataset)

        # Closing
        sess.close()
        print('--\nFinish run ({})'.format(run))

    sys.exit()


if __name__ == "__main__":
    main()
