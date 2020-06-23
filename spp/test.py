import os
import tensorflow as tf
import numpy as np
import glob
import operator

from helpers import cache_json


def test(args, model, sess, dataset):
    print('-- Test')

    saver = tf.train.Saver()

    # Identify which checkpoints are available.
    all_model_checkpoint_paths = glob.glob(os.path.join(args.path['model'], 'itr-*'))
    all_model_checkpoint_paths = list(set([f.split('.')[0] for f in all_model_checkpoint_paths]))
    model_files = {int(s[s.index('itr')+4:]): s for s in  all_model_checkpoint_paths}

    # Subset of iterations.
    itrs = sorted(model_files.keys())
    itr_subset = itrs
    assert itr_subset

    # Evaluate.
    acc = []
    for itr in itr_subset:
        print('evaluation: {} | itr-{}'.format(dataset.datasource, itr))
        # run evaluate and/or cache
        result = cache_json(
            os.path.join(args.path['assess'], 'itr-{}.json'.format(itr)),
            lambda: _evaluate(model, saver, model_files[itr], sess, dataset),
            makedir=True,
            allow_load=(not args.no_load_cache))
        # print
        print('Accuracy: {:.5f} (#examples:{})'.format(result['accuracy'], result['num_example']))
        acc.append(result['accuracy'])
    print('Max: {:.5f}, Min: {:.5f} (#Eval: {})'.format(max(acc), min(acc), len(acc)))
    print('Error: {:.3f} %'.format((1 - max(acc))*100))


def _evaluate(model, saver, model_file, sess, dataset):
    # load model
    if saver is not None and model_file is not None:
        saver.restore(sess, model_file)
    else:
        raise FileNotFoundError

    # load test set
    generator = dataset.get_generator('epoch', 'test', False)

    # run
    accuracy = []
    while True:
        batch = dataset.get_next_batch(100, generator)
        if batch is not None:
            feed_dict = {}
            feed_dict.update({model.inputs[key]: batch[key] for key in ['image', 'label']})
            feed_dict.update({model.compress: False, model.is_train: False})
            result = sess.run([model.outputs], feed_dict)
            accuracy.extend(result[0]['acc_individual'])
        else:
            break

    results = { # has to be JSON serialiazable
        'accuracy': np.mean(accuracy).astype(float),
        'num_example': len(accuracy),
    }
    assert results['num_example'] == dataset.num_example['test']
    return results
