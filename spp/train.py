import os
import tensorflow as tf
import time
import numpy as np
import pickle

from augment import augment


def train(args, model, sess, dataset):
    print('-- Train')

    for key in ['model', 'log']:
        if not os.path.isdir(args.path[key]):
            os.makedirs(args.path[key])
    saver = tf.train.Saver()
    random_state = np.random.RandomState(seed=0)
    logs = {'train': {'itr': [], 'los': [], 'acc': []}, 'val': {'itr': [], 'los': [], 'acc': []}}
    t_start = time.time()

    generators = {}
    generators['train'] = dataset.get_generator('unlimited', 'train', True)
    generators['val'] = dataset.get_generator('unlimited', 'val', True)

    for itr in range(args.train_iterations):
        batch = dataset.get_next_batch(args.batch_size, generators['train'])
        batch = augment(batch, args.aug_kinds, random_state)
        feed_dict = {}
        feed_dict.update({model.inputs[key]: batch[key] for key in ['image', 'label']})
        feed_dict.update({model.is_train: True})
        input_tensors = [model.outputs]
        if (itr+1) % args.check_interval == 0:
            input_tensors.extend([model.sparsity])
        input_tensors.extend([model.train_op])
        result = sess.run(input_tensors, feed_dict)
        logs['train'] = _update_logs(logs['train'],
            {'itr': itr+1, 'los': result[0]['los'], 'acc': result[0]['acc']})

        # Check on validation set.
        if (itr+1) % args.check_interval == 0:
            batch = dataset.get_next_batch(args.batch_size, generators['val'])
            batch = augment(batch, args.aug_kinds, random_state)
            feed_dict = {}
            feed_dict.update({model.inputs[key]: batch[key] for key in ['image', 'label']})
            input_tensors = [model.outputs]
            result_val = sess.run(input_tensors, feed_dict)
            logs['val'] = _update_logs(logs['val'],
                {'itr': itr+1, 'los': result_val[0]['los'], 'acc': result_val[0]['acc']})

        # Print results
        if (itr+1) % args.check_interval == 0:
            pstr = '(train/val) los:{:.3f}/{:.3f} acc:{:.3f}/{:.3f} spa:{:.3f}'.format(
                result[0]['los'], result_val[0]['los'],
                result[0]['acc'], result_val[0]['acc'],
                result[1],
            )
            print('itr{}: {} (t:{:.1f})'.format(itr+1, pstr, time.time() - t_start))
            t_start = time.time()

        # Save
        if (itr+1) % args.save_interval == 0:
            saver.save(sess, args.path['model'] + '/itr-' + str(itr))
            _save_logs(os.path.join(args.path['log'], 'train.pickle'), logs['train'])
            _save_logs(os.path.join(args.path['log'], 'val.pickle'), logs['val'])


def _update_logs(logs, log):
    for key in logs.keys():
        logs[key].extend([log[key]])
    return logs

def _save_logs(filename, results):
    with open(filename, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
