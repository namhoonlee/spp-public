import time
import numpy as np


def jacobian_singular_value(args, model, sess, dataset, flag):
    print('-- Check Jacobian singular values [{}]'.format(flag))
    t_start = time.time()
    jsv = []
    generator = dataset.get_generator('epoch', 'train', True)
    batch_size = 10
    nitr = dataset.num_example['train'] // batch_size
    remainder = dataset.num_example['train'] % batch_size
    if remainder > 0:
        nitr += 1
    for b in range(nitr):
        batch = dataset.get_next_batch(batch_size, generator)
        feed_dict = {}
        feed_dict.update({model.inputs[key]: batch[key] for key in ['image', 'label']})
        input_tensors = [model.jsv]
        result = sess.run(input_tensors, feed_dict)
        jsv.append(result[-1])
    jsv_all = np.concatenate(jsv)
    jsv_stat = [func(jsv_all) for func in [np.mean, np.std, np.max, np.min]]
    print('Jacobian Singular Value (mean/std/max/min) (t:{:.1f}):'.format(time.time() - t_start))
    print(jsv_stat)
