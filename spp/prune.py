import time


def prune(args, model, sess, dataset):
    print('-- Prune network')
    t_start = time.time()
    generator = dataset.get_generator('epoch', 'train', True)
    batch_size = 100
    nitr = dataset.num_example['train'] // batch_size
    remainder = dataset.num_example['train'] % batch_size
    if remainder > 0:
        nitr += 1
    for b in range(nitr):
        batch = dataset.get_next_batch(batch_size, generator)
        feed_dict = {}
        feed_dict.update({model.inputs[key]: batch[key] for key in ['image', 'label']})
        feed_dict.update({model.accumulate_g: True})
        feed_dict.update({model.compress: True if b == nitr - 1 else False})
        input_tensors = [model.sparsity]
        result = sess.run(input_tensors, feed_dict)
    print('sparsity: {:.3f} (t:{:.1f})'.format(result[-1], time.time() - t_start))
