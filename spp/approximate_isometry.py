import time


def optimize(args, model, sess, dataset):
    print('-- Enforce approximate dynamical isometry on sparse network')
    t_start = time.time()
    generator = dataset.get_generator('unlimited', 'train', True)
    for itr in range(10000):
        feed_dict = {}
        batch = dataset.get_next_batch(1, generator) # ideally no need to pass data
        feed_dict.update({model.inputs[key]: batch[key] for key in ['image', 'label']})
        input_tensors = [model.isometry_penalty_op]
        result = sess.run(input_tensors, feed_dict)
    print('finished (t:{:.1f})'.format(time.time() - t_start))
