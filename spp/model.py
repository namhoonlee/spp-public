import tensorflow as tf
import functools
import numpy as np

from functools import reduce
from network import load_network
from helpers import static_size

dtype = tf.float32

class Model(object):
    def __init__(self,
                 datasource,
                 arch,
                 target_sparsity,
                 optimizer,
                 learning_rate,
                 decay_type,
                 decay_boundaries,
                 decay_values,
                 init_w,
                 init_b,
                 enforce_isometry,
                 **kwargs):
        self.datasource = datasource
        self.arch = arch
        self.target_sparsity = target_sparsity
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.decay_type = decay_type
        self.decay_boundaries = decay_boundaries
        self.decay_values = decay_values
        self.init_w = init_w
        self.init_b = init_b
        self.enforce_isometry = enforce_isometry

    def construct_model(self, inputs=None):
        # Base-learner
        self.net = net = load_network(**{
            'arch': self.arch, 'datasource': self.datasource,
            'init_w': self.init_w, 'init_b': self.init_b,
        })

        # Input nodes
        self.inputs = net.inputs
        self.compress = tf.placeholder_with_default(False, [])
        self.accumulate_g = tf.placeholder_with_default(False, [])
        self.is_train = tf.placeholder_with_default(False, [])
        self.init = tf.placeholder_with_default(False, [])

        # For convenience
        prn_keys = [k for p in ['w', 'b'] for k in net.weights.keys() if p in k]
        var_no_train = functools.partial(tf.Variable, trainable=False, dtype=tf.float32)
        self.weights = weights = net.weights

        # Initialize weights, if enabled.
        weights_reinit = tf.cond(self.init,
            lambda: reinitialize(weights, self.init_w, self.init_b),
            lambda: weights)
        with tf.control_dependencies(
                [tf.assign(weights[k], weights_reinit[k]) for k in weights.keys()]):
            self.weights_init = {k: tf.identity(v) for k, v in weights.items()}

        # Pruning
        mask_init = {k: var_no_train(tf.ones(weights[k].shape)) for k in prn_keys}
        mask_prev = {k: var_no_train(tf.ones(weights[k].shape)) for k in prn_keys}
        g_mmt_prev = {k: var_no_train(tf.zeros(weights[k].shape)) for k in prn_keys}
        cs_prev = {k: var_no_train(tf.zeros(weights[k].shape)) for k in prn_keys}
        def update_g_mmt():
            w_mask = apply_mask(weights, mask_init)
            logits = net.forward_pass(w_mask, self.inputs['image'],
                self.is_train, trainable=False)
            loss = tf.reduce_mean(compute_loss(self.inputs['label'], logits))
            grads = tf.gradients(loss, [mask_init[k] for k in prn_keys])
            gradients = dict(zip(prn_keys, grads))
            g_mmt = {k: g_mmt_prev[k] + gradients[k] for k in prn_keys}
            return g_mmt
        g_mmt = tf.cond(self.accumulate_g, lambda: update_g_mmt(), lambda: g_mmt_prev)
        def get_sparse_mask():
            cs = normalize_dict({k: tf.abs(g_mmt[k]) for k in prn_keys})
            return (create_sparse_mask(cs, self.target_sparsity), cs)
        with tf.control_dependencies([tf.assign(g_mmt_prev[k], g_mmt[k]) for k in prn_keys]):
            mask, cs = tf.cond(self.compress,
                lambda: get_sparse_mask(), lambda: (mask_prev, cs_prev))
        with tf.control_dependencies([tf.assign(mask_prev[k], v) for k,v in mask.items()]):
            w_final = apply_mask(weights, mask)

        # Forward pass
        logits = net.forward_pass(w_final, self.inputs['image'], self.is_train)

        # Loss
        loss_emp = tf.reduce_mean(compute_loss(self.inputs['label'], logits))
        reg = 0.00025 * tf.reduce_sum([tf.reduce_sum(tf.square(v)) for v in w_final.values()])
        loss_opt = loss_emp + reg

        # Optimization
        optim, learning_rate, global_step = prepare_optimization(self.optimizer,
            self.learning_rate, self.decay_type, self.decay_boundaries, self.decay_values)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optim.minimize(loss_opt, global_step=global_step)

        # Outputs
        output_class = tf.argmax(logits, axis=1, output_type=tf.int32)
        output_correct_prediction = tf.equal(self.inputs['label'], output_class)
        output_accuracy_individual = tf.cast(output_correct_prediction, tf.float32)
        output_accuracy = tf.reduce_mean(output_accuracy_individual)
        self.outputs = {
            'los': loss_opt,
            'acc': output_accuracy,
            'acc_individual': output_accuracy_individual,
        }

        # Sparsity
        self.sparsity = compute_sparsity(w_final, prn_keys)

        # Approximate dynamical isometry
        self.isometry_penalty_op = compute_isometry(w_final)

        # Jacobian singular values
        self.jsv = compute_jsv(self.inputs['image'], logits)


def reinitialize(w, init_w, init_b):
    ''' Re-initialize weights.

    Initialization functions are taken from the TF code:
        https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/init_ops.py
    '''

    def random_sample(shape, initializer):
        seed = None
        kind = initializer['kind']
        hyperparams = {k: v for k, v in initializer.items() if k != 'kind'}

        if kind == 'zeros':
            sample = tf.zeros(shape, dtype)
        elif kind == 'normal':
            sample = tf.random_normal(shape, 0.0, 1.0, dtype, seed=seed)
        elif kind == 'normal_custom':
            # `untruncated_normal` is the special case when scale = 1.0
            scale = hyperparams['scale']
            stddev = tf.sqrt(scale)
            sample = tf.random_normal(shape, 0.0, stddev, dtype, seed=seed)
        elif kind == 'normal_lecun':
            scale = 1.0
            fan_in, fan_out = compute_fans(shape)
            scale /= max(1., fan_in)
            stddev = tf.sqrt(scale)
            sample = tf.random_normal(shape, 0.0, stddev, dtype, seed=seed)
        elif kind == 'normal_glorot':
            scale = 1.0
            fan_in, fan_out = compute_fans(shape)
            scale /= max(1., (fan_in + fan_out) / 2.)
            stddev = tf.sqrt(scale)
            sample = tf.random_normal(shape, 0.0, stddev, dtype, seed=seed)
        elif kind == 'normal_he':
            scale = 2.0
            fan_in, fan_out = compute_fans(shape)
            scale /= max(1., fan_in)
            stddev = tf.sqrt(scale)
            sample = tf.random_normal(shape, 0.0, stddev, dtype, seed=seed)
        elif kind == 'vs_custom':
            scale = hyperparams['scale']
            mode = hyperparams['mode']
            distribution = hyperparams['distribution']
            fan_in, fan_out = compute_fans(shape)
            if mode == "fan_in":
                scale /= max(1., fan_in)
            elif mode == "fan_out":
                scale /= max(1., fan_out)
            elif mode == 'fan_avg':
                scale /= max(1., (fan_in + fan_out) / 2.)
            else:
                raise NotImplementedError
            assert distribution == 'untruncated_normal' # currently only allow untruncated normal.
            stddev = tf.sqrt(scale)
            sample = tf.random_normal(shape, 0.0, stddev, dtype, seed=seed)
        elif kind == 'orthogonal':
            gain = 1.0
            if len(shape) < 2:
                raise ValueError("The tensor to initialize must be at least two-dimensional")
            # Flatten the input shape with the last dimension remaining
            # its original shape so it works for conv2d
            num_rows = 1
            for dim in shape[:-1]:
                num_rows *= dim
            num_cols = shape[-1]
            flat_shape = (num_cols, num_rows) if num_rows < num_cols else (num_rows, num_cols)

            # Generate a random matrix
            a = tf.random_normal(flat_shape, dtype=dtype, seed=seed)
            # Compute the qr factorization
            q, r = tf.qr(a, full_matrices=False)
            # Make Q uniform
            d = tf.diag_part(r)
            q *= tf.sign(d)
            if num_rows < num_cols:
                q = tf.matrix_transpose(q)
            sample = gain * tf.reshape(q, shape)
        elif kind == 'conv_orthogonal':
            def _dict_to_tensor(x, k1, k2):
                """Convert a dictionary to a tensor.
                Args:
                    x: A k1 * k2 dictionary.
                    k1: First dimension of x.
                    k2: Second dimension of x.
                Returns:
                    A k1 * k2 tensor.
                """

                return tf.stack([tf.stack([x[i, j] for j in range(k2)]) for i in range(k1)])

            def _block_orth(p1, p2):
                """Construct a 2 x 2 kernel. Used to construct orthgonal kernel.
                Args:
                  p1: A symmetric projection matrix.
                  p2: A symmetric projection matrix.
                Returns:
                  A 2 x 2 kernel [[p1p2,         p1(1-p2)],
                                  [(1-p1)p2, (1-p1)(1-p2)]].
                Raises:
                  ValueError: If the dimensions of p1 and p2 are different.
                """
                if p1.shape.as_list() != p2.shape.as_list():
                  raise ValueError("The dimension of the matrices must be the same.")
                n = p1.shape.as_list()[0]
                kernel2x2 = {}
                eye = tf.eye(n, dtype=dtype)
                kernel2x2[0, 0] = tf.matmul(p1, p2)
                kernel2x2[0, 1] = tf.matmul(p1, (eye - p2))
                kernel2x2[1, 0] = tf.matmul((eye - p1), p2)
                kernel2x2[1, 1] = tf.matmul((eye - p1), (eye - p2))

                return kernel2x2

            def _matrix_conv(m1, m2):
                """Matrix convolution.
                Args:
                    m1: A k x k dictionary, each element is a n x n matrix.
                    m2: A l x l dictionary, each element is a n x n matrix.
                Returns:
                    (k + l - 1) * (k + l - 1) dictionary each element is a n x n matrix.
                Raises:
                    ValueError: if the entries of m1 and m2 are of different dimensions.
                """

                n = (m1[0, 0]).shape.as_list()[0]
                if n != (m2[0, 0]).shape.as_list()[0]:
                    raise ValueError("The entries in matrices m1 and m2 "
                                     "must have the same dimensions!")
                k = int(np.sqrt(len(m1)))
                l = int(np.sqrt(len(m2)))
                result = {}
                size = k + l - 1
                # Compute matrix convolution between m1 and m2.
                for i in range(size):
                    for j in range(size):
                        result[i, j] = tf.zeros([n, n], dtype)
                        for index1 in range(min(k, i + 1)):
                            for index2 in range(min(k, j + 1)):
                                if (i - index1) < l and (j - index2) < l:
                                    result[i, j] += tf.matmul(m1[index1, index2],
                                                              m2[i - index1, j - index2])
                return result

            def _orthogonal_kernel(ksize, cin, cout):
                """Construct orthogonal kernel for convolution.
                Args:
                    ksize: Kernel size.
                    cin: Number of input channels.
                    cout: Number of output channels.
                Returns:
                    An [ksize, ksize, cin, cout] orthogonal kernel.
                Raises:
                    ValueError: If cin > cout.
                """
                if cin > cout:
                    raise ValueError("The number of input channels cannot exceed "
                                     "the number of output channels.")
                orth = _orthogonal_matrix(cout)[0:cin, :]
                if ksize == 1:
                    return tf.expand_dims(tf.expand_dims(orth, 0), 0)

                p = _block_orth(_symmetric_projection(cout), _symmetric_projection(cout))
                for _ in range(ksize - 2):
                    temp = _block_orth(_symmetric_projection(cout), _symmetric_projection(cout))
                    p = _matrix_conv(p, temp)
                for i in range(ksize):
                    for j in range(ksize):
                        p[i, j] = tf.matmul(orth, p[i, j])

                return _dict_to_tensor(p, ksize, ksize)

            def _orthogonal_matrix(n):
                """Construct an n x n orthogonal matrix.
                Args:
                    n: Dimension.
                Returns:
                    A n x n orthogonal matrix.
                """
                seed = None
                a = tf.random_normal([n, n], dtype=dtype, seed=seed)
                if seed:
                    seed += 1
                q, r = tf.qr(a)
                d = tf.diag_part(r)
                # make q uniform
                q *= tf.sign(d)
                return q

            def _symmetric_projection(n):
                """Compute a n x n symmetric projection matrix.
                Args:
                    n: Dimension.
                Returns:
                    A n x n symmetric projection matrix, i.e. a matrix P s.t. P=P*P, P=P^T.
                """
                seed = None
                q = _orthogonal_matrix(n)
                # randomly zeroing out some columns
                mask = tf.cast(tf.random_normal([n], seed=seed) > 0, dtype)
                if seed:
                    seed += 1
                c = tf.multiply(q, mask)
                return tf.matmul(c, tf.matrix_transpose(c))
            if len(shape) != 4:
                raise ValueError("The tensor to initialize must be four-dimensional")
            if shape[-2] > shape[-1]:
                raise ValueError("In_filters cannot be greater than out_filters.")
            if shape[0] != shape[1]:
                raise ValueError("Kernel sizes must be equal.")
            gain = 1.0
            kernel = _orthogonal_kernel(shape[0], shape[2], shape[3])
            kernel *= tf.cast(gain, dtype=dtype)
            sample = kernel
        else:
            raise NotImplementedError
        return sample

    w_reinit = {}
    for k, v in w.items():
        shape = v.shape.as_list()
        if None in shape: # rnn state tensors -> skip initialization
            w_reinit[k] = tf.identity(v)
        else:
            if len(shape) == 1: # b
                w_reinit[k] = random_sample(shape, init_b)
            elif len(shape) == 2: # w (fc)
                w_reinit[k] = random_sample(shape, init_w)
            elif len(shape) == 4: # w (conv)
                if init_w['kind'] == 'orthogonal':
                    init_w_update = init_w.copy()
                    init_w_update['kind'] = 'conv_orthogonal'
                    w_reinit[k] = random_sample(shape, init_w_update)
                else:
                    w_reinit[k] = random_sample(shape, init_w)
            else:
                raise NotImplementedError
    return w_reinit

def batch_jacobian(output, inp, use_pfor=True, parallel_iterations=None):
    """Computes and stacks jacobians of `output[i,...]` w.r.t. `input[i,...]`.

    e.g.
    x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    y = x * x
    jacobian = batch_jacobian(y, x)
    # => [[[2,  0], [0,  4]], [[6,  0], [0,  8]]]

    Args:
        output: A tensor with shape [b, y1, ..., y_n]. `output[i,...]` should
            only depend on `inp[i,...]`.
        inp: A tensor with shape [b, x1, ..., x_m]
        use_pfor: If true, uses pfor for computing the Jacobian. Else uses a tf.while_loop.
        parallel_iterations: A knob to control how many iterations and dispatched in
            parallel. This knob can be used to control the total memory usage.

    Returns:
        A tensor `t` with shape [b, y_1, ..., y_n, x1, ..., x_m] where `t[i, ...]`
        is the jacobian of `output[i, ...]` w.r.t. `inp[i, ...]`, i.e. stacked
        per-example jacobians.

    Raises:
        ValueError: if first dimension of `output` and `inp` do not match.

    (NL) This function is taken from the following (and minimally modified to be used):
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/parallel_for/gradients.py#L81
    """

    output_shape = output.shape
    if not output_shape[0].is_compatible_with(inp.shape[0]):
        raise ValueError("Need first dimension of output shape (%s) and inp shape "
                         "(%s) to match." % (output.shape, inp.shape))
    if output_shape.is_fully_defined():
        batch_size = int(output_shape[0])
        output_row_size = output_shape.num_elements() // batch_size
    else:
        output_shape = tf.shape(output)
        batch_size = output_shape[0]
        output_row_size = tf.size(output) // batch_size
    inp_shape = tf.shape(inp)
    # Flatten output to 2-D.
    with tf.control_dependencies([tf.assert_equal(batch_size, inp_shape[0])]):
        output = tf.reshape(output, [batch_size, output_row_size])

    def loop_fn(i):
        y = tf.gather(output, i, axis=1)
        return tf.gradients(y, inp)[0]

    #if use_pfor:
    if False:
        pfor_output = tf.pfor(loop_fn, output_row_size,
                                            parallel_iterations=parallel_iterations)
    else:
        pfor_output = for_loop(
            loop_fn, output.dtype,
            output_row_size,
            parallel_iterations=parallel_iterations)
    if pfor_output is None:
        return None
    pfor_output = tf.reshape(pfor_output, [output_row_size, batch_size, -1])
    output = tf.transpose(pfor_output, [1, 0, 2])
    new_shape = tf.concat([output_shape, inp_shape[1:]], axis=0)
    return tf.reshape(output, new_shape)

def for_loop(loop_fn, loop_fn_dtypes, iters, parallel_iterations=None):
    """Runs `loop_fn` `iters` times and stacks the outputs.

    Runs `loop_fn` `iters` times, with input values from 0 to `iters - 1`, and
    stacks corresponding outputs of the different runs.

    Args:
        loop_fn: A function that takes an int32 scalar tf.Tensor object representing
            the iteration number, and returns a possibly nested structure of tensor
            objects. The shape of these outputs should not depend on the input.
        loop_fn_dtypes: dtypes for the outputs of loop_fn.
        iters: Number of iterations for which to run loop_fn.
        parallel_iterations: The number of iterations that can be dispatched in
            parallel. This knob can be used to control the total memory usage.

    Returns:
        Returns a nested structure of stacked output tensor objects with the same
        nested structure as the output of `loop_fn`.

    (NL) This function is taken from the following (and minimally modified to be used):
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/parallel_for/control_flow_ops.py#L40
    """
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import tensor_array_ops
    from tensorflow.python.util import nest


    flat_loop_fn_dtypes = nest.flatten(loop_fn_dtypes)
    is_none_list = []

    def while_body(i, *ta_list):
        """Body of while loop."""
        fn_output = nest.flatten(loop_fn(i))
        if len(fn_output) != len(flat_loop_fn_dtypes):
            raise ValueError(
                "Number of expected outputs, %d, does not match the number of "
                "actual outputs, %d, from loop_fn" % (len(flat_loop_fn_dtypes),
                                                    len(fn_output)))
        outputs = []
        del is_none_list[:]
        is_none_list.extend([x is None for x in fn_output])
        for out, ta in zip(fn_output, ta_list):
            # TODO(agarwal): support returning Operation objects from loop_fn.
            if out is not None:
                ta = ta.write(i, array_ops.expand_dims(out, 0))
            outputs.append(ta)
        return tuple([i + 1] + outputs)

    if parallel_iterations is not None:
        extra_args = {"parallel_iterations": parallel_iterations}
    else:
        extra_args = {}
    ta_list = tf.while_loop(
        lambda i, *ta: i < iters,
        while_body,
        [0] + [tensor_array_ops.TensorArray(dtype, iters)
               for dtype in flat_loop_fn_dtypes],
        **extra_args)[1:]

    # TODO(rachelim): enable this for sparse tensors

    output = [None if is_none else ta.concat() for ta, is_none in zip(ta_list, is_none_list)]
    return nest.pack_sequence_as(loop_fn_dtypes, output)

def compute_loss(labels, logits, fake_uniform=False):
    ''' Computes cross-entropy loss.
    '''
    assert len(labels.shape)+1 == len(logits.shape)
    num_classes = logits.shape.as_list()[-1]
    labels = tf.one_hot(labels, num_classes, dtype=tf.float32)
    if fake_uniform:
        labels = tf.ones_like(logits) / num_classes
    return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

def get_optimizer(optimizer, learning_rate):
    if optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    else:
        raise NotImplementedError
    return optimizer

def prepare_optimization(optimizer, learning_rate, decay_type, boundaries=None, values=None):
    global_step = tf.Variable(0, trainable=False)
    if decay_type == 'constant':
        learning_rate = tf.constant(learning_rate)
    elif decay_type == 'piecewise':
        assert len(boundaries)+1 == len(values)
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    else:
        raise NotImplementedError
    # optimizer
    optim = get_optimizer(optimizer, learning_rate)
    return optim, learning_rate, global_step

def vectorize_dict(x, sortkeys=None):
    assert isinstance(x, dict)
    if sortkeys is None:
        sortkeys = x.keys()
    def restore(v, x_shape, sortkeys):
        # v splits for each key
        split_sizes = []
        for key in sortkeys:
            split_sizes.append(reduce(lambda x, y: x*y, x_shape[key]))
        v_splits = tf.split(v, num_or_size_splits=split_sizes)
        # x restore
        x_restore = {}
        for i, key in enumerate(sortkeys):
            x_restore.update({key: tf.reshape(v_splits[i], x_shape[key])})
        return x_restore
    # vectorized dictionary
    x_vec = tf.concat([tf.reshape(x[k], [-1]) for k in sortkeys], axis=0)
    # restore function
    x_shape = {k: x[k].shape.as_list() for k in sortkeys}
    restore_fn = functools.partial(restore, x_shape=x_shape, sortkeys=sortkeys)
    return x_vec, restore_fn

def normalize_dict(x):
    x_v, restore_fn = vectorize_dict(x)
    x_v_norm = tf.divide(x_v, tf.reduce_sum(x_v))
    x_norm = restore_fn(x_v_norm)
    return x_norm

def compute_fans(shape):
    """Computes the number of input and output units for a weight shape.
    Args:
        shape: Integer shape tuple or TF tensor shape.
    Returns:
        A tuple of scalars (fan_in, fan_out).
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1.
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out

def create_sparse_mask(mask, target_sparsity, soft=False):
    def threshold_vec(vec, target_sparsity, soft):
        num_params = vec.shape.as_list()[0]
        kappa = int(round(num_params * (1. - target_sparsity)))
        if soft:
            topk, ind = tf.nn.top_k(vec, k=kappa+1, sorted=True)
            mask_sparse_v = tf.to_float(tf.greater(vec, topk[-1]))
        else:
            topk, ind = tf.nn.top_k(vec, k=kappa, sorted=True)
            mask_sparse_v = tf.sparse_to_dense(ind, tf.shape(vec),
                tf.ones_like(ind, dtype=tf.float32), validate_indices=False)
        return mask_sparse_v
    if isinstance(mask, dict):
        mask_v, restore_fn = vectorize_dict(mask)
        mask_sparse_v = threshold_vec(mask_v, target_sparsity, soft)
        return restore_fn(mask_sparse_v)
    else:
        return threshold_vec(mask, target_sparsity, soft)

def apply_mask(weights, mask):
    all_keys = weights.keys()
    target_keys = mask.keys()
    remain_keys = list(set(all_keys) - set(target_keys))
    w_sparse = {k: mask[k] * weights[k] for k in target_keys}
    w_sparse.update({k: weights[k] for k in remain_keys})
    return w_sparse

def compute_sparsity(weights, target_keys):
    assert isinstance(weights, dict)
    w = {k: weights[k] for k in target_keys}
    w_v, _ = vectorize_dict(w)
    sparsity = tf.nn.zero_fraction(w_v)
    return sparsity

def compute_isometry(weights):
    isometry_penalty_layer = {}
    for k, v in weights.items():
        if 'w' in k:
            if len(v.shape) > 2:
                v = tf.reshape(v, [-1, v.shape[-1]])
            isometry_penalty_layer[k] = tf.norm(
                tf.matmul(tf.transpose(v), v) - tf.eye(v.shape.as_list()[1]), ord='euclidean')
    isometry_penalty = tf.reduce_sum(list(isometry_penalty_layer.values()))
    optim_o, leraning_rate_o, global_step_o = prepare_optimization('sgd', 0.1, 'constant')
    isometry_penalty_op = optim_o.minimize(isometry_penalty, global_step=global_step_o)
    return isometry_penalty_op

def compute_jsv(inputs, outputs):
    jacobian_raw = batch_jacobian(outputs, inputs)
    jacobian_shape = [-1, static_size(outputs[0]), static_size(inputs[0])]
    jacobian = tf.reshape(jacobian_raw, jacobian_shape) # [b, dim(out), dim(in)]
    s, u, v = tf.svd(jacobian)
    return s
