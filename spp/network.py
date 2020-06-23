import tensorflow as tf

from functools import reduce
from helpers import static_size

dtype = tf.float32


def load_network(**kwargs):
    arch = kwargs.pop('arch')
    networks = {
        'mlp-3-linear': lambda: MLP(depth=3, width=100, activation='linear', **kwargs),
        'mlp-5-linear': lambda: MLP(depth=5, width=100, activation='linear', **kwargs),
        'mlp-7-linear': lambda: MLP(depth=7, width=100, activation='linear', **kwargs),
        'mlp-3-tanh': lambda: MLP(depth=3, width=100, activation='tanh', **kwargs),
        'mlp-5-tanh': lambda: MLP(depth=5, width=100, activation='tanh', **kwargs),
        'mlp-7-tanh': lambda: MLP(depth=7, width=100, activation='tanh', **kwargs),
        'lenet': lambda: LeNet(**kwargs),
        'vgg': lambda: VGG(**kwargs),
        'resnet32': lambda: ResNet(num_filters=16, k=1, block_sizes=[5, 5, 5],
            resnet_version=1, shortcut_type='zeropad', resnet_type='resnet', **kwargs),
        'resnet56': lambda: ResNet(num_filters=16, k=1, block_sizes=[9, 9, 9],
            resnet_version=1, shortcut_type='zeropad', resnet_type='resnet', **kwargs),
        'resnet110': lambda: ResNet(num_filters=16, k=1, block_sizes=[18, 18, 18],
            resnet_version=1, shortcut_type='zeropad', resnet_type='resnet', **kwargs),
        'wrn-16': lambda: ResNet(num_filters=16, k=6, block_sizes=[2, 2, 2],
            resnet_version=2, shortcut_type='projection', resnet_type='wide-resnet', **kwargs),
    }
    return networks[arch]()


def get_initializer(initializer):
    return tf.zeros_initializer(dtype=dtype) # to disable TF initialization.


def get_activation(activation):
    if activation == 'linear':
        return tf.identity
    elif activation == 'tanh':
        return tf.nn.tanh
    elif activation == 'relu':
        return tf.nn.relu
    elif activation == 'selu':
        return tf.nn.selu
    elif activation == 'lrelu':
        return tf.nn.leaky_relu
    else:
        raise NotImplementedError


class Network(object):
    def __init__(self, datasource):
        self.dtype = tf.float32
        self.datasource = datasource
        if self.datasource == 'mnist':
            self.input_dims = [28, 28, 1]
            self.output_dims = 10
        elif self.datasource == 'fashion-mnist':
            self.input_dims = [28, 28, 1]
            self.output_dims = 10
        elif self.datasource == 'cifar-10':
            self.input_dims = [32, 32, 3]
            self.output_dims = 10
        elif self.datasource == 'tiny-imagenet':
            self.input_dims = [64, 64, 3]
            self.output_dims = 200
        else:
            raise NotImplementedError

    def construct_inputs(self):
        return {
            'image': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, init_w, init_b, trainable, scope):
        params = {
            'dtype': self.dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.variable_scope(scope):
            for k, v in self.weights_shape.items():
                if len(v) == 1: # b
                    params.update({'initializer': get_initializer(init_b)})
                elif len(v) == 2: # w (fc)
                    params.update({'initializer': get_initializer(init_w)})
                elif len(v) == 4: # w (conv)
                    if init_w['kind'] == 'orthogonal':
                        init_w_update = init_w.copy()
                        init_w_update['kind'] = 'conv_orthogonal'
                        params.update({'initializer': get_initializer(init_w_update)})
                    else:
                        params.update({'initializer': get_initializer(init_w)})
                else:
                    raise NotImplementedError
                weights[k] = tf.get_variable(k, v, **params)
        return weights

    def construct_network(self, init_w, init_b):
        self.inputs = self.construct_inputs()
        self.weights_shape, self.states_shape = self.set_weights_shape()
        self.weights = self.construct_weights(init_w, init_b, True, 'net')
        self.num_params = sum([static_size(v) for v in self.weights.values()])


class MLP(Network):
    def __init__(self,
                 datasource,
                 init_w,
                 init_b,
                 depth,
                 width,
                 activation,
                 ):
        super(MLP, self).__init__(datasource)
        self.name = 'mlp-{}-{}-{}'.format(depth, width, activation)
        self.depth = depth
        self.width = width
        self.activation = activation
        self.construct_network(init_w, init_b)

    def set_weights_shape(self):
        weights_shape = {}
        weights_shape.update({
            'w1': [784, self.width],
            'b1': [self.width],
        })
        for i in range(2, self.depth):
            weights_shape.update({
                'w{}'.format(i): [self.width, self.width],
                'b{}'.format(i): [self.width],
            })
        weights_shape.update({
            'w{}'.format(self.depth): [self.width, self.output_dims],
            'b{}'.format(self.depth): [self.output_dims],
        })
        return weights_shape, None

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
        for i in range(self.depth):
            inputs = tf.matmul(inputs, weights['w{}'.format(i+1)]) + weights['b{}'.format(i+1)]
            if i < self.depth-1:
                inputs = get_activation(self.activation)(inputs)
        return inputs


class LeNet(Network):
    def __init__(self,
                 datasource,
                 init_w,
                 init_b,
                 ):
        super(LeNet, self).__init__(datasource)
        self.name = 'lenet'
        self.construct_network(init_w, init_b)

    def set_weights_shape(self):
        weights_shape = {
            'w1': [784, 300],
            'w2': [300, 100],
            'w3': [100, self.output_dims],
            'b1': [300],
            'b2': [100],
            'b3': [self.output_dims],
        }
        return weights_shape, None

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
        inputs = tf.matmul(inputs, weights['w1']) + weights['b1']
        inputs = tf.nn.relu(inputs)
        inputs = tf.matmul(inputs, weights['w2']) + weights['b2']
        inputs = tf.nn.relu(inputs)
        inputs = tf.matmul(inputs, weights['w3']) + weights['b3']
        return inputs


class VGG(Network):
    def __init__(self,
                 datasource,
                 init_w,
                 init_b,
                 ):
        super(VGG, self).__init__(datasource)
        self.name = 'vgg'
        self.construct_network(init_w, init_b)

    def set_weights_shape(self):
        weights_shape = {}
        weights_shape.update({
            'w1': [3, 3, 3, 64],
            'w2': [3, 3, 64, 64],
            'w3': [3, 3, 64, 128],
            'w4': [3, 3, 128, 128],
            'w5': [3, 3, 128, 256],
            'w6': [3, 3, 256, 256],
            'w7': [1, 1, 256, 256],
            'w8': [3, 3, 256, 512],
            'w9': [3, 3, 512, 512],
            'w10': [1, 1, 512, 512],
            'w11': [3, 3, 512, 512],
            'w12': [3, 3, 512, 512],
            'w13': [1, 1, 512, 512],
            'w14': [512, 512],
            'w15': [512, 512],
            'w16': [512, self.output_dims],
            'b1': [64],
            'b2': [64],
            'b3': [128],
            'b4': [128],
            'b5': [256],
            'b6': [256],
            'b7': [256],
            'b8': [512],
            'b9': [512],
            'b10': [512],
            'b11': [512],
            'b12': [512],
            'b13': [512],
            'b14': [512],
            'b15': [512],
            'b16': [self.output_dims],
        })
        return weights_shape, None

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        def _conv_block(inputs, bn_params, filt, st=1):
            inputs = tf.nn.conv2d(inputs, filt['w'], [1, st, st, 1], 'SAME') + filt['b']
            inputs = tf.layers.batch_normalization(inputs, **bn_params)
            inputs = tf.nn.relu(inputs)
            return inputs

        bn_params = {
            'training': is_train,
            'trainable': trainable,
        }
        init_st = 2 if self.datasource == 'tiny-imagenet' else 1
        inputs = _conv_block(inputs, bn_params, {'w': weights['w1'], 'b': weights['b1']}, init_st)
        inputs = _conv_block(inputs, bn_params, {'w': weights['w2'], 'b': weights['b2']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w3'], 'b': weights['b3']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w4'], 'b': weights['b4']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w5'], 'b': weights['b5']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w6'], 'b': weights['b6']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w7'], 'b': weights['b7']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w8'], 'b': weights['b8']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w9'], 'b': weights['b9']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w10'], 'b': weights['b10']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w11'], 'b': weights['b11']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w12'], 'b': weights['b12']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w13'], 'b': weights['b13']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
        inputs = tf.matmul(inputs, weights['w14']) + weights['b14']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.matmul(inputs, weights['w15']) + weights['b15']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.matmul(inputs, weights['w16']) + weights['b16']
        return inputs


class ResNet(Network):
    def __init__(self,
                 datasource,
                 init_w,
                 init_b,
                 num_filters,
                 k,
                 block_sizes,
                 resnet_version,
                 shortcut_type,
                 resnet_type,
                 ):
        super(ResNet, self).__init__(datasource)
        self.num_filters = num_filters
        self.k = k
        self.block_sizes = block_sizes
        self.resnet_version = resnet_version
        self.shortcut_type = shortcut_type
        self.resnet_type = resnet_type
        assert self.resnet_version in [1, 2]
        assert len(self.block_sizes) == 3 and len(set(self.block_sizes)) == 1
        self.num_block_layers = len(self.block_sizes)
        self.n = block_sizes[0]
        if self.resnet_type == 'resnet': # resnet
            num_weighted_layers_without_projection = 6 * self.n + 2
            self.name = '{}{}-v{}-{}'.format(self.resnet_type,
                num_weighted_layers_without_projection, self.resnet_version, self.shortcut_type)
        else: # wide-resnet
            assert self.resnet_version == 2
            assert self.shortcut_type == 'projection'
            num_conv_layers = ((2 * self.n) + 1) * self.num_block_layers + 1
            self.name = '{}-{}-{}'.format(self.resnet_type, num_conv_layers, self.k)
        self.construct_network(init_w, init_b)

    def set_weights_shape(self):
        weights_shape = {}
        cnt = 1
        weights_shape.update({
            'w1': [3, 3, 3, self.num_filters],
            'b1': [self.num_filters],
        })
        nChIn = self.num_filters
        for l in range(self.num_block_layers):
            nChOut = self.num_filters * (2**l) * self.k
            if self.shortcut_type == 'projection' and nChIn != nChOut:
                weights_shape.update({
                    'w{}-shortcut'.format(cnt): [1, 1, nChIn, nChOut],
                    'b{}-shortcut'.format(cnt): [nChOut],
                })
            for i in range(2 * self.block_sizes[l]):
                cnt += 1
                weights_shape.update({
                    'w{}'.format(cnt): [3, 3, nChIn if i == 0 else nChOut, nChOut],
                    'b{}'.format(cnt): [nChOut],
                })
            nChIn = nChOut
        cnt += 1
        weights_shape.update({
            'w{}'.format(cnt): [nChOut, self.output_dims],
            'b{}'.format(cnt): [self.output_dims],
        })
        return weights_shape, None

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        def _res_block_v1(inputs, bn_params, filt1, filt2, filt3=None, zeropad=False, k=1, st=1):
            shortcut = inputs
            if filt3 is not None:
                shortcut = tf.nn.conv2d(inputs, filt3['w'], [1, st, st, 1], 'SAME') + filt3['b']
                shortcut = tf.layers.batch_normalization(shortcut, **bn_params)
            elif zeropad:
                shortcut = tf.nn.avg_pool(inputs, [1, st, st, 1], [1, st, st, 1], 'SAME')
                shortcut = tf.concat([shortcut] + [tf.zeros_like(shortcut)]*(k-1), -1)
            else:
                pass
            inputs = tf.nn.conv2d(inputs, filt1['w'], [1, st, st, 1], 'SAME') + filt1['b']
            inputs = tf.layers.batch_normalization(inputs, **bn_params)
            inputs = tf.nn.relu(inputs)
            inputs = tf.nn.conv2d(inputs, filt2['w'], [1, 1, 1, 1], 'SAME') + filt2['b']
            inputs = tf.layers.batch_normalization(inputs, **bn_params)
            inputs += shortcut
            inputs = tf.nn.relu(inputs)
            return inputs

        def _res_block_v2(inputs, bn_params, filt1, filt2, filt3=None, zeropad=False, k=1, st=1):
            shortcut = inputs
            inputs = tf.layers.batch_normalization(inputs, **bn_params)
            inputs = tf.nn.relu(inputs)
            if filt3 is not None:
                shortcut = tf.nn.conv2d(inputs, filt3['w'], [1, st, st, 1], 'SAME') + filt3['b']
            elif zeropad:
                shortcut = tf.nn.avg_pool(inputs, [1, st, st, 1], [1, st, st, 1], 'SAME')
                shortcut = tf.concat([shortcut] + [tf.zeros_like(shortcut)]*(k-1), -1)
            else:
                pass
            inputs = tf.nn.conv2d(inputs, filt1['w'], [1, st, st, 1], 'SAME') + filt1['b']
            inputs = tf.layers.batch_normalization(inputs, **bn_params)
            inputs = tf.nn.relu(inputs)
            inputs = tf.nn.conv2d(inputs, filt2['w'], [1, 1, 1, 1], 'SAME') + filt2['b']
            return inputs + shortcut

        if self.resnet_version == 1:
            _res_block = _res_block_v1
        elif self.resnet_version == 2:
            _res_block = _res_block_v2
        else:
            raise NotImplementedError
        bn_params = {
            'training': is_train,
            'trainable': trainable,
        }
        init_st = 2 if self.datasource == 'tiny-imagenet' else 1
        cnt = 1
        inputs = tf.nn.conv2d(inputs, weights['w1'], [1,init_st,init_st,1], 'SAME') + weights['b1']
        if self.resnet_version == 1:
            inputs = tf.layers.batch_normalization(inputs, **bn_params)
            inputs = tf.nn.relu(inputs)
        for l in range(self.num_block_layers):
            dims = weights['w{}'.format(cnt+1)].shape.as_list()
            if dims[2] != dims[3]:
                if self.shortcut_type == 'zeropad':
                    inputs = _res_block(inputs, bn_params,
                        {'w': weights['w{}'.format(cnt+1)], 'b': weights['b{}'.format(cnt+1)]},
                        {'w': weights['w{}'.format(cnt+2)], 'b': weights['b{}'.format(cnt+2)]},
                        zeropad=True,
                        k=self.k if l == 0 else 2,
                        st=1 if l == 0 else 2,
                    )
                elif self.shortcut_type == 'projection':
                    inputs = _res_block(inputs, bn_params,
                        {'w': weights['w{}'.format(cnt+1)], 'b': weights['b{}'.format(cnt+1)]},
                        {'w': weights['w{}'.format(cnt+2)], 'b': weights['b{}'.format(cnt+2)]},
                        {'w': weights['w{}-shortcut'.format(cnt)],
                         'b': weights['b{}-shortcut'.format(cnt)]},
                        st=1 if l == 0 else 2,
                    )
                else:
                    raise NotImplementedError
            else:
                inputs = _res_block(inputs, bn_params,
                    {'w': weights['w{}'.format(cnt+1)], 'b': weights['b{}'.format(cnt+1)]},
                    {'w': weights['w{}'.format(cnt+2)], 'b': weights['b{}'.format(cnt+2)]},
                )
            cnt += 2
            for _ in range(1, self.block_sizes[l]):
                inputs = _res_block(inputs, bn_params,
                    {'w': weights['w{}'.format(cnt+1)], 'b': weights['b{}'.format(cnt+1)]},
                    {'w': weights['w{}'.format(cnt+2)], 'b': weights['b{}'.format(cnt+2)]},
                )
                cnt += 2
        if self.resnet_version == 2:
            inputs = tf.layers.batch_normalization(inputs, **bn_params)
            inputs = tf.nn.relu(inputs)
        cnt += 1
        assert inputs.shape.as_list()[1] == 8
        inputs = tf.reduce_mean(inputs, [1, 2])
        inputs = tf.matmul(inputs, weights['w{}'.format(cnt)]) + weights['b{}'.format(cnt)]
        return inputs
