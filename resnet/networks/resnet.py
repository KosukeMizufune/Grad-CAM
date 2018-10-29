import collections
import os
import sys

import chainer
from chainer import links as L
from chainer import functions as F
from chainer.initializers import constant, normal
from chainer.dataset import download
from chainer.serializers import npz


class ResNet(chainer.Chain):
    def __init__(self, pretrained_model, n_layers):
        super(ResNet, self).__init__()
        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            kwargs = {'initialW': constant.Zero()}
        else:
            # employ default initializers used in the original paper
            kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        if n_layers == 50:
            block = [3, 4, 6, 3]
        elif n_layers == 101:
            block = [3, 4, 23, 3]
        elif n_layers == 152:
            block = [3, 8, 36, 3]
        else:
            raise ValueError('The n_layers argument should be either 50, 101,'
                             ' or 152, but {} was given.'.format(n_layers))

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, 7, 2, 3, **kwargs)
            self.bn1 = L.BatchNormalization(64)
            self.res2 = BuildingBlock(block[0], 64, 64, 256, 1, **kwargs)
            self.res3 = BuildingBlock(block[1], 256, 128, 512, 2, **kwargs)
            self.res4 = BuildingBlock(block[2], 512, 256, 1024, 2, **kwargs)
            self.res5 = BuildingBlock(block[3], 1024, 512, 2048, 2, **kwargs)
            self.fc6 = L.Linear(2048, 1000)

        if pretrained_model and pretrained_model.endswith('.caffemodel'):
            _retrieve(n_layers, 'ResNet-{}-model.npz'.format(n_layers),
                      pretrained_model, self)
        elif pretrained_model:
            npz.load_npz(pretrained_model, self)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('pool1', [lambda x: F.max_pooling_2d(x, ksize=3, stride=2)]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
            ('res5', [self.res5]),
            ('pool5', [_global_average_pooling_2d]),
            ('fc6', [self.fc6]),
            ('prob', [F.softmax]),
        ])

    @property
    def available_layers(self):
        return list(self.functions.keys())

    @classmethod
    def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz, n_layers=50):
        """Converts a pre-trained caffemodel to a chainer model.
        Args:
            path_caffemodel (str): Path of the pre-trained caffemodel.
            path_npz (str): Path of the converted chainer model.
        """

        # As CaffeFunction uses shortcut symbols,
        # we import CaffeFunction here.
        from chainer.links.caffe.caffe_function import CaffeFunction
        caffemodel = CaffeFunction(path_caffemodel)
        chainermodel = cls(pretrained_model=None, n_layers=n_layers)
        if n_layers == 50:
            _transfer_resnet50(caffemodel, chainermodel)
        elif n_layers == 101:
            _transfer_resnet101(caffemodel, chainermodel)
        elif n_layers == 152:
            _transfer_resnet152(caffemodel, chainermodel)
        else:
            raise ValueError('The n_layers argument should be either 50, 101,'
                             ' or 152, but {} was given.'.format(n_layers))
        npz.save_npz(path_npz, chainermodel, compression=False)

    def __call__(self, x, layers=None, **kwargs):
        """__call__(self, x, layers=['prob'])
        Computes all the feature maps specified by ``layers``.
        .. warning::
           ``test`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', train)``.
           See :func:`chainer.using_config`.
        Args:
            x (~chainer.Variable): Input variable. It should be prepared by
                ``prepare`` function.
            layers (list of str): The list of layer names you want to extract.
        Returns:
            Dictionary of ~chainer.Variable: A directory in which
            the key contains the layer name and the value contains
            the corresponding feature map variable.
        """

        if layers is None:
            layers = ['prob']

        h = x
        activations = {'input': h}
        target_layers = set(layers)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        return activations


class BuildingBlock(chainer.Chain):
    """A building block that consists of several Bottleneck layers.
    Args:
        n_layer (int): Number of layers used in the building block.
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, initialW=None):
        super(BuildingBlock, self).__init__()
        with self.init_scope():
            self.a = BottleneckA(
                in_channels, mid_channels, out_channels, stride, initialW)
            self._forward = ["a"]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = BottleneckB(out_channels, mid_channels, initialW)
                setattr(self, name, bottleneck)
                self._forward.append(name)

    def __call__(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
        return x

    @property
    def forward(self):
        return [getattr(self, name) for name in self._forward]


class BottleneckA(chainer.Chain):
    """A bottleneck layer that reduces the resolution of the feature map.
    Args:
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=2, initialW=None):
        super(BottleneckA, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, out_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(out_channels)
            self.conv4 = L.Convolution2D(
                in_channels, out_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn4 = L.BatchNormalization(out_channels)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('conv2', [self.conv2, self.bn2, F.relu]),
            ('conv3', [self.conv3, self.bn3]),
            ('proj4', [self.conv4, self.bn4]),
            ('relu5', [F.add, F.relu]),
        ])

    def __call__(self, x):
        h = x
        for key, funcs in self.functions.items():
            if key == 'proj4':
                for func in funcs:
                    x = func(x)
            else:
                for func in funcs:
                    if func is F.add:
                        h = func(h, x)
                    else:
                        h = func(h)
        return h


class BottleneckB(chainer.Chain):
    """A bottleneck layer that maintains the resolution of the feature map.
    Args:
        in_channels (int): Number of channels of input and output arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, initialW=None):
        super(BottleneckB, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, in_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(in_channels)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('conv2', [self.conv2, self.bn2, F.relu]),
            ('conv3', [self.conv3, self.bn3]),
            ('relu4', [F.add, F.relu]),
        ])

    def __call__(self, x):
        h = x
        for key, funcs in self.functions.items():
            for func in funcs:
                if func is F.add:
                    h = func(h, x)
                else:
                    h = func(h)
        return h


class ResNet101(ResNet):
    def __init__(self, pretrained_model='auto'):
        if pretrained_model == 'auto':
            pretrained_model = 'ResNet-101-model.caffemodel'
        super(ResNet101, self).__init__(pretrained_model, 101)


class ResNetTrain101(chainer.Chain):
    def __init__(self, n_out, init_param=None):
        super(ResNetTrain101, self).__init__()

        with self.init_scope():
            self.base = ResNet101()
            self.fc = L.Linear(None, n_out, initialW=init_param)

    def __call__(self, x):
        h = self.base(x, layers=['pool5'])['pool5']
        y = self.fc(h)
        return y


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = F.reshape(h, (n, channel))
    return h


def _transfer_components(src, dst_conv, dst_bn, bname, cname):
    src_conv = getattr(src, 'res{}_branch{}'.format(bname, cname))
    src_bn = getattr(src, 'bn{}_branch{}'.format(bname, cname))
    src_scale = getattr(src, 'scale{}_branch{}'.format(bname, cname))
    dst_conv.W.data[:] = src_conv.W.data
    dst_bn.avg_mean[:] = src_bn.avg_mean
    dst_bn.avg_var[:] = src_bn.avg_var
    dst_bn.gamma.data[:] = src_scale.W.data
    dst_bn.beta.data[:] = src_scale.bias.b.data


def _transfer_bottleneckA(src, dst, name):
    _transfer_components(src, dst.conv1, dst.bn1, name, '2a')
    _transfer_components(src, dst.conv2, dst.bn2, name, '2b')
    _transfer_components(src, dst.conv3, dst.bn3, name, '2c')
    _transfer_components(src, dst.conv4, dst.bn4, name, '1')


def _transfer_bottleneckB(src, dst, name):
    _transfer_components(src, dst.conv1, dst.bn1, name, '2a')
    _transfer_components(src, dst.conv2, dst.bn2, name, '2b')
    _transfer_components(src, dst.conv3, dst.bn3, name, '2c')


def _transfer_block(src, dst, names):
    _transfer_bottleneckA(src, dst.a, names[0])
    for i, name in enumerate(names[1:]):
        dst_bottleneckB = getattr(dst, 'b{}'.format(i + 1))
        _transfer_bottleneckB(src, dst_bottleneckB, name)


def _transfer_resnet50(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.conv1.b.data[:] = src.conv1.b.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3, ['3a', '3b', '3c', '3d'])
    _transfer_block(src, dst.res4, ['4a', '4b', '4c', '4d', '4e', '4f'])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

    dst.fc6.W.data[:] = src.fc1000.W.data
    dst.fc6.b.data[:] = src.fc1000.b.data


def _transfer_resnet101(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3, ['3a', '3b1', '3b2', '3b3'])
    _transfer_block(src, dst.res4,
                    ['4a'] + ['4b{}'.format(i) for i in range(1, 23)])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

    dst.fc6.W.data[:] = src.fc1000.W.data
    dst.fc6.b.data[:] = src.fc1000.b.data


def _transfer_resnet152(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3,
                    ['3a'] + ['3b{}'.format(i) for i in range(1, 8)])
    _transfer_block(src, dst.res4,
                    ['4a'] + ['4b{}'.format(i) for i in range(1, 36)])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

    dst.fc6.W.data[:] = src.fc1000.W.data
    dst.fc6.b.data[:] = src.fc1000.b.data


def _make_npz(path_npz, path_caffemodel, model, n_layers):
    sys.stderr.write(
        'Now loading caffemodel (usually it may take few minutes)\n')
    sys.stderr.flush()
    if not os.path.exists(path_caffemodel):
        raise IOError(
            'The pre-trained caffemodel does not exist. Please download it '
            'from \'https://github.com/KaimingHe/deep-residual-networks\', '
            'and place it on {}'.format(path_caffemodel))
    ResNet.convert_caffemodel_to_npz(path_caffemodel, path_npz, n_layers)
    npz.load_npz(path_npz, model)
    return model


def _retrieve(n_layers, name_npz, name_caffemodel, model):
    root = download.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name_npz)
    path_caffemodel = os.path.join(root, name_caffemodel)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, path_caffemodel, model, n_layers),
        lambda path: npz.load_npz(path, model))


class ResNetTest101(chainer.Chain):
    def __init__(self, n_out, size):
        super(ResNetTest101, self).__init__()
        self.n_out = n_out

        with self.init_scope():
            self.base = ResNet101()
            self.fc = L.Linear(None, n_out)

        self.size = size

    @property
    def functions(self):
        return collections.OrderedDict([
            ('base', [self.base]),
            ('fc', [self.fc]),
        ])

    def __call__(self, x, layers=['pool5']):
        h = chainer.Variable(x)
        if isinstance(layers, str):
            layers = [layers]
        if 'pool5' not in layers:
            layers.extend(['pool5'])
        acts = self.base(h, layers=layers)
        acts['fc'] = self.fc(acts['pool5'])
        acts['prob'] = F.softmax(acts['fc'])
        acts['input'] = h
        return acts
