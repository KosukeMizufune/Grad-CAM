import copy

import chainer
from chainer import cuda, utils
from chainer import functions as F


class BackProp(object):
    def __init__(self, model):
        self.model = model

    def backprop(self, x, lab, layer):
        if len(x.shape) == 3:
            x = x.reshape(1, 3, self.model.size, self.model.size)
        layer = [layer, 'prob']
        with chainer.using_config('train', False):
            acts = self.model(self.model.xp.asarray(x), layers=layer)
        acts['prob'].grad = self.model.xp.zeros_like(acts['prob'].data)
        acts['prob'].grad[:, lab] = 1

        self.model.cleargrads()
        acts['prob'].backward(retain_grad=True)
        return acts


class GradCAM(BackProp):
    def __init__(self, model):
        super(GradCAM, self).__init__(model)

    def execute(self, x, lab, layer):
        acts = self.backprop(x, lab, layer)
        alpha = self.model.xp.mean(acts[layer].grad, axis=(2, 3))
        gcam = self.model.xp.tensordot(alpha[0], acts[layer].data[0], axes=(0, 0))
        gcam = self.model.xp.maximum(gcam, 0)
        gcam = chainer.cuda.to_cpu(gcam)
        return gcam


class GuidedBackProp(BackProp):
    def __init__(self, model):
        super(GuidedBackProp, self).__init__(copy.deepcopy(model))
        for _, funcs in self.model.functions.items():
            for i in range(len(funcs)):
                if funcs[i] is F.relu:
                    funcs[i] = GuidedReLU()
                elif isinstance(funcs[i], chainer.Chain):
                    self._replace_relu(funcs[i])

    def _replace_relu(self, chain):
        if hasattr(chain, 'functions'):
            for _, funcs in chain.functions.items():
                for i in range(len(funcs)):
                    if funcs[i] is F.relu:
                        funcs[i] = GuidedReLU()
        for child in chain.children():
            if hasattr(child, 'functions'):
                for _, funcs in child.functions.items():
                    for i in range(len(funcs)):
                        if funcs[i] is F.relu:
                            funcs[i] = GuidedReLU()
            elif isinstance(child, chainer.Chain):
                self._replace_relu(child)

    def execute(self, x, label, layer):
        acts = self.backprop(x, label, layer)
        gbp = chainer.cuda.to_cpu(acts['input'].grad[0])
        gbp = gbp.transpose(1, 2, 0)
        return gbp


class GuidedReLU(chainer.function.Function):
    def forward(self, x):
        xp = chainer.cuda.get_array_module(x[0])
        self.retain_inputs(())
        self.retain_outputs((0,))
        y = xp.maximum(x[0], 0)
        return y,

    def backward_cpu(self, x, gy):
        y = self.output_data[0]
        return utils.force_array(gy[0] * (y > 0) * (gy[0] > 0)),

    def backward_gpu(self, x, gy):
        y = self.output_data[0]
        gx = cuda.elementwise(
            'T y, T gy', 'T gx',
            'gx = (y > 0 && gy > 0) ? gy : (T)0',
            'relu_bwd')(y, gy[0])
        return gx,
