import numpy

from chainer import cuda
from chainer import optimizer


class COCOBBackprop(optimizer.GradientMethod):

    """COCOB-Backprop optimization algorithm.

    See: https://arxiv.org/abs/1705.07795

    """

    def __init__(self, alpha=100, eps=1e-8):
        self.alpha = alpha
        self.eps = eps

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['w1'] = param.data.copy()
            state['L'] = xp.zeros_like(param.data)
            state['G'] = xp.zeros_like(param.data)
            state['Reward'] = xp.zeros_like(param.data)
            state['theta'] = xp.zeros_like(param.data)

    def update_one_cpu(self, param, state):
        w1 = state['w1']
        L = state['L']
        G = state['G']
        Reward = state['Reward']
        theta = state['theta']
        grad = param.grad

        abs_grad = numpy.absolute(grad)
        L[:] = numpy.maximum(L, abs_grad)
        G += abs_grad
        Reward[:] = numpy.maximum(Reward - param.data * grad, 0)
        theta += grad
        beta = theta / (L * numpy.maximum(G + L, self.alpha * L) + self.eps)
        param.data[:] = w1 - beta * (L + Reward)

    def update_one_gpu(self, param, state):
        cuda.elementwise(
            'T grad, T w1, T alpha, T eps',
            'T param, T L, T G, T Reward, T theta',
            '''T abs_grad = abs(grad);
               L = max(L, abs_grad);
               G += abs_grad;
               Reward = max(Reward - param * grad, 0.0);
               theta += grad;
               T beta = theta / (L * max(G + L, alpha * L) + eps);
               param = w1 - beta * (L + Reward);''',
            'cocob_backprop')(
                param.grad, state['w1'], self.alpha, self.eps,
                param.data, state['L'], state['G'], state['Reward'],
                state['theta'])
