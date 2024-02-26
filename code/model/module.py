import torch
import numpy as np


def activation_2nd_ord(func, x, dx_dh=None, d2x_dh2=None, ord=1):
    '''
    func must be element-wise activation
    x: ([n_batch], ndim)
    dx_dh: Jacobian, tensor of shape ([n_batch], ndim, nhdim)
    d2x_dh2: Diagonal of Hessian, tensor of shape  ([n_batch], ndim, nhdim)
    '''
    y, dy_dh, d2y_dh2 = func(x), None, None
    if dx_dh is not None and ord >= 1:
        dy_dx = func.grad(x, ord=1).unsqueeze(dim=-1)  # ([n_batch], ndim, 1)
        dy_dh = dy_dx * dx_dh

        if d2x_dh2 is not None and ord >1:
            d2y_dx2 = func.grad(x, ord=2).unsqueeze(dim=-1)  # ([n_batch], ndim, 1)
            d2y_dh2 = d2y_dx2 * (dx_dh ** 2) + d2x_dh2 * dy_dx  # ([n_batch], ndim, nhdim)

    return y, dy_dh, d2y_dh2

def linear_2nd_ord(linear, x, dx_dh=None, d2x_dh2=None, ord=1):
    '''
    linear must be linear layer
    x: ([n_batch], ndim)
    dx_dh: Jacobian, tensor of shape ([n_batch], ndim, nhdim)
    d2x_dh2: Diagonal of Hessian, tensor of shape  ([n_batch], ndim, nhdim)
    '''
    y, dy_dh, d2y_dh2 = linear(x), None, None
    if dx_dh is not None and ord >= 1:
        dy_dh = linear.weight @ dx_dh

        if d2x_dh2 is not None and ord > 1:
            d2y_dh2 = linear.weight @ d2x_dh2

    return y, dy_dh, d2y_dh2


class ActivationFunc(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return x

    def grad(self, x, ord):
        if ord == 1 or ord == 2:
            return torch.ones_like(x) * (2 - ord)
        raise NotImplementedError


class Sigmoid(ActivationFunc):
    def __call__(self, x):
        return torch.sigmoid(x)

    def grad(self, x, ord):
        y = torch.sigmoid(x)
        dy = y * (1 - y)
        if ord == 1:
            return dy
        if ord == 2:
            return (1 - 2 * y) * dy
        raise NotImplementedError


class Tanh(ActivationFunc):
    def __call__(self, x):
        return torch.tanh(x)

    def grad(self, x, ord):
        y = torch.tanh(x)
        dy = 1 - y ** 2
        if ord == 1:
            return dy
        if ord == 2:
            return -2 * y * dy
        raise NotImplementedError


class Sinusoidal(ActivationFunc):
    def __call__(self, x):
        return torch.sin(x)

    def grad(self, x, ord):
        if ord == 1:
            return torch.cos(x)
        if ord == 2:
            return -torch.sin(x)
        raise NotImplementedError


class Cosusoidal(ActivationFunc):
    def __call__(self, x):
        return torch.cos(x)

    def grad(self, x, ord):
        if ord == 1:
            return -torch.sin(x)
        if ord == 2:
            return -torch.cos(x)
        raise NotImplementedError

class PosEmb(torch.nn.Module):
    def __init__(self, input_dim, num_freqs, max_freq):
        super().__init__()
        freq_bands = 2. ** torch.linspace(0., max_freq, num_freqs)
        w = torch.zeros([input_dim,num_freqs*input_dim])
        for i in range(input_dim):
            w[i,num_freqs*i:num_freqs*(i+1)] = freq_bands
        self.register_buffer('weights', w)
        self.sin = Sinusoidal()
        self.cos = Cosusoidal()
        self.identity = ActivationFunc()
        self.out_dim = 2*self.weights.shape[1] + input_dim
        self.in_dim = input_dim

    def forward(self, x, dx_dh=None, d2x_dh2=None, ord=1):
        y0, dy0_dh, d2y0_dh2 = activation_2nd_ord(self.identity, x, dx_dh, d2x_dh2,ord=ord)
        y,dy_dh,d2y_dh2 = torch.matmul(x,self.weights),None,None
        if dx_dh is not None and ord >=1:
            dy_dh = self.weights.transpose(0,1) @ dx_dh
            if d2x_dh2 is not None and ord > 1:
                d2y_dh2 = self.weights.transpose(0,1) @ d2x_dh2
        y1, dy_dh_1, d2y_dh2_1 = activation_2nd_ord(self.sin,y,dy_dh,d2y_dh2,ord=ord)
        y2, dy_dh_2, d2y_dh2_2 = activation_2nd_ord(self.cos,y,dy_dh,d2y_dh2,ord=ord)

        x = torch.cat([y0, y1, y2],dim=-1)
        if dy_dh is not None and ord >=1:
            dx_dh = torch.cat([dy0_dh, dy_dh_1,dy_dh_2],dim=1)
            if d2y_dh2 is not None and ord > 1:
                d2x_dh2 = torch.cat([d2y0_dh2, d2y_dh2_1,d2y_dh2_2],dim=1)

        return x, dx_dh, d2x_dh2

# class Sinusoidal(ActivationFunc):
#     def __call__(self, x):
#         return torch.sin(30*x)
#
#     def grad(self, x, ord):
#         if ord == 1:
#             return 30*torch.cos(30*x)
#         if ord == 2:
#             return -30*30*torch.sin(30*x)
#         raise NotImplementedError

class LeakyReLU(ActivationFunc):
    def __init__(self, negative_slope=0.01):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def __call__(self, x):
        return torch.nn.functional.leaky_relu(x, self.negative_slope)

    def grad(self, x, ord):
        if ord == 1:
            return (x >= 0).to(x.dtype) * (1 - self.negative_slope) + self.negative_slope
        if ord == 2:
            return torch.zeros_like(x)
        raise NotImplementedError

class SoftPlus(ActivationFunc):
    def __init__(self, beta=1,threshold=20):
        super(SoftPlus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def __call__(self, x):
        return torch.nn.functional.softplus(x, beta=self.beta,threshold=self.threshold)

    # def grad(self, x, ord):
    #     if ord == 1:
    #         return (x >= 0).to(x.dtype)
    #     if ord == 2:
    #         return torch.zeros_like(x)
    #     raise NotImplementedError
    def grad(self, x, ord):
        if ord > 0:
            y = torch.sigmoid(self.beta*x)
            mask = (x < self.threshold).to(x.dtype)
        if ord == 1:
            return y*mask + (1-mask)
        if ord == 2:
            return y * (1 - y)*self.beta * mask
        raise NotImplementedError


class Normalise(ActivationFunc):
    def __init__(self, eps=1e-12):
        super(Normalise, self).__init__()
        self.eps = eps

    def __call__(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    def grad(self, x, ord):
        raise NotImplementedError


class ConcatFunc(ActivationFunc):
    def __init__(self, funcs, split_sizes, dim=-1):
        super(ConcatFunc, self).__init__()
        self.funcs = funcs
        self.split_sizes = split_sizes
        self.dim = dim

    def __call__(self, x):
        return torch.cat([_f(_x) for _f, _x in zip(self.funcs, torch.split(x, self.split_sizes, self.dim))], self.dim)

    def grad(self, x, ord):
        return torch.cat([_f.grad(_x, ord) for _f, _x in zip(self.funcs, torch.split(x, self.split_sizes, self.dim))],
                         self.dim)

### initialising SIREN ###

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-30 / num_input, 30 / num_input)

### end SIREN ###

def _parse_activation(string):
    name2func = {
        'none': ActivationFunc(),
        'sigmoid': Sigmoid(),
        'tanh': Tanh(),
        'relu': LeakyReLU(0),
        'leaky_relu': LeakyReLU(0.01),
        'lrelu': LeakyReLU(0.01),
        'softplus': SoftPlus(),
        'sin': Sinusoidal(),
        'siren': Sinusoidal(),
    }

    if len(string.split('|')) == 1:
        return name2func[string]

    funcs = []
    sizes = []
    for name_size in string.split('|'):
        name = name_size.rstrip('0123456789')
        size = int(name_size[len(name):])
        assert size > 0
        funcs.append(name2func[name])
        sizes.append(size)

    return ConcatFunc(funcs, sizes)


def _append_zero_rows(x, nrows=1):
    if x is None or nrows == 0:
        return x
    return torch.cat([x] + [torch.zeros_like(x[..., 0:1, :])] * nrows, dim=-2)

