import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
import math
import torch.nn.init as init
import torch.nn.functional as F
import collections.abc as container_abcs
from itertools import repeat
import logging

logger = logging.getLogger()

_DEFAULT_ALPHA = 1e-6
__all__ = ['iResBlock', 'ActNorm1d', 'ActNorm2d', 'SpectralNormLinear', 'SpectralNormConv2d', 'LopLinear', 'LopConv2d', 'get_linear', 'get_conv2d', 'InducedNormLinear', 'InducedNormConv2d', 'CouplingBlock', 'ChannelCouplingBlock', 'MaskedCouplingBlock', 'MovingBatchNorm1d', 'MovingBatchNorm2d', 'SqueezeLayer']

class SpectralNormLinear(nn.Module):

    def __init__(
        self, in_features, out_features, bias=True, coeff=0.97, n_iterations=None, atol=None, rtol=None, **unused_kwargs
    ):
        del unused_kwargs
        super(SpectralNormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        h, w = self.weight.shape
        self.register_buffer('scale', torch.tensor(0.))
        self.register_buffer('u', F.normalize(self.weight.new_empty(h).normal_(0, 1), dim=0))
        self.register_buffer('v', F.normalize(self.weight.new_empty(w).normal_(0, 1), dim=0))
        self.compute_weight(True, 200)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def compute_weight(self, update=True, n_iterations=None, atol=None, rtol=None):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol

        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')

        if n_iterations is None:
            n_iterations = 20000

        u = self.u
        v = self.v
        weight = self.weight
        if update:
            with torch.no_grad():
                itrs_used = 0.
                for _ in range(n_iterations):
                    old_v = v.clone()
                    old_u = u.clone()
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = F.normalize(torch.mv(weight.t(), u), dim=0, out=v)
                    u = F.normalize(torch.mv(weight, v), dim=0, out=u)
                    itrs_used = itrs_used + 1
                    if atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                if itrs_used > 0:
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight, v))
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight

    def forward(self, input):
        weight = self.compute_weight(update=self.training)
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, coeff={}, n_iters={}, atol={}, rtol={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.coeff, self.n_iterations, self.atol,
            self.rtol
        )


class SpectralNormConv2d(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True, coeff=0.97, n_iterations=None,
        atol=None, rtol=None, **unused_kwargs
    ):
        del unused_kwargs
        super(SpectralNormConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.initialized = False
        self.register_buffer('spatial_dims', torch.tensor([1., 1.]))
        self.register_buffer('scale', torch.tensor(0.))

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _initialize_u_v(self):
        if self.kernel_size == (1, 1):
            self.register_buffer('u', F.normalize(self.weight.new_empty(self.out_channels).normal_(0, 1), dim=0))
            self.register_buffer('v', F.normalize(self.weight.new_empty(self.in_channels).normal_(0, 1), dim=0))
        else:
            c, h, w = self.in_channels, int(self.spatial_dims[0].item()), int(self.spatial_dims[1].item())
            with torch.no_grad():
                num_input_dim = c * h * w
                v = F.normalize(torch.randn(num_input_dim).to(self.weight), dim=0, eps=1e-12)
                # forward call to infer the shape
                u = F.conv2d(v.reshape(1, c, h, w), self.weight, stride=self.stride, padding=self.padding, bias=None)
                num_output_dim = u.shape[0] * u.shape[1] * u.shape[2] * u.shape[3]
                self.out_shape = u.shape
                # overwrite u with random init
                u = F.normalize(torch.randn(num_output_dim).to(self.weight), dim=0, eps=1e-12)

                self.register_buffer('u', u)
                self.register_buffer('v', v)

    def compute_weight(self, update=True, n_iterations=None):
        if not self.initialized:
            self._initialize_u_v()
            self.initialized = True

        if self.kernel_size == (1, 1):
            return self._compute_weight_1x1(update, n_iterations)
        else:
            return self._compute_weight_kxk(update, n_iterations)

    def _compute_weight_1x1(self, update=True, n_iterations=None, atol=None, rtol=None):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol

        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')

        if n_iterations is None:
            n_iterations = 20000

        u = self.u
        v = self.v
        weight = self.weight.reshape(self.out_channels, self.in_channels)
        if update:
            with torch.no_grad():
                itrs_used = 0
                for _ in range(n_iterations):
                    old_v = v.clone()
                    old_u = u.clone()
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = F.normalize(torch.mv(weight.t(), u), dim=0, out=v)
                    u = F.normalize(torch.mv(weight, v), dim=0, out=u)
                    itrs_used = itrs_used + 1
                    if atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                if itrs_used > 0:
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight, v))
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight.reshape(self.out_channels, self.in_channels, 1, 1)

    def _compute_weight_kxk(self, update=True, n_iterations=None, atol=None, rtol=None):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol

        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')

        if n_iterations is None:
            n_iterations = 20000

        u = self.u
        v = self.v
        weight = self.weight
        c, h, w = self.in_channels, int(self.spatial_dims[0].item()), int(self.spatial_dims[1].item())
        if update:
            with torch.no_grad():
                itrs_used = 0
                for _ in range(n_iterations):
                    old_u = u.clone()
                    old_v = v.clone()
                    v_s = F.conv_transpose2d(
                        u.reshape(self.out_shape), weight, stride=self.stride, padding=self.padding, output_padding=0
                    )
                    v = F.normalize(v_s.reshape(-1), dim=0, out=v)
                    u_s = F.conv2d(v.reshape(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
                    u = F.normalize(u_s.reshape(-1), dim=0, out=u)
                    itrs_used = itrs_used + 1
                    if atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                if itrs_used > 0:
                    u = u.clone()
                    v = v.clone()

        weight_v = F.conv2d(v.reshape(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
        weight_v = weight_v.reshape(-1)
        sigma = torch.dot(u.reshape(-1), weight_v)
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight

    def forward(self, input):
        if not self.initialized: self.spatial_dims.copy_(torch.tensor(input.shape[2:4]).to(self.spatial_dims))
        weight = self.compute_weight(update=self.training)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, 1, 1)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}' ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        s += ', coeff={}, n_iters={}, atol={}, rtol={}'.format(self.coeff, self.n_iterations, self.atol, self.rtol)
        return s.format(**self.__dict__)


class LopLinear(nn.Linear):
    """Lipschitz constant defined using operator norms."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        coeff=0.97,
        domain=float('inf'),
        codomain=float('inf'),
        local_constraint=True,
        **unused_kwargs,
    ):
        del unused_kwargs
        super(LopLinear, self).__init__(in_features, out_features, bias)
        self.coeff = coeff
        self.domain = domain
        self.codomain = codomain
        self.local_constraint = local_constraint
        max_across_input_dims, self.norm_type = operator_norm_settings(self.domain, self.codomain)
        self.max_across_dim = 1 if max_across_input_dims else 0
        self.register_buffer('scale', torch.tensor(0.))

    def compute_weight(self):
        scale = _norm_except_dim(self.weight, self.norm_type, dim=self.max_across_dim)
        if not self.local_constraint: scale = scale.max()
        with torch.no_grad():
            self.scale.copy_(scale.max())

        # soft normalization
        factor = torch.max(torch.ones(1).to(self.weight), scale / self.coeff)

        return self.weight / factor

    def forward(self, input):
        weight = self.compute_weight()
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        s = super(LopLinear, self).extra_repr()
        return s + ', coeff={}, domain={}, codomain={}, local={}'.format(
            self.coeff, self.domain, self.codomain, self.local_constraint
        )


class LopConv2d(nn.Conv2d):
    """Lipschitz constant defined using operator norms."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        coeff=0.97,
        domain=float('inf'),
        codomain=float('inf'),
        local_constraint=True,
        **unused_kwargs,
    ):
        del unused_kwargs
        super(LopConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.coeff = coeff
        self.domain = domain
        self.codomain = codomain
        self.local_constraint = local_constraint
        max_across_input_dims, self.norm_type = operator_norm_settings(self.domain, self.codomain)
        self.max_across_dim = 1 if max_across_input_dims else 0
        self.register_buffer('scale', torch.tensor(0.))

    def compute_weight(self):
        scale = _norm_except_dim(self.weight, self.norm_type, dim=self.max_across_dim)
        if not self.local_constraint: scale = scale.max()
        with torch.no_grad():
            self.scale.copy_(scale.max())

        # soft normalization
        factor = torch.max(torch.ones(1).to(self.weight.device), scale / self.coeff)

        return self.weight / factor

    def forward(self, input):
        weight = self.compute_weight()
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, 1, 1)

    def extra_repr(self):
        s = super(LopConv2d, self).extra_repr()
        return s + ', coeff={}, domain={}, codomain={}, local={}'.format(
            self.coeff, self.domain, self.codomain, self.local_constraint
        )


class LipNormLinear(nn.Linear):
    """Lipschitz constant defined using operator norms."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        coeff=0.97,
        domain=float('inf'),
        codomain=float('inf'),
        local_constraint=True,
        **unused_kwargs,
    ):
        del unused_kwargs
        super(LipNormLinear, self).__init__(in_features, out_features, bias)
        self.coeff = coeff
        self.domain = domain
        self.codomain = codomain
        self.local_constraint = local_constraint
        max_across_input_dims, self.norm_type = operator_norm_settings(self.domain, self.codomain)
        self.max_across_dim = 1 if max_across_input_dims else 0

        # Initialize scale parameter.
        with torch.no_grad():
            w_scale = _norm_except_dim(self.weight, self.norm_type, dim=self.max_across_dim)
            if not self.local_constraint: w_scale = w_scale.max()
            self.scale = nn.Parameter(_logit(w_scale / self.coeff))

    def compute_weight(self):
        w_scale = _norm_except_dim(self.weight, self.norm_type, dim=self.max_across_dim)
        if not self.local_constraint: w_scale = w_scale.max()
        return self.weight / w_scale * torch.sigmoid(self.scale) * self.coeff

    def forward(self, input):
        weight = self.compute_weight()
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        s = super(LipNormLinear, self).extra_repr()
        return s + ', coeff={}, domain={}, codomain={}, local={}'.format(
            self.coeff, self.domain, self.codomain, self.local_constraint
        )


class LipNormConv2d(nn.Conv2d):
    """Lipschitz constant defined using operator norms."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        coeff=0.97,
        domain=float('inf'),
        codomain=float('inf'),
        local_constraint=True,
        **unused_kwargs,
    ):
        del unused_kwargs
        super(LipNormConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.coeff = coeff
        self.domain = domain
        self.codomain = codomain
        self.local_constraint = local_constraint
        max_across_input_dims, self.norm_type = operator_norm_settings(self.domain, self.codomain)
        self.max_across_dim = 1 if max_across_input_dims else 0

        # Initialize scale parameter.
        with torch.no_grad():
            w_scale = _norm_except_dim(self.weight, self.norm_type, dim=self.max_across_dim)
            if not self.local_constraint: w_scale = w_scale.max()
            self.scale = nn.Parameter(_logit(w_scale / self.coeff))

    def compute_weight(self):
        w_scale = _norm_except_dim(self.weight, self.norm_type, dim=self.max_across_dim)
        if not self.local_constraint: w_scale = w_scale.max()
        return self.weight / w_scale * torch.sigmoid(self.scale)

    def forward(self, input):
        weight = self.compute_weight()
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, 1, 1)

    def extra_repr(self):
        s = super(LipNormConv2d, self).extra_repr()
        return s + ', coeff={}, domain={}, codomain={}, local={}'.format(
            self.coeff, self.domain, self.codomain, self.local_constraint
        )


def _logit(p):
    p = torch.max(torch.ones(1) * 0.1, torch.min(torch.ones(1) * 0.9, p))
    return torch.log(p + 1e-10) + torch.log(1 - p + 1e-10)


def _norm_except_dim(w, norm_type, dim):
    if norm_type == 1 or norm_type == 2:
        return torch.norm_except_dim(w, norm_type, dim)
    elif norm_type == float('inf'):
        return _max_except_dim(w, dim)


def _max_except_dim(input, dim):
    maxed = input
    for axis in range(input.ndimension() - 1, dim, -1):
        maxed, _ = maxed.max(axis, keepdim=True)
    for axis in range(dim - 1, -1, -1):
        maxed, _ = maxed.max(axis, keepdim=True)
    return maxed


def operator_norm_settings(domain, codomain):
    if domain == 1 and codomain == 1:
        # maximum l1-norm of column
        max_across_input_dims = True
        norm_type = 1
    elif domain == 1 and codomain == 2:
        # maximum l2-norm of column
        max_across_input_dims = True
        norm_type = 2
    elif domain == 1 and codomain == float("inf"):
        # maximum l-inf norm of column
        max_across_input_dims = True
        norm_type = float("inf")
    elif domain == 2 and codomain == float("inf"):
        # maximum l2-norm of row
        max_across_input_dims = False
        norm_type = 2
    elif domain == float("inf") and codomain == float("inf"):
        # maximum l1-norm of row
        max_across_input_dims = False
        norm_type = 1
    else:
        raise ValueError('Unknown combination of domain "{}" and codomain "{}"'.format(domain, codomain))

    return max_across_input_dims, norm_type


def get_linear(in_features, out_features, bias=True, coeff=0.97, domain=None, codomain=None, **kwargs):
    _linear = InducedNormLinear
    if domain == 1:
        if codomain in [1, 2, float('inf')]:
            _linear = LopLinear
    elif codomain == float('inf'):
        if domain in [2, float('inf')]:
            _linear = LopLinear
    return _linear(in_features, out_features, bias, coeff, domain, codomain, **kwargs)


def get_conv2d(
    in_channels, out_channels, kernel_size, stride, padding, bias=True, coeff=0.97, domain=None, codomain=None, **kwargs
):
    _conv2d = InducedNormConv2d
    if domain == 1:
        if codomain in [1, 2, float('inf')]:
            _conv2d = LopConv2d
    elif codomain == float('inf'):
        if domain in [2, float('inf')]:
            _conv2d = LopConv2d
    return _conv2d(in_channels, out_channels, kernel_size, stride, padding, bias, coeff, domain, codomain, **kwargs)
    
class InducedNormLinear(nn.Module):

    def __init__(
        self, in_features, out_features, bias=True, coeff=0.97, domain=2, codomain=2, n_iterations=None, atol=None,
        rtol=None, zero_init=False, **unused_kwargs
    ):
        del unused_kwargs
        super(InducedNormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol
        self.domain = domain
        self.codomain = codomain
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(zero_init)

        with torch.no_grad():
            domain, codomain = self.compute_domain_codomain()

        h, w = self.weight.shape
        self.register_buffer('scale', torch.tensor(0.))
        self.register_buffer('u', normalize_u(self.weight.new_empty(h).normal_(0, 1), codomain))
        self.register_buffer('v', normalize_v(self.weight.new_empty(w).normal_(0, 1), domain))

        # Try different random seeds to find the best u and v.
        with torch.no_grad():
            self.compute_weight(True, n_iterations=200, atol=None, rtol=None)
            best_scale = self.scale.clone()
            best_u, best_v = self.u.clone(), self.v.clone()
            if not (domain == 2 and codomain == 2):
                for _ in range(10):
                    self.register_buffer('u', normalize_u(self.weight.new_empty(h).normal_(0, 1), codomain))
                    self.register_buffer('v', normalize_v(self.weight.new_empty(w).normal_(0, 1), domain))
                    self.compute_weight(True, n_iterations=200)
                    if self.scale > best_scale:
                        best_u, best_v = self.u.clone(), self.v.clone()
            self.u.copy_(best_u)
            self.v.copy_(best_v)

    def reset_parameters(self, zero_init=False):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if zero_init:
            # normalize cannot handle zero weight in some cases.
            self.weight.data.div_(1000)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def compute_domain_codomain(self):
        if torch.is_tensor(self.domain):
            domain = asym_squash(self.domain)
            codomain = asym_squash(self.codomain)
        else:
            domain, codomain = self.domain, self.codomain
        return domain, codomain

    def compute_one_iter(self):
        domain, codomain = self.compute_domain_codomain()
        u = self.u.detach()
        v = self.v.detach()
        weight = self.weight.detach()
        u = normalize_u(torch.mv(weight, v), codomain)
        v = normalize_v(torch.mv(weight.t(), u), domain)
        return torch.dot(u, torch.mv(weight, v))

    def compute_weight(self, update=True, n_iterations=None, atol=None, rtol=None):
        u = self.u
        v = self.v
        weight = self.weight

        if update:

            n_iterations = self.n_iterations if n_iterations is None else n_iterations
            atol = self.atol if atol is None else atol
            rtol = self.rtol if rtol is None else atol

            if n_iterations is None and (atol is None or rtol is None):
                raise ValueError('Need one of n_iteration or (atol, rtol).')

            max_itrs = 200
            if n_iterations is not None:
                max_itrs = n_iterations

            with torch.no_grad():
                domain, codomain = self.compute_domain_codomain()
                for _ in range(max_itrs):
                    # Algorithm from http://www.qetlab.com/InducedMatrixNorm.
                    if n_iterations is None and atol is not None and rtol is not None:
                        old_v = v.clone()
                        old_u = u.clone()

                    u = normalize_u(torch.mv(weight, v), codomain, out=u)
                    v = normalize_v(torch.mv(weight.t(), u), domain, out=v)

                    if n_iterations is None and atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                self.v.copy_(v)
                self.u.copy_(u)
                u = u.clone()
                v = v.clone()

        sigma = torch.dot(u, torch.mv(weight, v))
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight

    def forward(self, input):
        weight = self.compute_weight(update=False)
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        domain, codomain = self.compute_domain_codomain()
        return (
            'in_features={}, out_features={}, bias={}'
            ', coeff={}, domain={:.2f}, codomain={:.2f}, n_iters={}, atol={}, rtol={}, learnable_ord={}'.format(
                self.in_features, self.out_features, self.bias is not None, self.coeff, domain, codomain,
                self.n_iterations, self.atol, self.rtol, torch.is_tensor(self.domain)
            )
        )


class InducedNormConv2d(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True, coeff=0.97, domain=2, codomain=2,
        n_iterations=None, atol=None, rtol=None, **unused_kwargs
    ):
        del unused_kwargs
        super(InducedNormConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.domain = domain
        self.codomain = codomain
        self.atol = atol
        self.rtol = rtol
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.register_buffer('initialized', torch.tensor(0))
        self.register_buffer('spatial_dims', torch.tensor([1., 1.]))
        self.register_buffer('scale', torch.tensor(0.))
        self.register_buffer('u', self.weight.new_empty(self.out_channels))
        self.register_buffer('v', self.weight.new_empty(self.in_channels))

    def compute_domain_codomain(self):
        if torch.is_tensor(self.domain):
            domain = asym_squash(self.domain)
            codomain = asym_squash(self.codomain)
        else:
            domain, codomain = self.domain, self.codomain
        return domain, codomain

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _initialize_u_v(self):
        with torch.no_grad():
            domain, codomain = self.compute_domain_codomain()
            if self.kernel_size == (1, 1):
                self.u.resize_(self.out_channels).normal_(0, 1)
                self.u.copy_(normalize_u(self.u, codomain))
                self.v.resize_(self.in_channels).normal_(0, 1)
                self.v.copy_(normalize_v(self.v, domain))
            else:
                c, h, w = self.in_channels, int(self.spatial_dims[0].item()), int(self.spatial_dims[1].item())
                with torch.no_grad():
                    num_input_dim = c * h * w
                    self.v.resize_(num_input_dim).normal_(0, 1)
                    self.v.copy_(normalize_v(self.v, domain))
                    # forward call to infer the shape
                    u = F.conv2d(
                        self.v.reshape(1, c, h, w), self.weight, stride=self.stride, padding=self.padding, bias=None
                    )
                    num_output_dim = u.shape[0] * u.shape[1] * u.shape[2] * u.shape[3]
                    # overwrite u with random init
                    self.u.resize_(num_output_dim).normal_(0, 1)
                    self.u.copy_(normalize_u(self.u, codomain))

            self.initialized.fill_(1)

            # Try different random seeds to find the best u and v.
            self.compute_weight(True)
            best_scale = self.scale.clone()
            best_u, best_v = self.u.clone(), self.v.clone()
            if not (domain == 2 and codomain == 2):
                for _ in range(10):
                    if self.kernel_size == (1, 1):
                        self.u.copy_(normalize_u(self.weight.new_empty(self.out_channels).normal_(0, 1), codomain))
                        self.v.copy_(normalize_v(self.weight.new_empty(self.in_channels).normal_(0, 1), domain))
                    else:
                        self.u.copy_(normalize_u(torch.randn(num_output_dim).to(self.weight), codomain))
                        self.v.copy_(normalize_v(torch.randn(num_input_dim).to(self.weight), domain))
                    self.compute_weight(True, n_iterations=200)
                    if self.scale > best_scale:
                        best_u, best_v = self.u.clone(), self.v.clone()
            self.u.copy_(best_u)
            self.v.copy_(best_v)

    def compute_one_iter(self):
        if not self.initialized:
            raise ValueError('Layer needs to be initialized first.')
        domain, codomain = self.compute_domain_codomain()
        if self.kernel_size == (1, 1):
            u = self.u.detach()
            v = self.v.detach()
            weight = self.weight.detach().reshape(self.out_channels, self.in_channels)
            u = normalize_u(torch.mv(weight, v), codomain)
            v = normalize_v(torch.mv(weight.t(), u), domain)
            return torch.dot(u, torch.mv(weight, v))
        else:
            u = self.u.detach()
            v = self.v.detach()
            weight = self.weight.detach()
            c, h, w = self.in_channels, int(self.spatial_dims[0].item()), int(self.spatial_dims[1].item())
            u_s = F.conv2d(v.reshape(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
            out_shape = u_s.shape
            u = normalize_u(u_s.reshape(-1), codomain)
            v_s = F.conv_transpose2d(
                u.reshape(out_shape), weight, stride=self.stride, padding=self.padding, output_padding=0
            )
            v = normalize_v(v_s.reshape(-1), domain)
            weight_v = F.conv2d(v.reshape(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
            return torch.dot(u.reshape(-1), weight_v.reshape(-1))

    def compute_weight(self, update=True, n_iterations=None, atol=None, rtol=None):
        if not self.initialized:
            self._initialize_u_v()

        if self.kernel_size == (1, 1):
            return self._compute_weight_1x1(update, n_iterations, atol, rtol)
        else:
            return self._compute_weight_kxk(update, n_iterations, atol, rtol)

    def _compute_weight_1x1(self, update=True, n_iterations=None, atol=None, rtol=None):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol

        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')

        max_itrs = 200
        if n_iterations is not None:
            max_itrs = n_iterations

        u = self.u
        v = self.v
        weight = self.weight.reshape(self.out_channels, self.in_channels)
        if update:
            with torch.no_grad():
                domain, codomain = self.compute_domain_codomain()
                itrs_used = 0
                for _ in range(max_itrs):
                    old_v = v.clone()
                    old_u = u.clone()

                    u = normalize_u(torch.mv(weight, v), codomain, out=u)
                    v = normalize_v(torch.mv(weight.t(), u), domain, out=v)

                    itrs_used = itrs_used + 1

                    if n_iterations is None and atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                if itrs_used > 0:
                    if domain != 1 and domain != 2:
                        self.v.copy_(v)
                    if codomain != 2 and codomain != float('inf'):
                        self.u.copy_(u)
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight, v))
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight.reshape(self.out_channels, self.in_channels, 1, 1)

    def _compute_weight_kxk(self, update=True, n_iterations=None, atol=None, rtol=None):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol

        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')

        max_itrs = 200
        if n_iterations is not None:
            max_itrs = n_iterations

        u = self.u
        v = self.v
        weight = self.weight
        c, h, w = self.in_channels, int(self.spatial_dims[0].item()), int(self.spatial_dims[1].item())
        if update:
            with torch.no_grad():
                domain, codomain = self.compute_domain_codomain()
                itrs_used = 0
                for _ in range(max_itrs):
                    old_u = u.clone()
                    old_v = v.clone()

                    u_s = F.conv2d(v.reshape(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
                    out_shape = u_s.shape
                    u = normalize_u(u_s.reshape(-1), codomain, out=u)

                    v_s = F.conv_transpose2d(
                        u.reshape(out_shape), weight, stride=self.stride, padding=self.padding, output_padding=0
                    )
                    v = normalize_v(v_s.reshape(-1), domain, out=v)

                    itrs_used = itrs_used + 1
                    if n_iterations is None and atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                if itrs_used > 0:
                    if domain != 2:
                        self.v.copy_(v)
                    if codomain != 2:
                        self.u.copy_(u)
                    v = v.clone()
                    u = u.clone()

        weight_v = F.conv2d(v.reshape(1, c, h, w), weight, stride=self.stride, padding=self.padding, bias=None)
        weight_v = weight_v.reshape(-1)
        sigma = torch.dot(u.reshape(-1), weight_v)
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight

    def forward(self, input):
        if not self.initialized: self.spatial_dims.copy_(torch.tensor(input.shape[2:4]).to(self.spatial_dims))
        weight = self.compute_weight(update=False)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, 1, 1)

    def extra_repr(self):
        domain, codomain = self.compute_domain_codomain()
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}' ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        s += ', coeff={}, domain={:.2f}, codomain={:.2f}, n_iters={}, atol={}, rtol={}, learnable_ord={}'.format(
            self.coeff, domain, codomain, self.n_iterations, self.atol, self.rtol, torch.is_tensor(self.domain)
        )
        return s.format(**self.__dict__)


def projmax_(v):
    """Inplace argmax on absolute value."""
    ind = torch.argmax(torch.abs(v))
    v.zero_()
    v[ind] = 1
    return v


def normalize_v(v, domain, out=None):
    if not torch.is_tensor(domain) and domain == 2:
        v = F.normalize(v, p=2, dim=0, out=out)
    elif domain == 1:
        v = projmax_(v)
    else:
        vabs = torch.abs(v)
        vph = v / vabs
        vph[torch.isnan(vph)] = 1
        vabs = vabs / torch.max(vabs)
        vabs = vabs**(1 / (domain - 1))
        v = vph * vabs / vector_norm(vabs, domain)
    return v


def normalize_u(u, codomain, out=None):
    if not torch.is_tensor(codomain) and codomain == 2:
        u = F.normalize(u, p=2, dim=0, out=out)
    elif codomain == float('inf'):
        u = projmax_(u)
    else:
        uabs = torch.abs(u)
        uph = u / uabs
        uph[torch.isnan(uph)] = 1
        uabs = uabs / torch.max(uabs)
        uabs = uabs**(codomain - 1)
        if codomain == 1:
            u = uph * uabs / vector_norm(uabs, float('inf'))
        else:
            u = uph * uabs / vector_norm(uabs, codomain / (codomain - 1))
    return u


def vector_norm(x, p):
    x = x.reshape(-1)
    return torch.sum(x**p)**(1 / p)


def leaky_elu(x, a=0.3):
    return a * x + (1 - a) * F.elu(x)


def asym_squash(x):
    return torch.tanh(-leaky_elu(-x + 0.5493061829986572)) * 2 + 3


# def asym_squash(x):
#     return torch.tanh(x) / 2. + 2.


def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

if __name__ == '__main__':

    p = nn.Parameter(torch.tensor(2.1))

    m = InducedNormConv2d(10, 2, 3, 1, 1, atol=1e-3, rtol=1e-3, domain=p, codomain=p)
    W = m.compute_weight()

    m.compute_one_iter().backward()
    print(p.grad)

    # m.weight.data.copy_(W)
    # W = m.compute_weight().cpu().detach().numpy()
    # import numpy as np
    # print(
    #     '{} {} {}'.format(
    #         np.linalg.norm(W, ord=2, axis=(0, 1)),
    #         '>' if np.linalg.norm(W, ord=2, axis=(0, 1)) > m.scale else '<',
    #         m.scale,
    #     )
    # )    

class ActNormNd(nn.Module):

    def __init__(self, num_features, eps=1e-12):
        super(ActNormNd, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer('initialized', torch.tensor(0))

    @property
    def shape(self):
        raise NotImplementedError

    def forward(self, x, logpx=None):
        c = x.size(1)

        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_t = x.transpose(0, 1).contiguous().reshape(c, -1)
                batch_mean = torch.mean(x_t, dim=1)
                batch_var = torch.var(x_t, dim=1)

                # for numerical issues
                batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))

                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var))
                self.initialized.fill_(1)

        bias = self.bias.reshape(*self.shape).expand_as(x)
        weight = self.weight.reshape(*self.shape).expand_as(x)

        y = (x + bias) * torch.exp(weight)

        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(x)

    def reverse(self, y, logpy=None):
        assert self.initialized
        bias = self.bias.reshape(*self.shape).expand_as(y)
        weight = self.weight.reshape(*self.shape).expand_as(y)

        x = y * torch.exp(-weight) - bias

        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x)

    def _logdetgrad(self, x):
        return self.weight.reshape(*self.shape).expand(*x.size()).contiguous().reshape(x.size(0), -1).sum(1, keepdim=True)

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))

class ActNorm1d(ActNormNd):

    @property
    def shape(self):
        return [1, -1]


class ActNorm2d(ActNormNd):

    @property
    def shape(self):
        return [1, -1, 1, 1]


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None):
        if logpx is None:
            for i in range(len(self.chain)):
                x = self.chain[i](x)
            return x
        else:
            for i in range(len(self.chain)):
                x, logpx = self.chain[i](x, logpx)
            return x, logpx

    def reverse(self, y, logpy=None):
        if logpy is None:
            for i in range(len(self.chain) - 1, -1, -1):
                y = self.chain[i].reverse(y)
            return y
        else:
            for i in range(len(self.chain) - 1, -1, -1):
                y, logpy = self.chain[i].reverse(y, logpy)
            return y, logpy


class Inverse(nn.Module):

    def __init__(self, flow):
        super(Inverse, self).__init__()
        self.flow = flow

    def forward(self, x, logpx=None):
        return self.flow.reverse(x, logpx)

    def reverse(self, y, logpy=None):
        return self.flow.forward(y, logpy)
    
class CouplingBlock(nn.Module):
    """Basic coupling layer for Tensors of shape (n,d).

    Forward computation:
        y_a = x_a
        y_b = y_b * exp(s(x_a)) + t(x_a)
    Inverse computation:
        x_a = y_a
        x_b = (y_b - t(y_a)) * exp(-s(y_a))
    """

    def __init__(self, dim, nnet, swap=False):
        """
        Args:
            s (nn.Module)
            t (nn.Module)
        """
        super(CouplingBlock, self).__init__()
        assert (dim % 2 == 0)
        self.d = dim // 2
        self.nnet = nnet
        self.swap = swap

    def func_s_t(self, x):
        f = self.nnet(x)
        s = f[:, :self.d]
        t = f[:, self.d:]
        return s, t

    def forward(self, x, logpx=None):
        """Forward computation of a simple coupling split on the axis=1.
        """
        x_a = x[:, :self.d] if not self.swap else x[:, self.d:]
        x_b = x[:, self.d:] if not self.swap else x[:, :self.d]
        y_a, y_b, logdetgrad = self._forward_computation(x_a, x_b)
        y = [y_a, y_b] if not self.swap else [y_b, y_a]

        if logpx is None:
            return torch.cat(y, dim=1)
        else:
            return torch.cat(y, dim=1), logpx - logdetgrad.reshape(x.size(0), -1).sum(1, keepdim=True)

    def reverse(self, y, logpy=None):
        """reverse computation of a simple coupling split on the axis=1.
        """
        y_a = y[:, :self.d] if not self.swap else y[:, self.d:]
        y_b = y[:, self.d:] if not self.swap else y[:, :self.d]
        x_a, x_b, logdetgrad = self._reverse_computation(y_a, y_b)
        x = [x_a, x_b] if not self.swap else [x_b, x_a]
        if logpy is None:
            return torch.cat(x, dim=1)
        else:
            return torch.cat(x, dim=1), logpy + logdetgrad

    def _forward_computation(self, x_a, x_b):
        y_a = x_a
        s_a, t_a = self.func_s_t(x_a)
        scale = torch.sigmoid(s_a + 2.)
        y_b = x_b * scale + t_a
        logdetgrad = self._logdetgrad(scale)
        return y_a, y_b, logdetgrad

    def _reverse_computation(self, y_a, y_b):
        x_a = y_a
        s_a, t_a = self.func_s_t(y_a)
        scale = torch.sigmoid(s_a + 2.)
        x_b = (y_b - t_a) / scale
        logdetgrad = self._logdetgrad(scale)
        return x_a, x_b, logdetgrad

    def _logdetgrad(self, scale):
        """
        Returns:
            Tensor (N, 1): containing ln |det J| where J is the jacobian
        """
        return torch.log(scale).reshape(scale.shape[0], -1).sum(1, keepdim=True)

    def extra_repr(self):
        return 'dim={d}, swap={swap}'.format(**self.__dict__)


class ChannelCouplingBlock(CouplingBlock):
    """Channel-wise coupling layer for images.
    """

    def __init__(self, dim, nnet, mask_type='channel0'):
        if mask_type == 'channel0':
            swap = False
        elif mask_type == 'channel1':
            swap = True
        else:
            raise ValueError('Unknown mask type.')
        super(ChannelCouplingBlock, self).__init__(dim, nnet, swap)
        self.mask_type = mask_type

    def extra_repr(self):
        return 'dim={d}, mask_type={mask_type}'.format(**self.__dict__)


class MaskedCouplingBlock(nn.Module):
    """Coupling layer for images implemented using masks.
    """

    def __init__(self, dim, nnet, mask_type='checkerboard0'):
        nn.Module.__init__(self)
        self.d = dim
        self.nnet = nnet
        self.mask_type = mask_type

    def func_s_t(self, x):
        f = self.nnet(x)
        s = torch.sigmoid(f[:, :self.d] + 2.)
        t = f[:, self.d:]
        return s, t

    def forward(self, x, logpx=None):
        # get mask
        b = get_mask(x, mask_type=self.mask_type)

        # masked forward
        x_a = b * x
        s, t = self.func_s_t(x_a)
        y = (x * s + t) * (1 - b) + x_a

        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(s, b)

    def reverse(self, y, logpy=None):
        # get mask
        b = get_mask(y, mask_type=self.mask_type)

        # masked forward
        y_a = b * y
        s, t = self.func_s_t(y_a)
        x = y_a + (1 - b) * (y - t) / s

        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(s, b)

    def _logdetgrad(self, s, mask):
        return torch.log(s).mul_(1 - mask).reshape(s.shape[0], -1).sum(1, keepdim=True)

    def extra_repr(self):
        return 'dim={d}, mask_type={mask_type}'.format(**self.__dict__)




class ZeroMeanTransform(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, logpx=None):
        x = x - .5
        if logpx is None:
            return x
        return x, logpx

    def reverse(self, y, logpy=None):
        y = y + .5
        if logpy is None:
            return y
        return y, logpy


class Normalize(nn.Module):

    def __init__(self, mean, std):
        nn.Module.__init__(self)
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x, logpx=None):
        y = x.clone()
        c = len(self.mean)
        y[:, :c].sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(x)

    def reverse(self, y, logpy=None):
        x = y.clone()
        c = len(self.mean)
        x[:, :c].mul_(self.std[None, :, None, None]).add_(self.mean[None, :, None, None])
        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x)

    def _logdetgrad(self, x):
        logdetgrad = (
            self.std.abs().log().mul_(-1).reshape(1, -1, 1, 1).expand(x.shape[0], len(self.std), x.shape[2], x.shape[3])
        )
        return logdetgrad.reshape(x.shape[0], -1).sum(-1, keepdim=True)


class LogitTransform(nn.Module):
    """
    The proprocessing step used in Real NVP:
    y = sigmoid(x) - a / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """

    def __init__(self, alpha=_DEFAULT_ALPHA):
        nn.Module.__init__(self)
        self.alpha = alpha

    def forward(self, x, logpx=None):
        s = self.alpha + (1 - 2 * self.alpha) * x
        y = torch.log(s) - torch.log(1 - s)
        if logpx is None:
            return y
        return y, logpx - self._logdetgrad(x).reshape(x.size(0), -1).sum(1, keepdim=True)

    def reverse(self, y, logpy=None):
        x = (torch.sigmoid(y) - self.alpha) / (1 - 2 * self.alpha)
        if logpy is None:
            return x
        return x, logpy + self._logdetgrad(x).reshape(x.size(0), -1).sum(1, keepdim=True)

    def _logdetgrad(self, x):
        s = self.alpha + (1 - 2 * self.alpha) * x
        logdetgrad = -torch.log(s - s * s) + math.log(1 - 2 * self.alpha)
        return logdetgrad

    def __repr__(self):
        return ('{name}({alpha})'.format(name=self.__class__.__name__, **self.__dict__))

class InvertibleLinear(nn.Module):

    def __init__(self, dim):
        super(InvertibleLinear, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.eye(dim)[torch.randperm(dim)])

    def forward(self, x, logpx=None):
        y = F.linear(x, self.weight)
        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad

    def reverse(self, y, logpy=None):
        x = F.linear(y, self.weight.reverse())
        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad

    @property
    def _logdetgrad(self):
        return torch.log(torch.abs(torch.det(self.weight)))

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


class InvertibleConv2d(nn.Module):

    def __init__(self, dim):
        super(InvertibleConv2d, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.eye(dim)[torch.randperm(dim)])

    def forward(self, x, logpx=None):
        y = F.conv2d(x, self.weight.reshape(self.dim, self.dim, 1, 1))
        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad.expand_as(logpx) * x.shape[2] * x.shape[3]

    def reverse(self, y, logpy=None):
        x = F.conv2d(y, self.weight.reverse().reshape(self.dim, self.dim, 1, 1))
        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad.expand_as(logpy) * x.shape[2] * x.shape[3]

    @property
    def _logdetgrad(self):
        return torch.log(torch.abs(torch.det(self.weight)))

    def extra_repr(self):
        return 'dim={}'.format(self.dim)

class iResBlock(nn.Module):

    def __init__(
        self,
        nnet,
        geom_p=0.5,
        lamb=2.,
        n_power_series=None,
        exact_trace=False,
        brute_force=False,
        n_samples=1,
        n_exact_terms=2,
        n_dist='geometric',
        neumann_grad=True,
        grad_in_forward=False,
    ):
        """
        Args:
            nnet: a nn.Module
            n_power_series: number of power series. If not None, uses a biased approximation to logdet.
            exact_trace: if False, uses a Hutchinson trace estimator. Otherwise computes the exact full Jacobian.
            brute_force: Computes the exact logdet. Only available for 2D inputs.
        """
        nn.Module.__init__(self)
        self.nnet = nnet
        self.n_dist = n_dist
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1. - geom_p)))
        self.lamb = nn.Parameter(torch.tensor(lamb))
        self.n_samples = n_samples
        self.n_power_series = n_power_series
        self.exact_trace = exact_trace
        self.brute_force = brute_force
        self.n_exact_terms = n_exact_terms
        self.grad_in_forward = grad_in_forward
        self.neumann_grad = neumann_grad

        # store the samples of n.
        self.register_buffer('last_n_samples', torch.zeros(self.n_samples))
        self.register_buffer('last_firmom', torch.zeros(1))
        self.register_buffer('last_secmom', torch.zeros(1))

    def forward(self, x, logpx=None):
        if logpx is None:
            y = x + self.nnet(x)
            return y
        else:
            g, logdetgrad = self._logdetgrad(x)
            return x + g, logpx - logdetgrad

    def reverse(self, y, logpy=None):
        x = self._reverse_fixed_point(y)
        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x)[1]

    def _reverse_fixed_point(self, y, atol=1e-5, rtol=1e-5):
        x, x_prev = y - self.nnet(y), y
        i = 0
        tol = atol + y.abs() * rtol
        while not torch.all((x - x_prev)**2 / tol < 1):
            x, x_prev = y - self.nnet(x), x
            i += 1
            if i > 1000:
                logger.info('Iterations exceeded 1000 for reverse.')
                break
        return x

    def _logdetgrad(self, x):
        """Returns g(x) and logdet|d(x+g(x))/dx|."""

        with torch.enable_grad():
            if (self.brute_force or not self.training) and (x.ndimension() == 2 and x.shape[1] == 2):
                ###########################################
                # Brute-force compute Jacobian determinant.
                ###########################################
                x = x.requires_grad_(True)
                g = self.nnet(x)
                # Brute-force logdet only available for 2D.
                jac = batch_jacobian(g, x)
                batch_dets = (jac[:, 0, 0] + 1) * (jac[:, 1, 1] + 1) - jac[:, 0, 1] * jac[:, 1, 0]
                return g, torch.log(torch.abs(batch_dets)).reshape(-1, 1)

            if self.n_dist == 'geometric':
                geom_p = torch.sigmoid(self.geom_p).item()
                sample_fn = lambda m: geometric_sample(geom_p, m)
                rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)
            elif self.n_dist == 'poisson':
                lamb = self.lamb.item()
                sample_fn = lambda m: poisson_sample(lamb, m)
                rcdf_fn = lambda k, offset: poisson_1mcdf(lamb, k, offset)

            if self.training:
                if self.n_power_series is None:
                    # Unbiased estimation.
                    lamb = self.lamb.item()
                    n_samples = sample_fn(self.n_samples)
                    n_power_series = max(n_samples) + self.n_exact_terms
                    coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms) * \
                        sum(n_samples >= k - self.n_exact_terms) / len(n_samples)
                else:
                    # Truncated estimation.
                    n_power_series = self.n_power_series
                    coeff_fn = lambda k: 1.
            else:
                # Unbiased estimation with more exact terms.
                lamb = self.lamb.item()
                n_samples = sample_fn(self.n_samples)
                n_power_series = max(n_samples) + 20
                coeff_fn = lambda k: 1 / rcdf_fn(k, 20) * \
                    sum(n_samples >= k - 20) / len(n_samples)

            if not self.exact_trace:
                ####################################
                # Power series with trace estimator.
                ####################################
                vareps = torch.randn_like(x)

                # Choose the type of estimator.
                if self.training and self.neumann_grad:
                    estimator_fn = neumann_logdet_estimator
                else:
                    estimator_fn = basic_logdet_estimator

                # Do backprop-in-forward to save memory.
                if self.training and self.grad_in_forward:
                    g, logdetgrad = mem_eff_wrapper(
                        estimator_fn, self.nnet, x, n_power_series, vareps, coeff_fn, self.training
                    )
                else:
                    x = x.requires_grad_(True)
                    g = self.nnet(x)
                    logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, self.training)
            else:
                ############################################
                # Power series with exact trace computation.
                ############################################
                x = x.requires_grad_(True)
                g = self.nnet(x)
                jac = batch_jacobian(g, x)
                logdetgrad = batch_trace(jac)
                jac_k = jac
                for k in range(2, n_power_series + 1):
                    jac_k = torch.bmm(jac, jac_k)
                    logdetgrad = logdetgrad + (-1)**(k + 1) / k * coeff_fn(k) * batch_trace(jac_k)

            if self.training and self.n_power_series is None:
                self.last_n_samples.copy_(torch.tensor(n_samples).to(self.last_n_samples))
                estimator = logdetgrad.detach()
                self.last_firmom.copy_(torch.mean(estimator).to(self.last_firmom))
                self.last_secmom.copy_(torch.mean(estimator**2).to(self.last_secmom))
            return g, logdetgrad.reshape(-1, 1)

    def extra_repr(self):
        return 'dist={}, n_samples={}, n_power_series={}, neumann_grad={}, exact_trace={}, brute_force={}'.format(
            self.n_dist, self.n_samples, self.n_power_series, self.neumann_grad, self.exact_trace, self.brute_force
        )


def batch_jacobian(g, x):
    jac = []
    for d in range(g.shape[1]):
        jac.append(torch.autograd.grad(torch.sum(g[:, d]), x, create_graph=True)[0].reshape(x.shape[0], 1, x.shape[1]))
    return torch.cat(jac, 1)


def batch_trace(M):
    return M.reshape(M.shape[0], -1)[:, ::M.shape[1] + 1].sum(1)


#####################
# Logdet Estimators
#####################
class MemoryEfficientLogDetEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training, *g_params):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = gnet(x)
            ctx.g = g
            ctx.x = x
            logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, training)

            if training:
                grad_x, *grad_params = torch.autograd.grad(
                    logdetgrad.sum(), (x,) + g_params, retain_graph=True, allow_unused=True
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        return safe_detach(g), safe_detach(logdetgrad)

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x

            # Precomputed gradients.
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]

            dg_x, *dg_params = torch.autograd.grad(g, [x] + g_params, grad_g, allow_unused=True)

        # Update based on gradient from logdetgrad.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple([g.mul_(dL) if g is not None else None for g in grad_params])

        # Update based on gradient from g.
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple([dg.add_(djac) if djac is not None else dg for dg, djac in zip(dg_params, grad_params)])

        return (None, None, grad_x, None, None, None, None) + grad_params


def basic_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    logdetgrad = torch.tensor(0.).to(x)
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[0]
        tr = torch.sum(vjp.reshape(x.shape[0], -1) * vareps.reshape(x.shape[0], -1), 1)
        delta = (-1)**(k + 1) / k * coeff_fn(k) * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad


def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1)**k * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(vjp_jac.reshape(x.shape[0], -1) * vareps.reshape(x.shape[0], -1), 1)
    return logdetgrad


def mem_eff_wrapper(estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(gnet, nn.Module):
        raise ValueError('g is required to be an instance of nn.Module.')

    return MemoryEfficientLogDetEstimator.apply(
        estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training, *list(gnet.parameters())
    )


# -------- Helper distribution functions --------
# These take python ints or floats, not PyTorch tensors.


def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)


def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p)**max(k - 1, 0)


def poisson_sample(lamb, n_samples):
    return np.random.poisson(lamb, n_samples)


def poisson_1mcdf(lamb, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    sum = 1.
    for i in range(1, k):
        sum += lamb**i / math.factorial(i)
    return 1 - np.exp(-lamb) * sum


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


# -------------- Helper functions --------------


def safe_detach(tensor):
    return tensor.detach().requires_grad_(tensor.requires_grad)


def _flatten(sequence):
    flat = [p.reshape(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def _flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [p.reshape(-1) if p is not None else torch.zeros_like(q).reshape(-1) for p, q in zip(sequence, like_sequence)]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])

def _get_checkerboard_mask(x, swap=False):
    n, c, h, w = x.size()

    H = ((h - 1) // 2 + 1) * 2  # H = h + 1 if h is odd and h if h is even
    W = ((w - 1) // 2 + 1) * 2

    # construct checkerboard mask
    if not swap:
        mask = torch.Tensor([[1, 0], [0, 1]]).repeat(H // 2, W // 2)
    else:
        mask = torch.Tensor([[0, 1], [1, 0]]).repeat(H // 2, W // 2)
    mask = mask[:h, :w]
    mask = mask.contiguous().reshape(1, 1, h, w).expand(n, c, h, w).type_as(x.data)

    return mask


def _get_channel_mask(x, swap=False):
    n, c, h, w = x.size()
    assert (c % 2 == 0)

    # construct channel-wise mask
    mask = torch.zeros(x.size())
    if not swap:
        mask[:, :c // 2] = 1
    else:
        mask[:, c // 2:] = 1
    return mask


def get_mask(x, mask_type=None):
    if mask_type is None:
        return torch.zeros(x.size()).to(x)
    elif mask_type == 'channel0':
        return _get_channel_mask(x, swap=False)
    elif mask_type == 'channel1':
        return _get_channel_mask(x, swap=True)
    elif mask_type == 'checkerboard0':
        return _get_checkerboard_mask(x, swap=False)
    elif mask_type == 'checkerboard1':
        return _get_checkerboard_mask(x, swap=True)
    else:
        raise ValueError('Unknown mask type {}'.format(mask_type))

class MovingBatchNormNd(nn.Module):

    def __init__(self, num_features, eps=1e-4, decay=0.1, bn_lag=0., affine=True):
        super(MovingBatchNormNd, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.decay = decay
        self.bn_lag = bn_lag
        self.register_buffer('step', torch.zeros(1))
        if self.affine:
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.reset_parameters()

    @property
    def shape(self):
        raise NotImplementedError

    def reset_parameters(self):
        self.running_mean.zero_()
        if self.affine:
            self.bias.data.zero_()

    def forward(self, x, logpx=None):
        c = x.size(1)
        used_mean = self.running_mean.clone().detach()

        if self.training:
            # compute batch statistics
            x_t = x.transpose(0, 1).contiguous().reshape(c, -1)
            batch_mean = torch.mean(x_t, dim=1)

            # moving average
            if self.bn_lag > 0:
                used_mean = batch_mean - (1 - self.bn_lag) * (batch_mean - used_mean.detach())
                used_mean /= (1. - self.bn_lag**(self.step[0] + 1))

            # update running estimates
            self.running_mean -= self.decay * (self.running_mean - batch_mean.data)
            self.step += 1

        # perform normalization
        used_mean = used_mean.reshape(*self.shape).expand_as(x)

        y = x - used_mean

        if self.affine:
            bias = self.bias.reshape(*self.shape).expand_as(x)
            y = y + bias

        if logpx is None:
            return y
        else:
            return y, logpx

    def reverse(self, y, logpy=None):
        used_mean = self.running_mean

        if self.affine:
            bias = self.bias.reshape(*self.shape).expand_as(y)
            y = y - bias

        used_mean = used_mean.reshape(*self.shape).expand_as(y)
        x = y + used_mean

        if logpy is None:
            return x
        else:
            return x, logpy

    def __repr__(self):
        return (
            '{name}({num_features}, eps={eps}, decay={decay}, bn_lag={bn_lag},'
            ' affine={affine})'.format(name=self.__class__.__name__, **self.__dict__)
        )


class MovingBatchNorm1d(MovingBatchNormNd):

    @property
    def shape(self):
        return [1, -1]


class MovingBatchNorm2d(MovingBatchNormNd):

    @property
    def shape(self):
        return [1, -1, 1, 1]


class SqueezeLayer(nn.Module):

    def __init__(self, downscale_factor):
        super(SqueezeLayer, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x, logpx=None):
        squeeze_x = squeeze(x, self.downscale_factor)
        if logpx is None:
            return squeeze_x
        else:
            return squeeze_x, logpx

    def reverse(self, y, logpy=None):
        unsqueeze_y = unsqueeze(y, self.downscale_factor)
        if logpy is None:
            return unsqueeze_y
        else:
            return unsqueeze_y, logpy


def unsqueeze(input, upscale_factor=2):
    batch_size, in_channels, in_width = input.shape
    out_width = in_width * (upscale_factor)

    out_channels = in_channels // upscale_factor

    input_reshape = input.reshape(batch_size, out_channels, in_width, upscale_factor)

    output = input_reshape.permute(0, 1, 3, 2)
    return output.reshape(batch_size, out_channels, out_width)

def squeeze(input, downscale_factor=2):
    '''
    [:, C, W*r] -> [:, C*r, W]
    '''
    batch_size, in_channels, in_width = input.shape
    out_channels = in_channels * (downscale_factor)

    out_width = in_width // downscale_factor

    input_reshape = input.reshape(batch_size, in_channels, out_width, downscale_factor)

    output = input_reshape.permute(0, 1, 3, 2)
    return output.reshape(batch_size, out_channels, out_width)