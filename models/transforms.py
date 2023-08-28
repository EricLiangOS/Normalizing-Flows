import warnings


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models import transforms, splines
from models.splines import rational_quadratic, cubic
from models import made as made_module
import helper_functions


class CompositeTransform(nn.Module):

    def __init__(self, transforms):
        super().__init__()
        self._transforms = nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, funcs, context):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = torch.zeros(batch_size)
        for func in funcs:
            outputs, logabsdet = func(outputs, context)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs, context=None):
        funcs = self._transforms
        return self._cascade(inputs, funcs, context)

    def inverse(self, inputs, context=None):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context)

class CouplingTransform(nn.Module):

    def __init__(self,
                 mask,
                 transform_net_create_fn,
                 unconditional_transform=None):
        
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError('Mask must be a 1-dim tensor.')
        if mask.numel() <= 0:
            raise ValueError('Mask can\'t be empty.')

        super().__init__()
        self.features = len(mask)
        features_vector = torch.arange(self.features)

        self.register_buffer('identity_features', features_vector.masked_select(mask <= 0))
        self.register_buffer('transform_features', features_vector.masked_select(mask > 0))

        assert self.num_identity_features + self.num_transform_features == self.features

        self.transform_net = transform_net_create_fn(
            self.num_identity_features,
            self.num_transform_features * self._transform_dim_multiplier()
        )

        if unconditional_transform is None:
            self.unconditional_transform = None
        else:
            self.unconditional_transform = unconditional_transform(
                features=self.num_identity_features
            )

    @property
    def num_identity_features(self):
        return len(self.identity_features)

    @property
    def num_transform_features(self):
        return len(self.transform_features)

    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Inputs must be a 2D or a 4D tensor.')

        if inputs.shape[1] != self.features:
            raise ValueError('Expected features = {}, got {}.'.format(
                self.features, inputs.shape[1]))

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet = self._coupling_transform_forward(
            inputs=transform_split,
            transform_params=transform_params
        )

        if self.unconditional_transform is not None:
            identity_split, logabsdet_identity =\
                self.unconditional_transform(identity_split, context)
            logabsdet += logabsdet_identity

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features, ...] = identity_split
        outputs[:, self.transform_features, ...] = transform_split

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Inputs must be a 2D or a 4D tensor.')

        if inputs.shape[1] != self.features:
            raise ValueError('Expected features = {}, got {}.'.format(
                self.features, inputs.shape[1]))

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        logabsdet = 0.0
        if self.unconditional_transform is not None:
            identity_split, logabsdet = self.unconditional_transform.inverse(identity_split,
                                                                             context)

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet_split = self._coupling_transform_inverse(
            inputs=transform_split,
            transform_params=transform_params
        )
        logabsdet += logabsdet_split

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features] = identity_split
        outputs[:, self.transform_features] = transform_split

        return outputs, logabsdet

    def _transform_dim_multiplier(self):
        """Number of features to output for each transform dimension."""
        raise NotImplementedError()

    def _coupling_transform_forward(self, inputs, transform_params):
        """Forward pass of the coupling transform."""
        raise NotImplementedError()

    def _coupling_transform_inverse(self, inputs, transform_params):
        """Inverse of the coupling transform."""
        raise NotImplementedError()
    
class PiecewiseCouplingTransform(CouplingTransform):
    def _coupling_transform_forward(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=False)

    def _coupling_transform_inverse(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=True)

    def _coupling_transform(self, inputs, transform_params, inverse=False):
        if inputs.dim() == 4:
            b, c, h, w = inputs.shape
            # For images, reshape transform_params from Bx(C*?)xHxW to BxCxHxWx?
            transform_params = transform_params.reshape(b, c, -1, h, w).permute(0, 1, 3, 4, 2)
        elif inputs.dim() == 2:
            b, d = inputs.shape
            # For 2D data, reshape transform_params from Bx(D*?) to BxDx?
            transform_params = transform_params.reshape(b, d, -1)

        outputs, logabsdet = self._piecewise_cdf(inputs, transform_params, inverse)

        return outputs, helper_functions.sum_except_batch(logabsdet)

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        raise NotImplementedError()

class PiecewiseRationalQuadraticCouplingTransform(PiecewiseCouplingTransform):
    def __init__(self, mask, transform_net_create_fn,
                 num_bins=10,
                 tails=None,
                 tail_bound=1.,
                 apply_unconditional_transform=False,
                 img_shape=None,
                 min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE):
        
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        if apply_unconditional_transform:
            unconditional_transform = lambda features: PiecewiseRationalQuadraticCDF(
                shape=[features] + (img_shape if img_shape else []),
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative
            )
        else:
            unconditional_transform = None
        super().__init__(mask, transform_net_create_fn,
                         unconditional_transform=unconditional_transform)

    def _transform_dim_multiplier(self):
        if self.tails == 'linear':
            return self.num_bins * 3 - 1
        else:
            return self.num_bins * 3 + 1

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., :self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins:]

        if hasattr(self.transform_net, 'hidden_features'):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, 'hidden_channels'):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn('Inputs to the softmax are not scaled down: initialization might be bad.')


        if self.tails is None:
            spline_fn = splines.rational_quadratic.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.rational_quadratic.unconstrained_rational_quadratic_spline
            spline_kwargs = {
                'tails': self.tails,
                'tail_bound': self.tail_bound
            }

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )
        
class PiecewiseRationalQuadraticCDF(nn.Module):
    def __init__(self,
                 shape,
                 num_bins=10,
                 tails=None,
                 tail_bound=1.,
                 identity_init=False,
                 min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE):
        super().__init__()

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        self.tail_bound = tail_bound
        self.tails = tails

        if identity_init:
            self.unnormalized_widths = nn.Parameter(torch.zeros(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.zeros(*shape, num_bins))

            constant = np.log(np.exp(1 - min_derivative) - 1)
            num_derivatives = (num_bins - 1) if self.tails == 'linear' else (num_bins + 1)
            self.unnormalized_derivatives = nn.Parameter(constant * torch.ones(*shape,
                                                                               num_derivatives))
        else:
            self.unnormalized_widths = nn.Parameter(torch.rand(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.rand(*shape, num_bins))

            num_derivatives = (num_bins - 1) if self.tails == 'linear' else (num_bins + 1)
            self.unnormalized_derivatives = nn.Parameter(torch.rand(*shape, num_derivatives))

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths=_share_across_batch(self.unnormalized_widths, batch_size)
        unnormalized_heights=_share_across_batch(self.unnormalized_heights, batch_size)
        unnormalized_derivatives=_share_across_batch(self.unnormalized_derivatives, batch_size)

        if self.tails is None:
            spline_fn = splines.rational_quadratic.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.rational_quadratic.unconstrained_rational_quadratic_spline
            spline_kwargs = {
                'tails': self.tails,
                'tail_bound': self.tail_bound
            }

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, helper_functions.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)
    
def _share_across_batch(params, batch_size):
    return params[None,...].expand(batch_size, *params.shape)

class AffineCouplingTransform(CouplingTransform):

    def _transform_dim_multiplier(self):
        return 2

    def _scale_and_shift(self, transform_params):
        unconstrained_scale = transform_params[:, self.num_transform_features:, ...]
        shift = transform_params[:, :self.num_transform_features, ...]
        # scale = (F.softplus(unconstrained_scale) + 1e-3).clamp(0, 3)
        scale = torch.sigmoid(unconstrained_scale + 2) + 1e-3
        return scale, shift

    def _coupling_transform_forward(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        outputs = inputs * scale + shift
        logabsdet = helper_functions.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _coupling_transform_inverse(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -helper_functions.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet
    
class PiecewiseCubicCouplingTransform(PiecewiseCouplingTransform):
    def __init__(self,
                 mask,
                 transform_net_create_fn,
                 num_bins=10,
                 tails=None,
                 tail_bound=1.,
                 apply_unconditional_transform=False,
                 img_shape=None,
                 min_bin_width=splines.cubic.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.cubic.DEFAULT_MIN_BIN_HEIGHT):

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.tails = tails
        self.tail_bound = tail_bound

        if apply_unconditional_transform:
            unconditional_transform = lambda features: PiecewiseCubicCDF(
                shape=[features] + (img_shape if img_shape else []),
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height
            )
        else:
            unconditional_transform = None

        super().__init__(mask, transform_net_create_fn,
                         unconditional_transform=unconditional_transform)

    def _transform_dim_multiplier(self):
        return self.num_bins * 2 + 2

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., :self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins:2*self.num_bins]
        unnorm_derivatives_left = transform_params[..., 2*self.num_bins][..., None]
        unnorm_derivatives_right = transform_params[..., 2*self.num_bins + 1][..., None]

        if hasattr(self.transform_net, 'hidden_features'):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)

        if self.tails is None:
            spline_fn = splines.cubic.cubic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.cubic.unconstrained_cubic_spline
            spline_kwargs = {
                'tails': self.tails,
                'tail_bound': self.tail_bound
            }

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnorm_derivatives_left=unnorm_derivatives_left,
            unnorm_derivatives_right=unnorm_derivatives_right,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            **spline_kwargs
        )

class PiecewiseCubicCDF(nn.Module):
    def __init__(self,
                 shape,
                 num_bins=10,
                 tails=None,
                 tail_bound=1.,
                 min_bin_width=splines.cubic.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.cubic.DEFAULT_MIN_BIN_HEIGHT):
        super().__init__()

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.tail_bound = tail_bound
        self.tails = tails

        self.unnormalized_widths = nn.Parameter(torch.randn(*shape, num_bins))
        self.unnormalized_heights = nn.Parameter(torch.randn(*shape, num_bins))
        self.unnorm_derivatives_left = nn.Parameter(torch.randn(*shape, 1))
        self.unnorm_derivatives_right = nn.Parameter(torch.randn(*shape, 1))

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths = _share_across_batch(self.unnormalized_widths, batch_size)
        unnormalized_heights = _share_across_batch(self.unnormalized_heights, batch_size)
        unnorm_derivatives_left = _share_across_batch(self.unnorm_derivatives_left, batch_size)
        unnorm_derivatives_right = _share_across_batch(self.unnorm_derivatives_right, batch_size)

        if self.tails is None:
            spline_fn = splines.cubic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.cubic.unconstrained_cubic_spline
            spline_kwargs = {
                'tails': self.tails,
                'tail_bound': self.tail_bound
            }

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnorm_derivatives_left=unnorm_derivatives_left,
            unnorm_derivatives_right=unnorm_derivatives_right,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            **spline_kwargs
        )

        return outputs, helper_functions.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)

class AutoregressiveTransform(nn.Module):
    """Transforms each input variable with an invertible elementwise transformation.

    The parameters of each invertible elementwise transformation can be functions of previous input
    variables, but they must not depend on the current or any following input variables.

    NOTE: Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.
    """
    def __init__(self, autoregressive_net):
        super(AutoregressiveTransform, self).__init__()
        self.autoregressive_net = autoregressive_net

    def forward(self, inputs, context=None):
        autoregressive_params = self.autoregressive_net(inputs, context)
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        num_inputs = np.prod(inputs.shape[1:])
        outputs = torch.zeros_like(inputs)
        logabsdet = None
        for _ in range(num_inputs):
            autoregressive_params = self.autoregressive_net(outputs, context)
            outputs, logabsdet = self._elementwise_inverse(inputs, autoregressive_params)
        return outputs, logabsdet

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, inputs, autoregressive_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, inputs, autoregressive_params):
        raise NotImplementedError()

class MaskedAffineAutoregressiveTransform(AutoregressiveTransform):
    def __init__(self,
                 features,
                 hidden_features,
                 context_features=None,
                 num_blocks=2,
                 use_residual_blocks=True,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        self.features = features
        made = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        super(MaskedAffineAutoregressiveTransform, self).__init__(made)

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(autoregressive_params)
        scale = torch.sigmoid(unconstrained_scale + 2.) + 1e-3
        log_scale = torch.log(scale)
        outputs = scale * inputs + shift
        logabsdet = helper_functions.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(autoregressive_params)
        scale = torch.sigmoid(unconstrained_scale + 2.) + 1e-3
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -helper_functions.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _unconstrained_scale_and_shift(self, autoregressive_params):
        # split_idx = autoregressive_params.size(1) // 2
        # unconstrained_scale = autoregressive_params[..., :split_idx]
        # shift = autoregressive_params[..., split_idx:]
        # return unconstrained_scale, shift
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        unconstrained_scale = autoregressive_params[..., 0]
        shift = autoregressive_params[..., 1]
        return unconstrained_scale, shift

class MaskedPiecewiseLinearAutoregressiveTransform(AutoregressiveTransform):
    def __init__(self,
                 num_bins,
                 features,
                 hidden_features,
                 context_features=None,
                 num_blocks=2,
                 use_residual_blocks=True,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        self.num_bins = num_bins
        self.features = features
        made = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        super().__init__(made)

    def _output_dim_multiplier(self):
        return self.num_bins

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_pdf = autoregressive_params.view(batch_size,
                                                      self.features,
                                                      self._output_dim_multiplier())

        outputs, logabsdet = splines.linear_spline(inputs=inputs,
                                                   unnormalized_pdf=unnormalized_pdf,
                                                   inverse=inverse)

        return outputs, helper_functions.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)

class MaskedPiecewiseQuadraticAutoregressiveTransform(AutoregressiveTransform):
    def __init__(self,
                 features,
                 hidden_features,
                 context_features=None,
                 num_bins=10,
                 num_blocks=2,
                 tails=None,
                 tail_bound=1.,
                 use_residual_blocks=True,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE
                 ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound
        self.features = features
        made = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        super().__init__(made)

    def _output_dim_multiplier(self):
        if self.tails == 'linear':
            return self.num_bins * 2 - 1
        else:
            return self.num_bins * 2 + 1

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size = inputs.shape[0]

        transform_params = autoregressive_params.view(batch_size,
                                                      self.features,
                                                      self._output_dim_multiplier())

        unnormalized_widths = transform_params[..., :self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins:]

        if hasattr(self.autoregressive_net, 'hidden_features'):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = splines.quadratic_spline
            spline_kwargs = {}
        elif self.tails == 'linear':
            spline_fn = splines.rational_quadratic.unconstrained_quadratic_spline
            spline_kwargs = {
                'tails': self.tails,
                'tail_bound': self.tail_bound
            }
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_heights=unnormalized_heights,
            unnormalized_widths=unnormalized_widths,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            **spline_kwargs
        )

        return outputs, helper_functions.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)

class MaskedPiecewiseCubicAutoregressiveTransform(AutoregressiveTransform):
    def __init__(self,
                 num_bins,
                 features,
                 hidden_features,
                 context_features=None,
                 num_blocks=2,
                 use_residual_blocks=True,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        self.num_bins = num_bins
        self.features = features
        made = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        super(MaskedPiecewiseCubicAutoregressiveTransform, self).__init__(made)

    def _output_dim_multiplier(self):
        return self.num_bins * 2 + 2

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size = inputs.shape[0]

        transform_params = autoregressive_params.view(batch_size,
                                                 self.features,
                                                 self.num_bins * 2 + 2)

        unnormalized_widths = transform_params[...,:self.num_bins]
        unnormalized_heights = transform_params[...,self.num_bins:2*self.num_bins]
        derivatives = transform_params[...,2*self.num_bins:]
        unnorm_derivatives_left = derivatives[..., 0][..., None]
        unnorm_derivatives_right = derivatives[..., 1][..., None]

        if hasattr(self.autoregressive_net, 'hidden_features'):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        outputs, logabsdet = splines.cubic.cubic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnorm_derivatives_left=unnorm_derivatives_left,
            unnorm_derivatives_right=unnorm_derivatives_right,
            inverse=inverse
        )
        return outputs, helper_functions.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)

class MaskedPiecewiseRationalQuadraticAutoregressiveTransform(AutoregressiveTransform):
    def __init__(self,
                 features,
                 hidden_features,
                 context_features=None,
                 num_bins=10,
                 tails=None,
                 tail_bound=1.,
                 num_blocks=2,
                 use_residual_blocks=True,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE
                 ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        autoregressive_net = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        super().__init__(autoregressive_net)

    def _output_dim_multiplier(self):
        if self.tails == 'linear':
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(
            batch_size,
            features,
            self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[...,:self.num_bins]
        unnormalized_heights = transform_params[...,self.num_bins:2*self.num_bins]
        unnormalized_derivatives = transform_params[...,2*self.num_bins:]

        if hasattr(self.autoregressive_net, 'hidden_features'):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == 'linear':
            spline_fn = splines.rational_quadratic.unconstrained_rational_quadratic_spline
            spline_kwargs = {
                'tails': self.tails,
                'tail_bound': self.tail_bound
            }
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, helper_functions.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)


def main():
    inputs = torch.randn(16, 10)
    context = torch.randn(16, 24)
    transform = MaskedPiecewiseQuadraticAutoregressiveTransform(
        features=10,
        hidden_features=32,
        context_features=24,
        num_bins=10,
        tails='linear',
        num_blocks=2
    )
    outputs, logabsdet = transform(inputs, context)
    print(outputs.shape)

