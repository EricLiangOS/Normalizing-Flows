from typing import List, Optional
import numpy as np
from numpy.random import permutation, randint
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F, init
import helper_functions


class SimpleMaskedLinear(nn.Linear):

    def __init__(self, n_in, n_out, bias = True):
        super().__init__(n_in, n_out, bias)
        self.mask = None

    def initialise_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        return F.linear(x.clone(), self.mask * self.weight, self.bias)

class SimpleMADE(nn.Module):
    def __init__(self, n_in, hidden_dims, gaussian = False, random_order = False, seed = None):

        super().__init__()
        
        np.random.seed(seed)
        self.n_in = n_in
        self.n_out = 2 * n_in if gaussian else n_in
        self.hidden_dims = hidden_dims
        self.random_order = random_order
        self.gaussian = gaussian
        self.masks = {}
        self.mask_matrix = []
        self.layers = []

        dim_list = [self.n_in, *hidden_dims, self.n_out]
        
        for i in range(len(dim_list) - 2):
            self.layers.append(SimpleMaskedLinear(dim_list[i], dim_list[i + 1]),)
            self.layers.append(torch.nn.ReLU())
        
        self.layers.append(SimpleMaskedLinear(dim_list[-2], dim_list[-1]))
        
        self.model = nn.Sequential(*self.layers)
        
        self._create_masks()

    def forward(self, x):

        if self.gaussian:
            return self.model(x)
        else:
            return torch.sigmoid(self.model(x))

    def _create_masks(self):

        L = len(self.hidden_dims)
        D = self.n_in

        self.masks[0] = permutation(D) if self.random_order else np.arange(D)

        for l in range(L):
            low = self.masks[l].min()
            size = self.hidden_dims[l]
            self.masks[l + 1] = randint(low=low, high=D - 1, size=size)

        self.masks[L + 1] = self.masks[0]

        for i in range(len(self.masks) - 1):
            m = self.masks[i]
            m_next = self.masks[i + 1]
            
            M = torch.zeros(len(m_next), len(m))
            for j in range(len(m_next)):
                
                M[j, :] = torch.from_numpy((m_next[j] >= m).astype(int))
            
            self.mask_matrix.append(M)

        if self.gaussian:
            m = self.mask_matrix.pop(-1)
            self.mask_matrix.append(torch.cat((m, m), dim=0))

        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, SimpleMaskedLinear):
                module.initialise_mask(next(mask_iter))

def _get_input_degrees(in_features):
    """Returns the degrees an input to MADE should have."""
    return torch.arange(1, in_features + 1)


class MaskedLinear(nn.Linear):

    def __init__(self,
                 in_degrees,
                 out_features,
                 autoregressive_features,
                 random_mask,
                 is_output,
                 bias=True):
        super().__init__(
            in_features=len(in_degrees),
            out_features=out_features,
            bias=bias)
        mask, degrees = self._get_mask_and_degrees(
            in_degrees=in_degrees,
            out_features=out_features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=is_output)
        
        self.register_buffer('mask', mask)
        self.register_buffer('degrees', degrees)

    @classmethod
    def _get_mask_and_degrees(cls,
                              in_degrees,
                              out_features,
                              autoregressive_features,
                              random_mask,
                              is_output):
        if is_output:
            out_degrees = helper_functions.tile(
                _get_input_degrees(autoregressive_features),
                out_features // autoregressive_features
            )
            mask = (out_degrees[..., None] > in_degrees).float()

        else:
            if random_mask:
                min_in_degree = torch.min(in_degrees).item()
                min_in_degree = min(min_in_degree, autoregressive_features - 1)
                out_degrees = torch.randint(
                    low=min_in_degree,
                    high=autoregressive_features,
                    size=[out_features],
                    dtype=torch.long)
            else:
                max_ = max(1, autoregressive_features - 1)
                min_ = min(1, autoregressive_features - 1)
                out_degrees = torch.arange(out_features) % max_ + min_
            mask = (out_degrees[..., None] >= in_degrees).float()

        return mask, out_degrees

    def forward(self, x):

        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedFeedforwardBlock(nn.Module):
    """A feedforward block based on a masked linear module.

    NOTE: In this implementation, the number of output features is taken to be equal to
    the number of input features.
    """

    def __init__(self,
                 in_degrees,
                 autoregressive_features,
                 context_features=None,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        super().__init__()
        features = len(in_degrees)

        # Batch norm.
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features, eps=1e-3)
        else:
            self.batch_norm = None

        if context_features is not None:
            raise NotImplementedError()

        # Masked linear.
        self.linear = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=False,
        )
        self.degrees = self.linear.degrees

        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs, context=None):
        if context is not None:
            raise NotImplementedError()

        if self.batch_norm:
            outputs = self.batch_norm(inputs)
        else:
            outputs = inputs
        outputs = self.linear(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class MaskedResidualBlock(nn.Module):
    """A residual block containing masked linear modules."""

    def __init__(self,
                 in_degrees,
                 autoregressive_features,
                 context_features=None,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 zero_initialization=True):
        if random_mask:
            raise ValueError('Masked residual block can\'t be used with random masks.')
        super().__init__()
        features = len(in_degrees)

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)

        # Batch norm.
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(features, eps=1e-3)
                for _ in range(2)
            ])

        # Masked linear.
        linear_0 = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        linear_1 = MaskedLinear(
            in_degrees=linear_0.degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        self.linear_layers = nn.ModuleList([linear_0, linear_1])
        self.degrees = linear_1.degrees
        if torch.all(self.degrees >= in_degrees).item() != 1:
            raise RuntimeError('In a masked residual block, the output degrees can\'t be'
                               ' less than the corresponding input degrees.')

        # Activation and dropout
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

        # Initialization.
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, a=-1e-3, b=1e-3)
            init.uniform_(self.linear_layers[-1].bias, a=-1e-3, b=1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(
                torch.cat((temps, self.context_layer(context)), dim=1),
                dim=1
            )
        return inputs + temps


class MADE(nn.Module):
    """Implementation of MADE.

    It can use either feedforward blocks or residual blocks (default is residual).
    Optionally, it can use batch norm or dropout within blocks (default is no).
    """

    def __init__(self,
                 features,
                 hidden_features,
                 context_features=None,
                 num_blocks=2,
                 output_multiplier=1,
                 use_residual_blocks=True,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        if use_residual_blocks and random_mask:
            raise ValueError('Residual blocks can\'t be used with random masks.')
        super().__init__()

        # Initial layer.
        self.initial_layer = MaskedLinear(
            in_degrees=_get_input_degrees(features),
            out_features=hidden_features,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=False
        )

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, hidden_features)

        # Residual blocks.
        blocks = []
        if use_residual_blocks:
            block_constructor = MaskedResidualBlock
        else:
            block_constructor = MaskedFeedforwardBlock
        prev_out_degrees = self.initial_layer.degrees
        for _ in range(num_blocks):
            blocks.append(
                block_constructor(
                    in_degrees=prev_out_degrees,
                    autoregressive_features=features,
                    context_features=context_features,
                    random_mask=random_mask,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)

        # Final layer.
        self.final_layer = MaskedLinear(
            in_degrees=prev_out_degrees,
            out_features=features * output_multiplier,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=True
        )

    def forward(self, inputs, context=None):
        outputs = self.initial_layer(inputs)
        if context is not None:
            outputs += self.context_layer(context)
        for block in self.blocks:
            outputs = block(outputs, context)
        outputs = self.final_layer(outputs)
        return outputs