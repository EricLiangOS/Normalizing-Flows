
import numpy as np
import torch
import os
import wandb
import matplotlib.pyplot as plt
import packaging.version
from models import flows, transforms, distributions
import models.nn.lipschitz
import models.nn.resnet
import torch.nn as nn
import math

if packaging.version.parse(torch.__version__) < packaging.version.parse('1.5.0'):
    raise RuntimeError('Torch versions lower than 1.5.0 not supported')


def initiate_wandb(configurations):
    
    os.environ["WANDB_NOTEBOOK_NAME"] = "normalizing_flows_benchmark"
    wandb.login()
    wandb.init( project="normalizing_flows_benchmark",
            entity="ericos",
            config=configurations,
               dir = "/data/submit/ericos",
    )
    wandb.config.update({"id": wandb.run.id})

def grab(var):
    return var.squeeze().detach().cpu().numpy()

def grab2d(var):
    n_rows, n_cols = len(var), len(var[0])
    result = np.empty((n_rows, n_cols), dtype=object)

    for i in range(n_rows):
        for j in range(n_cols):
            result[i, j] = var[i][j].squeeze().detach().cpu().numpy()

    return result

def compute_ess(logp, logq):
    logw = logp - logq
    log_ess = 2*torch.logsumexp(logw, dim=0) - torch.logsumexp(2*logw, dim=0)
    ess_per_cfg = torch.exp(log_ess) / len(logw)
    return ess_per_cfg

def compute_cae(mus, sigmas, ess, standard_ess):
    cae = 0
    for i in range(len(mus)):
        for j in range(len(sigmas)):
            cae += abs((standard_ess - ess[i][j]) / (1+ abs(math.log(sigmas[j])) + math.sqrt(abs(mus[i]))))
    
    return cae/(len(mus)*len(sigmas))

def compute_average(values):
    total = 0
    
    for row in values:
        for value in row:
            total += value
            
    return total/(len(values[0])*len(values))

def make_flow_list(flow_name, batch_size, dimensions):
    if flow_name == "box_muller_flow":
        return [{"name": "box_muller_flow"}]
    
    if flow_name == "affine_flow":
        return [{"name": "affine_flow"}]
    
    if flow_name == "trainable_affine_flow":
        return [{"name": "trainable_affine_flow"}]
    
    if flow_name == "affine_coupling_flow":
        return [{"name": "affine_coupling_flow", "n_layers": 32, "batch_size": batch_size, "mask_type": "checker", "hidden_sizes": [0]*32, "kernel_size": 1}]
   
    if flow_name == "masked_autoregressive_flow":
        return [{"name": "masked_autoregressive_flow", "n_layers": 32, "batch-size": batch_size, "hidden_sizes": [8]*16, "reverse_input":False},
        {"name": "masked_autoregressive_flow", "n_layers": 32, "batch-size": batch_size, "hidden_sizes": [8]*16, "reverse_input":True}]
    
    if flow_name == "neural_spline_flow_coupling":
        return [{"name": "neural_spline_flow", "tail": 50, "layers": 8, "splines": [{"name": "piecewise_rq_coupling"}]}]
    
    if flow_name == "neual_spline_flow_autoregressive":
        return [{"name": "neural_spline_flow", "tail": 50, "layers": 8, "splines": [{"name": "masked_piecewise_rq_autoregressive"}]}]

    if flow_name == "residual_flow":
        return [{"name": "residual_flow", "input_size": (batch_size, 1, dimensions), "intermediate_dim": 64, "fc": True, "fc_actnorm": True, "fc_idim":64, "fc_end": True}]

def plot_2d_distribution(samples, path=None, suptitle=None, plot_range =None):
   
    fig, ax = plt.subplots()
   
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
            
    if plot_range:
        ax.hist2d(samples[:,0], samples[:,1], bins=60, range=[plot_range, plot_range])
    else:
        ax.hist2d(samples[:,0], samples[:,1], bins=60)
        
    ax.set_xlabel(r"$X_1$")
    ax.set_ylabel(r"$X_2$", rotation=90)
    
    if path:
        plt.savefig(path)
    else:
        plt.show()
        
    plt.clf()
    
def plot_nd_distribution(samples, path=None, suptitle=None, plot_range=None, target_sample=None, average_dimensions = False):
    if average_dimensions:
        samples = samples.mean(axis=1)
        samples = samples.reshape(samples.shape + (1,))
    
    shape = samples.shape
    dimensions = shape[1]
    
    fig, ax = plt.subplots()
    
    if suptitle:
        fig.suptitle(suptitle)
    
    for dimension in range(dimensions):
        ax.hist(samples[:, dimension], 60, histtype='step', stacked=False, fill=False, label=f"D{dimension + 1}")
    
    if target_sample is not None:
        dimensions += 1
        ax.hist(target_sample, 60, histtype='step', stacked=False, fill=False, label=f"Target Sample")
        
    
    handles, labels = ax.get_legend_handles_labels()
    
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=dimensions)
    ax.set_xlabel("Sample value")
    ax.set_ylabel("Frequency", rotation=90)
    
    if path:
        plt.savefig(path)
    else:
        plt.show()
        
    plt.clf()

def plot_multiple_distribution(samples, plot_type, path=None, suptitle=None, plot_range =None, plot_labels = None):
    n_rows, n_cols = len(samples), len(samples[0])
    fig, axs = plt.subplots(n_rows, n_cols, dpi=125, figsize=(5.5*n_cols,5.5*n_rows))
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    
    for i in range(n_rows):
        for j in range(n_cols):
            if n_rows == 1 and n_cols == 1:
                ax = axs
            elif n_cols == 1:
                ax = axs[i]
            elif n_rows == 1:
                ax = axs[j]
            else:                
                ax = axs[i, j]
            
            label = None
            if plot_labels:
                label = plot_labels[i][j]
                ax.title.set_text(label)
            
            if plot_type == "2D":
                if plot_range:
                    ax.hist2d(samples[i, j][:,0], samples[i, j][:,1], bins=60, range=[plot_range, plot_range])
                else:
                    ax.hist2d(samples[i, j][:,0], samples[i, j][:,1], bins=60)
                ax.set_xlabel(r"$X_1$")
                ax.set_ylabel(r"$X_2$", rotation=90, y=.46) 
            else:
                ax.hist(samples[i, j], 60, histtype='step', stacked=False, fill=False)
                ax.set_xlabel(r"Value")
                ax.set_ylabel(r"Frequency", rotation=90, y=.46) 
    
    if path:
        plt.savefig(path)
    else:
        plt.show()
        
    plt.clf()        
                
                
def make_checker_mask(shape, parity):
    total_elements = torch.prod(torch.tensor(shape))
    
    checker = torch.ones(total_elements)
    checker[1::2] = 1 - parity
    checker[::2] = parity
    
    if not isinstance(shape, int): 
        checker = checker.view(*torch.tensor(shape))
    
    return checker


def make_column_mask(shape, parity):
    c1 = torch.ones(shape[0], dtype=torch.uint8) - parity
    c2 = torch.ones(shape[0], dtype=torch.uint8) - c1
    column = torch.stack((c1, c2), dim=1)
    
    return column

def make_random_mask(shape):
    rand_bool = torch.rand(shape) < 0.5
    rand = rand_bool.long()
    
    return rand, 1-rand

def make_conv_net(*, hidden_sizes, kernel_size, in_channels, out_channels, use_final_tanh):
    sizes = [in_channels] + hidden_sizes + [out_channels]
    
    assert packaging.version.parse(torch.__version__) >= packaging.version.parse('1.5.0')
    assert kernel_size % 2 == 1, 'kernel size must be odd for PyTorch >= 1.5.0'
    
    padding_size = (kernel_size // 2)
    net = []
    
    for i in range(len(sizes) - 1):
        net.append(torch.nn.Conv1d(
            sizes[i], sizes[i+1], kernel_size, padding=padding_size,
            stride=1, padding_mode='circular'))
    
        if i != len(sizes) - 2:
            net.append(torch.nn.LeakyReLU())
        else:
            if use_final_tanh:
                net.append(torch.nn.Tanh())
    
    return torch.nn.Sequential(*net)

def make_affine_coupling_flow_layers(*, dimensions, n_layers, mask_type, mask_shape, hidden_sizes, kernel_size, log_params):
    layers = []
    net = make_conv_net(
            in_channels=dimensions, out_channels=2, hidden_sizes=hidden_sizes,
            kernel_size=kernel_size, use_final_tanh=True)
    
    if mask_type == "checker":
        for i in range(n_layers):
            parity = i % 2
            
            mask = make_checker_mask(mask_shape, parity)
            coupling = flows.AffineCouplingFlow(log_params=log_params, net=net, mask=mask)
            layers.append(coupling)
    
    if mask_type == "column":
        for i in range(n_layers):
            parity = i % 2
            
            mask = make_column_mask(mask_shape, parity)
            coupling = flows.AffineCouplingFlow(log_params=log_params, net=net, mask=mask)
            layers.append(coupling)
    
    if mask_type == "random":
        for i in range(n_layers/2):
                
            mask1, mask2 = make_random_mask(mask_shape)
            coupling1 = flows.AffineCouplingFlow(log_params=log_params, net=net, mask=mask1)
            coupling2 = flows.AffineCouplingFlow(log_params=log_params, net=net, mask=mask2)
            layers.append(coupling1)
            layers.append(coupling2)
        
    
    return torch.nn.ModuleList(layers)

def make_masked_autoregressive_flow_layers(*, dimensions, n_layers, hidden_sizes, reverse_input, log_params):
    layers = []
    
    for i in range(n_layers):
        maf = flows.MaskedAutoregressiveFlow(dim=dimensions, hidden_sizes=hidden_sizes, reverse_input=reverse_input, log_params=log_params)
        layers.append(maf)
        
    return torch.nn.ModuleList(layers)

def make_base_transform(i, args):
    if args["base_transform_type"] == 'affine':
        return transforms.AffineCouplingTransform(
            mask= create_alternating_binary_mask(features=args["dimensions"], even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: models.nn.resnet.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args["hidden_features"],
                num_blocks=2,
                use_batch_norm=True
            )
        )
    elif args["base_transform_type"] == "piecewise rational quadratic coupling":
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask= create_alternating_binary_mask(features=args["dimensions"], even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: models.nn.resnet.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args["hidden_features"],
                num_blocks=2,
                use_batch_norm=True
            ),
            tails='linear',
            tail_bound=args["tail"],
            num_bins=args["num_bins"],
            apply_unconditional_transform=False
        )
    elif args["base_transform_type"] == "piecewise cubic coupling":
        return transforms.PiecewiseCubicCouplingTransform(
            mask= create_alternating_binary_mask(features=args["dimensions"], even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: models.nn.resnet.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args["hidden_features"],
                num_blocks=2,
                use_batch_norm=True
            ),
            tails='linear',
            tail_bound=args["tail"],
            num_bins=args["num_bins"],
            apply_unconditional_transform=False
        )
        
    elif args["base_transform_type"] == "masked piecewise rational quadratic autoregressive":
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=args["dimensions"],
            hidden_features=args["hidden_features"],
            tails='linear',
            tail_bound=args["tail"],
            num_bins=args["num_bins"],
            use_batch_norm=True
            )
        
    elif args["base_transform_type"] == "masked piecewise cubic autoregressive":
        return transforms.MaskedPiecewiseCubicAutoregressiveTransform(
            features=args["dimensions"],
            hidden_features=args["hidden_features"],
            num_bins=args["num_bins"],
            use_batch_norm=True
        )

def make_neural_spline_flow(splines, layers, prior_distribution, dimensions, tail):
    layer_set = set()
    layer_names = ""
    for spline in splines:
        layer_set.add(spline["name"])

    s = list(layer_set)
    s.sort()    
    layer_names = " + ".join(s)

    inner_transform = []

    for spline in splines:
        if spline["name"] == "affine_coupling_transform":
            inner_transform += [make_base_transform(i, {"base_transform_type": "affine", "dimensions": dimensions, "num_bins": 32, "hidden_features": 32}) for i in range(layers)] 
        if spline["name"] == "piecewise_rq_coupling":
            inner_transform += [make_base_transform(i, {"base_transform_type": "piecewise rational quadratic coupling", "dimensions": dimensions, "num_bins": 32, "hidden_features": 32, "tail": tail}) for i in range(layers)] 
        if spline["name"] == "piecewise_cubic_coupling":
            inner_transform += [make_base_transform(i, {"base_transform_type": "piecewise cubic coupling", "dimensions": dimensions, "num_bins": 32, "hidden_features": 32, "tail": tail}) for i in range(layers)]
        if spline["name"] == "masked_piecewise_rq_autoregressive":
            inner_transform += [make_base_transform(i, {"base_transform_type": "masked piecewise rational quadratic autoregressive", "dimensions": dimensions, "hidden_features": 32, "num_bins": 32, "tail": tail}) for i in range(layers)]
        if spline["name"] == "masked_piecewise_cubic_autoregressive":
            inner_transform += [make_base_transform(i, {"base_transform_type": "masked piecewise cubic autoregressive", "dimensions": dimensions, "num_bins": 32, "hidden_features": 32, "tail": tail}) for i in range(layers)]
            
            
    transform = transforms.CompositeTransform(inner_transform)
    
    return distributions.BaseNeuralSplineFlow(transform, prior_distribution)

def make_simple_residual_flow(hidden_units, hidden_layers, latent_size, reduce_memory, exact_trace, n_dist, brute_force):
    layer = [latent_size] + [hidden_units] * hidden_layers + [latent_size]
    net = models.nn.lipschitz.LipschitzMLP(layer, init_zeros=exact_trace, lipschitz_const=0.9)
    flow = flows.Residual(net, reduce_memory=reduce_memory, n_dist=n_dist, exact_trace=exact_trace, brute_force=brute_force)
    
    return flow

def make_residual_flow(dimensions):
    dims = [dimensions] + list(map(int, ))
    blocks = []
    

def tile(x, n):
    if not is_positive_int(n):
        raise TypeError('Argument \'n\' must be a positive integer.')
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if not is_nonnegative_int(num_batch_dims):
        raise TypeError('Number of batch dimensions must be a non-negative integer.')
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


def merge_leading_dims(x, num_dims):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""
    if not is_positive_int(num_dims):
        raise TypeError('Number of leading dims must be a positive integer.')
    if num_dims > x.dim():
        raise ValueError('Number of leading dims can\'t be greater than total number of dims.')
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    if not is_positive_int(num_reps):
        raise TypeError('Number of repetitions must be a positive integer.')
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)


def tensor2numpy(x):
    return x.detach().cpu().numpy()


def logabsdet(x):
    """Returns the log absolute determinant of square matrix x."""
    # Note: torch.logdet() only works for positive determinant.
    _, res = torch.slogdet(x)
    return res


def random_orthogonal(size):
    """
    Returns a random orthogonal matrix as a 2-dim tensor of shape [size, size].
    """

    # Use the QR decomposition of a random Gaussian matrix.
    x = torch.randn(size, size)
    q, _ = torch.qr(x)
    return q


def get_num_parameters(model):
    """
    Returns the number of trainable parameters in a model of type nn.Module
    :param model: nn.Module containing trainable parameters
    :return: number of trainable parameters in model
    """
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters


def create_alternating_binary_mask(features, even=True):
    """
    Creates a binary mask of a given dimension which alternates its masking.

    :param features: Dimension of mask.
    :param even: If True, even values are assigned 1s, odd 0s. If False, vice versa.
    :return: Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


def create_mid_split_binary_mask(features):
    """
    Creates a binary mask of a given dimension which splits its masking at the midpoint.

    :param features: Dimension of mask.
    :return: Binary mask split at midpoint of type torch.Tensor
    """
    mask = torch.zeros(features).byte()
    midpoint = features // 2 if features % 2 == 0 else features // 2 + 1
    mask[:midpoint] += 1
    return mask


def create_random_binary_mask(features):
    """
    Creates a random binary mask of a given dimension with half of its entries
    randomly set to 1s.

    :param features: Dimension of mask.
    :return: Binary mask with half of its entries set to 1s, of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    weights = torch.ones(features).float()
    num_samples = features // 2 if features % 2 == 0 else features // 2 + 1
    indices = torch.multinomial(
        input=weights,
        num_samples=num_samples,
        replacement=False
    )
    mask[indices] += 1
    return mask

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1

def cbrt(x):
    """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)


def get_temperature(max_value, bound=1-1e-3):
    """
    For a dataset with max value 'max_value', returns the temperature such that

        sigmoid(temperature * max_value) = bound.

    If temperature is greater than 1, returns 1.

    :param max_value:
    :param bound:
    :return:
    """
    max_value = torch.Tensor([max_value])
    bound = torch.Tensor([bound])
    temperature = min(- (1 / max_value) * (torch.log1p(-bound) - torch.log(bound)), 1)
    return temperature

"""Functions that check types."""


def is_bool(x):
    return isinstance(x, bool)


def is_int(x):
    return isinstance(x, int)

def is_positive_int(x):
    return is_int(x) and x > 0

def is_nonnegative_int(x):
    return is_int(x) and x >= 0

def is_power_of_two(n):
    if is_positive_int(n):
        return not n & (n - 1)
    else:
        return False
    
def uniform_transform(sample, low, high):
    return (high - low)*sample + low

def batch_jacobian(g, x):
    jac = []
    for d in range(g.shape[1]):
        jac.append(
            torch.autograd.grad(torch.sum(g[:, d]), x, create_graph=True)[0].view(
                x.shape[0], 1, x.shape[1]
            )
        )
    return torch.cat(jac, 1)


def batch_trace(M):
    return M.view(M.shape[0], -1)[:, :: M.shape[1] + 1].sum(1)


# Logdet Estimators


class MemoryEfficientLogDetEstimator(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        estimator_fn,
        gnet,
        x,
        n_power_series,
        vareps,
        coeff_fn,
        training,
        *g_params
    ):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = gnet(x)
            ctx.g = g
            ctx.x = x
            logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, training)

            if training:
                grad_x, *grad_params = torch.autograd.grad(
                    logdetgrad.sum(),
                    (x,) + g_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        return safe_detach(g), safe_detach(logdetgrad)

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError("Provide training=True if using backward.")

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x

            # Precomputed gradients.
            g_params = params_and_grad[: len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2 :]

            dg_x, *dg_params = torch.autograd.grad(
                g, [x] + g_params, grad_g, allow_unused=True
            )

        # Update based on gradient from logdetgrad.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple(
                [g.mul_(dL) if g is not None else None for g in grad_params]
            )

        # Update based on gradient from g.
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple(
                [
                    dg.add_(djac) if djac is not None else dg
                    for dg, djac in zip(dg_params, grad_params)
                ]
            )

        return (None, None, grad_x, None, None, None, None) + grad_params


def basic_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    logdetgrad = torch.tensor(0.0).to(x)
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[
            0
        ]
        tr = torch.sum(vjp.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
        delta = (-1) ** (k + 1) / k * coeff_fn(k) * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad


def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1) ** k * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(
        vjp_jac.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1
    )
    return logdetgrad


def mem_eff_wrapper(estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(gnet, nn.Module):
        raise ValueError("g is required to be an instance of nn.Module.")

    return MemoryEfficientLogDetEstimator.apply(
        estimator_fn,
        gnet,
        x,
        n_power_series,
        vareps,
        coeff_fn,
        training,
        *list(gnet.parameters())
    )


# Helper distribution functions


def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)


def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.0
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p) ** max(k - 1, 0)


def poisson_sample(lamb, n_samples):
    return np.random.poisson(lamb, n_samples)


def poisson_1mcdf(lamb, k, offset):
    if k <= offset:
        return 1.0
    else:
        k = k - offset
    """P(n >= k)"""
    sum = 1.0
    for i in range(1, k):
        sum += lamb**i / math.factorial(i)
    return 1 - np.exp(-lamb) * sum


# Helper functions


def safe_detach(tensor):
    return tensor.detach().requires_grad_(tensor.requires_grad)