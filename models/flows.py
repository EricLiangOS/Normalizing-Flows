
import torch
import torch.nn as nn
import numpy as np
import helper_functions
from models import residual_layers
from models.made import SimpleMADE
import models.nn.activations
import models.residual_layers
import wandb

ACT_FNS = {
    'softplus': lambda b: nn.Softplus(),
    'elu': lambda b: nn.ELU(inplace=b),
    'swish': lambda b: models.nn.activations.Swish(),
    'lcube': lambda b: models.nn.activations.LipschitzCube(),
    'identity': lambda b: models.nn.activations.Identity(),
    'relu': lambda b: nn.ReLU(inplace=b),
}

class BoxMullerFlow(nn.Module):
    def __init__(self, log_params = {"log": False}):
        super().__init__()
        
        self.log_params = log_params
    
    def forward(self, z):
        u1 = z[:, 0]
        u2 = z[:, 1]
        
        r = torch.sqrt(-2 * torch.log(u1))
        theta = 2 * torch.pi * u2
        
        z1 = r * torch.cos(theta)
        z2 = r * torch.sin(theta)
        
        z = torch.stack([z1, z2], axis=1)
        
        logJ = torch.log(2*torch.pi/u1)
        
        return z, logJ
    
    def reverse(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        
        r = torch.sqrt(x1**2 + x2**2)
        theta = torch.atan2(x2, x1)
        
        u1 = torch.exp(-0.5 * r**2)
        u2 = theta / (2 * torch.pi)
        u2 %= 1
        
        x = torch.stack([u1, u2], axis=1)
        
        logJ = torch.log(2*torch.pi/u1) 
        
        return x, logJ
      
class AffineFlow(nn.Module):
    def __init__(self, mu, sigma, log_params = {"log": False}):
        super().__init__()
        
        self.log_params = log_params
        self.log_sigma = torch.log(sigma*torch.ones(1))
        self.mu = mu*torch.ones(1)
        
    def forward(self, z):
        
        mu = self.mu.squeeze(-1)
        log_sigma = self.log_sigma.squeeze(-1)
        
        x =  mu + torch.exp(log_sigma)*z
        logJ = log_sigma
        
        return x, logJ
    
    def reverse(self, x):

        mu = self.mu.squeeze(-1)
        log_sigma = self.log_sigma.squeeze(-1)
        
        z =  (x - mu) * torch.exp(-log_sigma)
        logJ = -log_sigma
        
        return z, logJ
      
class TrainableAffineFlow(nn.Module):
    def __init__(self, dim=2, log_params = {"log": False}):
        super().__init__()
        
        self.log_params = log_params
        self.p1 = nn.Parameter(torch.zeros((dim)))
        self.p1.requires_grad = True
        self.p2 = nn.Parameter(torch.ones((dim)))
        self.p2.requires_grad = True

    def forward(self, z):
        
        p1 = self.p1.squeeze(-1)
        p2 = self.p2.squeeze(-1)
        
        x = p1 + torch.exp(p2) * z
        logJ = torch.sum(p2)
        
        if self.log_params["log"]:
            wandb.log({"Mu for " + self.log_params["target_params"]: p1.mean()})
            wandb.log({"Sigma for " + self.log_params["target_params"]: torch.exp(p2).mean()})
        
        return x, logJ

    def reverse(self, x):

        p1 = self.p1.squeeze(-1)
        p2 = self.p2.squeeze(-1)
        
        z = (x - p1) * torch.exp(-p2)
        logJ = -torch.sum(p2)
        
        if self.log_params["log"]:
            wandb.log({"Mu for " + self.log_params["target_params"]: p1.mean()})
            wandb.log({"Sigma for " + self.log_params["target_params"]: torch.exp(p2).mean()})
        
        return z, logJ

class SimpleCouplingFlow(nn.Module):
    def __init__(self, alternate=False, log_params = {"log": False}):
        super().__init__()
        
        self.log_params = log_params
        
        self.p1 = torch.nn.Sequential(
        torch.nn.Linear(1, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
        torch.nn.Tanh()
        )
        
        self.p2 = torch.nn.Sequential(
        torch.nn.Linear(1, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
        torch.nn.Tanh()
        )
        
        self.alternate = alternate
        
    def forward(self, z):
        
        if self.alternate:
            z1, z2 = z[:,1], z[:,0]
        else:
            z1, z2 = z[:,0], z[:,1]
        p1 = self.p1(z2.unsqueeze(-1)).squeeze(-1)
        p2 = self.p2(z2.unsqueeze(-1)).squeeze(-1)
        
        x1 = p1 + torch.exp(p2) * z1
        x2 = z2
        logJ = 2*p2
        
        if self.log_params["log"]:
            wandb.log({"Mu for " + self.log_params["target_params"]: p1.mean()})
            wandb.log({"Sigma for " + self.log_params["target_params"]: torch.exp(p2).mean()})

        return torch.stack((x1, x2), dim=-1), logJ
    
    def reverse(self, x):
        if self.alternate:
            x1, x2 = x[:,1], x[:,0]
        else:
            x1, x2 = x[:,0], x[:,1]
        z2 = x2
        p1 = self.p1(z2.unsqueeze(-1)).squeeze(-1)
        p2 = self.p2(z2.unsqueeze(-1)).squeeze(-1)

        logJ = -2*p2
        z1 = (x1 - p1) * torch.exp(-p2)
        if self.log_params["log"]:
            wandb.log({"Mu for " + self.log_params["target_params"]: p1.mean()})
            wandb.log({"Sigma for " + self.log_params["target_params"]: torch.exp(p2).mean()})
            
        return torch.stack((z1, z2), dim=-1), logJ

class AffineCouplingFlow(torch.nn.Module):
    def __init__(self, net, *, mask, log_params = {"log": False}):
        super().__init__()
        
        self.log_params = log_params
        self.mask = mask
        self.net = net
        
    def forward(self, z):
        x_frozen = self.mask * z
        x_active = (1 - self.mask) * z
        net_out = self.net(x_frozen.unsqueeze(-1))
        
        p2, p1 = net_out[:,0], net_out[:,1]
        x = (1 - self.mask) * p1 + x_active * torch.exp(p2) + x_frozen
        
        axes = range(1,len(p2.size()))
        logJ = torch.sum((1 - self.mask) * p2, dim=tuple(axes))
        
        if self.log_params["log"]:
            wandb.log({"Mu for " + self.log_params["target_params"]: p1.mean()})
            wandb.log({"Sigma for " + self.log_params["target_params"]: torch.exp(p2.mean())})
        
        return x, logJ
    
    def reverse(self, x):
        fx_frozen = self.mask * x
        fx_active = (1 - self.mask) * x
        net_out = self.net(fx_frozen.unsqueeze(-1))
        
        p2, p1 = net_out[:,0], net_out[:,1]
        z = (fx_active - (1 - self.mask) * p1) * torch.exp(-p2) + fx_frozen
        
        axes = range(1,len(p2.size()))
        logJ = torch.sum((1 - self.mask)*(-p2), dim=tuple(axes))
        
        if self.log_params["log"]:
            wandb.log({"Mu for " + self.log_params["target_params"]: p1.mean()})
            wandb.log({"Sigma for " + self.log_params["target_params"]: torch.exp(p2.mean())})        
        
        return z, logJ

class MaskedAutoregressiveFlow(nn.Module):
    def __init__(self, dim, hidden_sizes, reverse_input = True, log_params = {"log": False}):
        super().__init__()
        
        self.log_params = log_params
        self.dim = dim
        self.made = SimpleMADE(dim, hidden_sizes, gaussian=True, seed=None)
        self.reverse_input = reverse_input

    def forward(self, z):
        out = self.made(z.double())
        
        mu, logp = torch.chunk(out, 2, dim=1)
        
        x = (z - mu) * torch.exp(0.5 * logp)
        x = x.flip(dims=(1,)) if self.reverse_input else x
        
        logJ = 0.5 * torch.sum(logp, dim=1)
        
        return x, logJ

    def reverse(self, x):
        x = x.flip(dims=(1,)) if self.reverse_input else x
        z = torch.zeros_like(x)
        
        for dim in range(self.dim):
            out = self.made(z)
            mu, logp = torch.chunk(out, 2, dim=1)
            mod_logp = torch.clamp(-0.5 * logp, max=10)
            z[:, dim] = mu[:, dim] + x[:, dim] * torch.exp(mod_logp[:, dim])
            
        logJ = torch.sum(mod_logp, axis=1)
        
        return z, logJ

class BatchNormFlow(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))
        self.batch_mean = None
        self.batch_var = None

    def forward(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps  # torch.mean((x - m) ** 2, axis=0) + self.eps
            self.batch_mean = None
        else:
            if self.batch_mean is None:
                self.set_batch_stats_func(x)
            m = self.batch_mean.clone()
            v = self.batch_var.clone()

        x_hat = (x - m) / torch.sqrt(v)
        x_hat = x_hat * torch.exp(self.gamma) + self.beta
        log_det = torch.sum(self.gamma - 0.5 * torch.log(v))
        return x_hat, log_det

    def reverse(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps
            self.batch_mean = None
        else:
            if self.batch_mean is None:
                self.set_batch_stats_func(x)
            m = self.batch_mean
            v = self.batch_var

        x_hat = (x - self.beta) * torch.exp(-self.gamma) * torch.sqrt(v) + m
        log_det = torch.sum(-self.gamma + 0.5 * torch.log(v))
        return x_hat, log_det

    def set_batch_stats_func(self, x):
        print("setting batch stats for validation")
        self.batch_mean = x.mean(dim=0)
        self.batch_var = x.var(dim=0) + self.eps 

class Residual(nn.Module):

    def __init__(
        self,
        net,
        backwards=True,
        reduce_memory=True,
        geom_p=0.5,
        lamb=2.0,
        n_power_series=None,
        exact_trace=False,
        brute_force=False,
        n_samples=1,
        n_exact_terms=2,
        n_dist="geometric"
    ):
        """Constructor

        Args:
          net: Neural network, must be Lipschitz continuous with L < 1
          backwards: Flag, if true the map ```f(x) = x + net(x)``` is applied in the reverse pass, otherwise it is done in forward
          reduce_memory: Flag, if true Neumann series and precomputations, for backward pass in forward pass are done
          geom_p: Parameter of the geometric distribution used for the Neumann series
          lamb: Parameter of the geometric distribution used for the Neumann series
          n_power_series: Number of terms in the Neumann series
          exact_trace: Flag, if true the trace of the Jacobian is computed exactly
          brute_force: Flag, if true the Jacobian is computed exactly in 2D
          n_samples: Number of samples used to estimate power series
          n_exact_terms: Number of terms always included in the power series
          n_dist: Distribution used for the power series, either "geometric" or "poisson"
        """
        super().__init__()
        self.backwards = backwards
        self.iresblock = iResBlock(
            net,
            n_samples=n_samples,
            n_exact_terms=n_exact_terms,
            neumann_grad=reduce_memory,
            grad_in_forward=reduce_memory,
            exact_trace=exact_trace,
            geom_p=geom_p,
            lamb=lamb,
            n_power_series=n_power_series,
            brute_force=brute_force,
            n_dist=n_dist,
        )

    def forward(self, z):
        if self.backwards:
            z, log_det = self.iresblock.reverse(z, 0)
        else:
            z, log_det = self.iresblock.forward(z, 0)
        return z, -log_det.reshape(-1)

    def reverse(self, z):
        if self.backwards:
            z, log_det = self.iresblock.forward(z, 0)
        else:
            z, log_det = self.iresblock.reverse(z, 0)
        return z, -log_det.reshape(-1)
    
class iResBlock(nn.Module):
    def __init__(
        self,
        nnet,
        geom_p=0.5,
        lamb=2.0,
        n_power_series=None,
        exact_trace=False,
        brute_force=False,
        n_samples=1,
        n_exact_terms=2,
        n_dist="geometric",
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
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1.0 - geom_p)))
        self.lamb = nn.Parameter(torch.tensor(lamb))
        self.n_samples = n_samples
        self.n_power_series = n_power_series
        self.exact_trace = exact_trace
        self.brute_force = brute_force
        self.n_exact_terms = n_exact_terms
        self.grad_in_forward = grad_in_forward
        self.neumann_grad = neumann_grad

        # store the samples of n.
        self.register_buffer("last_n_samples", torch.zeros(self.n_samples))
        self.register_buffer("last_firmom", torch.zeros(1))
        self.register_buffer("last_secmom", torch.zeros(1))

    def forward(self, x, logpx=0):
        if logpx is None:
            y = x + self.nnet(x)
            return y
        else:
            g, logdetgrad = self._logdetgrad(x)
            return x + g, logpx - logdetgrad

    def reverse(self, y, logpy=0):
        x = self._reverse_fixed_point(y)
        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x)[1]

    def _reverse_fixed_point(self, y, atol=1e-5, rtol=1e-5):
        x, x_prev = y - self.nnet(y), y
        i = 0
        tol = atol + y.abs() * rtol
        while not torch.all((x - x_prev) ** 2 / tol < 1):
            x, x_prev = y - self.nnet(x), x
            i += 1
            if i > 1000:
                break
        return x

    def _logdetgrad(self, x):
        """Returns g(x) and ```logdet|d(x+g(x))/dx|```"""

        with torch.enable_grad():
            if (self.brute_force or not self.training) and (
                x.ndimension() == 2 and x.shape[1] == 2
            ):
                ###########################################
                # Brute-force compute Jacobian determinant.
                ###########################################
                x = x.requires_grad_(True)
                g = self.nnet(x)
                # Brute-force logdet only available for 2D.
                jac = helper_functions.batch_jacobian(g, x)
                batch_dets = (jac[:, 0, 0] + 1) * (jac[:, 1, 1] + 1) - jac[
                    :, 0, 1
                ] * jac[:, 1, 0]
                return g, torch.log(torch.abs(batch_dets)).reshape(-1, 1)

            if self.n_dist == "geometric":
                geom_p = torch.sigmoid(self.geom_p).item()
                sample_fn = lambda m: helper_functions.geometric_sample(geom_p, m)
                rcdf_fn = lambda k, offset: helper_functions.geometric_1mcdf(geom_p, k, offset)
            elif self.n_dist == "poisson":
                lamb = self.lamb.item()
                sample_fn = lambda m: helper_functions.poisson_sample(lamb, m)
                rcdf_fn = lambda k, offset: helper_functions.poisson_1mcdf(lamb, k, offset)

            if self.training:
                if self.n_power_series is None:
                    # Unbiased estimation.
                    lamb = self.lamb.item()
                    n_samples = sample_fn(self.n_samples)
                    n_power_series = max(n_samples) + self.n_exact_terms
                    coeff_fn = (
                        lambda k: 1
                        / rcdf_fn(k, self.n_exact_terms)
                        * sum(n_samples >= k - self.n_exact_terms)
                        / len(n_samples)
                    )
                else:
                    # Truncated estimation.
                    n_power_series = self.n_power_series
                    coeff_fn = lambda k: 1.0
            else:
                # Unbiased estimation with more exact terms.
                lamb = self.lamb.item()
                n_samples = sample_fn(self.n_samples)
                n_power_series = max(n_samples) + 20
                coeff_fn = (
                    lambda k: 1
                    / rcdf_fn(k, 20)
                    * sum(n_samples >= k - 20)
                    / len(n_samples)
                )

            if not self.exact_trace:
                ####################################
                # Power series with trace estimator.
                ####################################
                vareps = torch.randn_like(x)

                # Choose the type of estimator.
                if self.training and self.neumann_grad:
                    estimator_fn = helper_functions.neumann_logdet_estimator
                else:
                    estimator_fn = helper_functions.basic_logdet_estimator

                # Do backprop-in-forward to save memory.
                if self.training and self.grad_in_forward:
                    g, logdetgrad = helper_functions.mem_eff_wrapper(
                        estimator_fn,
                        self.nnet,
                        x,
                        n_power_series,
                        vareps,
                        coeff_fn,
                        self.training,
                    )
                else:
                    x = x.requires_grad_(True)
                    g = self.nnet(x)
                    logdetgrad = estimator_fn(
                        g, x, n_power_series, vareps, coeff_fn, self.training
                    )
            else:
                ############################################
                # Power series with exact trace computation.
                ############################################
                x = x.requires_grad_(True)
                g = self.nnet(x)
                jac = helper_functions.batch_jacobian(g, x)
                logdetgrad = helper_functions.batch_trace(jac)
                jac_k = jac
                for k in range(2, n_power_series + 1):
                    jac_k = torch.bmm(jac, jac_k)
                    logdetgrad = logdetgrad + (-1) ** (k + 1) / k * coeff_fn(
                        k
                    ) * helper_functions.batch_trace(jac_k)

            if self.training and self.n_power_series is None:
                self.last_n_samples.copy_(
                    torch.tensor(n_samples).to(self.last_n_samples)
                )
                estimator = logdetgrad.detach()
                self.last_firmom.copy_(torch.mean(estimator).to(self.last_firmom))
                self.last_secmom.copy_(torch.mean(estimator**2).to(self.last_secmom))
            return g, logdetgrad.reshape(-1, 1)

    def extra_repr(self):
        return "dist={}, n_samples={}, n_power_series={}, neumann_grad={}, exact_trace={}, brute_force={}".format(
            self.n_dist,
            self.n_samples,
            self.n_power_series,
            self.neumann_grad,
            self.exact_trace,
            self.brute_force,
        )

class ResidualFlow(nn.Module):

    def __init__(
        self,
        input_size,
        n_blocks=[16, 16],
        intermediate_dim=64,
        factor_out=False,
        quadratic=False,
        init_layer=None,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
        learn_p=False,
        classification=False,
        classification_hdim=64,
        n_classes=10,
        block_type='resblock',
    ):
        super(ResidualFlow, self).__init__()
        self.n_scale = min(len(n_blocks), self._calc_n_scale(input_size))
        self.n_blocks = n_blocks
        self.intermediate_dim = intermediate_dim
        self.factor_out = factor_out
        self.quadratic = quadratic
        self.init_layer = init_layer
        self.actnorm = actnorm
        self.fc_actnorm = fc_actnorm
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.fc = fc
        self.coeff = coeff
        self.vnorms = vnorms
        self.n_lipschitz_iters = n_lipschitz_iters
        self.sn_atol = sn_atol
        self.sn_rtol = sn_rtol
        self.n_power_series = n_power_series
        self.n_dist = n_dist
        self.n_samples = n_samples
        self.kernels = kernels
        self.activation_fn = activation_fn
        self.fc_end = fc_end
        self.fc_idim = fc_idim
        self.n_exact_terms = n_exact_terms
        self.preact = preact
        self.neumann_grad = neumann_grad
        self.grad_in_forward = grad_in_forward
        self.first_resblock = first_resblock
        self.learn_p = learn_p
        self.classification = classification
        self.classification_hdim = classification_hdim
        self.n_classes = n_classes
        self.block_type = block_type

        if not self.n_scale > 0:
            raise ValueError('Could not compute number of scales for input of' 'size (%d,%d,%d)' % input_size)

        self.transforms = self._build_net(input_size)

        self.dims = [o[1:] for o in self.calc_output_size(input_size)]

        if self.classification:
            self.build_multiscale_classifier(input_size)

    def _build_net(self, input_size):
        _, c, w = input_size
        transforms = []
        _stacked_blocks = StackediResBlocks if self.block_type == 'resblock' else StackedCouplingBlocks
        for i in range(self.n_scale):
            transforms.append(
                _stacked_blocks(
                    initial_size=(c, w),
                    idim=self.intermediate_dim,
                    squeeze=(i < self.n_scale - 1),  # don't squeeze last layer
                    init_layer=self.init_layer if i == 0 else None,
                    n_blocks=self.n_blocks[i],
                    quadratic=self.quadratic,
                    actnorm=self.actnorm,
                    fc_actnorm=self.fc_actnorm,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    fc=self.fc,
                    coeff=self.coeff,
                    vnorms=self.vnorms,
                    n_lipschitz_iters=self.n_lipschitz_iters,
                    sn_atol=self.sn_atol,
                    sn_rtol=self.sn_rtol,
                    n_power_series=self.n_power_series,
                    n_dist=self.n_dist,
                    n_samples=self.n_samples,
                    kernels=self.kernels,
                    activation_fn=self.activation_fn,
                    fc_end=self.fc_end,
                    fc_idim=self.fc_idim,
                    n_exact_terms=self.n_exact_terms,
                    preact=self.preact,
                    neumann_grad=self.neumann_grad,
                    grad_in_forward=self.grad_in_forward,
                    first_resblock=self.first_resblock and (i == 0),
                    learn_p=self.learn_p,
                )
            )
            c, w = c if self.factor_out else c * 2, w // 2
        return nn.ModuleList(transforms)

    def _calc_n_scale(self, input_size):
        _, _, w = input_size
        n_scale = 0
        while w >= 4:
            n_scale += 1
            w = w // 2
        return n_scale

    def calc_output_size(self, input_size):
        n, c, w = input_size
        if not self.factor_out:
            k = self.n_scale - 1
            return [[n, c * 2**k, w // 2**k]]
        output_sizes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2
                w //= 2
                output_sizes.append((n, c, w))
            else:
                output_sizes.append((n, c, w))
        return tuple(output_sizes)

    def build_multiscale_classifier(self, input_size):
        n, c, w = input_size
        hidden_shapes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2 if self.factor_out else 2
                w //= 2
            hidden_shapes.append((n, c, w))

        classification_heads = []
        for i, hshape in enumerate(hidden_shapes):
            classification_heads.append(
                nn.Sequential(
                    nn.Conv1d(hshape[1], self.classification_hdim, 3, 1),
                    residual_layers.ActNorm1d(self.classification_hdim),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool1d((1)),
                )
            )
        self.classification_heads = nn.ModuleList(classification_heads)
        self.logit_layer = nn.Linear(self.classification_hdim * len(classification_heads), self.n_classes)

    def forward(self, x, logpx=torch.zeros(1), inverse=False, classify=False):

        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        if inverse:
            return self.reverse(x, logpx)
        out = []
        if classify: class_outs = []
        for idx in range(len(self.transforms)):
            if logpx is not None:
                x, logpx = self.transforms[idx].forward(x, logpx)
            else:
                x = self.transforms[idx].forward(x)
            if self.factor_out and (idx < len(self.transforms) - 1):
                d = x.size(1) // 2
                x, f = x[:, :d], x[:, d:]
                out.append(f)

            # Handle classification.
            if classify:
                if self.factor_out:
                    class_outs.append(self.classification_heads[idx](f))
                else:
                    class_outs.append(self.classification_heads[idx](x))

        out.append(x)
        out = torch.cat([o.reshape(o.size()[0], -1) for o in out], 1)
        output = out if logpx is None else (out, -logpx)
        if classify:
            h = torch.cat(class_outs, dim=1).squeeze(-1).squeeze(-1)
            logits = self.logit_layer(h)
            return output, logits
        else:
            return output

    def reverse(self, z, logpz=torch.zeros(1)):
        if len(z.size()) == 2:
            z = z.unsqueeze(1)
        if self.factor_out:
            z = z.reshape(z.shape[0], -1)
            zs = []
            i = 0
            for dims in self.dims:
                s = np.prod(dims)
                zs.append(z[:, 0:s])
                i += s
            zs = [_z.reshape(_z.size()[0], *zsize) for _z, zsize in zip(zs, self.dims)]

            if logpz is None:
                z_prev = self.transforms[-1].reverse(zs[-1])
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((z_prev, zs[idx]), dim=1)
                    z_prev = self.transforms[idx].reverse(z_prev)
                return z_prev
            else:
                z_prev, logpz = self.transforms[-1].reverse(zs[-1], logpz)
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((z_prev, zs[idx]), dim=1)
                    z_prev, logpz = self.transforms[idx].reverse(z_prev, logpz)
                return z_prev, -logpz
        else:
            z = z.reshape(z.shape[0], *self.dims[-1])
            for idx in range(len(self.transforms) - 1, -1, -1):
                if logpz is None:
                    z = self.transforms[idx].reverse(z)
                else:
                    z, logpz = self.transforms[idx].reverse(z, logpz)
            return z if logpz is None else (z, -logpz)


class StackediResBlocks(residual_layers.SequentialFlow):

    def __init__(
        self,
        initial_size,
        idim,
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        quadratic=False,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_nblocks=4,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
        learn_p=False,
    ):

        chain = []

        # Parse vnorms
        ps = []
        for p in vnorms:
            if p == 'f':
                ps.append(float('inf'))
            else:
                ps.append(float(p))
        domains, codomains = ps[:-1], ps[1:]
        assert len(domains) == len(kernels.split('-'))

        def _actnorm(size, fc):
            if fc:
                return FCWrapper(residual_layers.ActNorm1d(size[0] * size[1]))
            else:
                return residual_layers.ActNorm1d(size[0])

        def _quadratic_layer(initial_size, fc):
            if fc:
                c, w = initial_size
                dim = c * w
                return FCWrapper(residual_layers.InvertibleLinear(dim))
            else:
                return residual_layers.InvertibleLinear(initial_size[0])

        def _lipschitz_layer(fc):
            return residual_layers.get_linear if fc else residual_layers.get_linear

        def _resblock(initial_size, fc, idim=idim, first_resblock=False):
            if fc:
                return residual_layers.iResBlock(
                    FCNet(
                        input_shape=initial_size,
                        idim=idim,
                        lipschitz_layer=_lipschitz_layer(True),
                        nhidden=len(kernels.split('-')) - 1,
                        coeff=coeff,
                        domains=domains,
                        codomains=codomains,
                        n_iterations=n_lipschitz_iters,
                        activation_fn=activation_fn,
                        preact=preact,
                        dropout=dropout,
                        sn_atol=sn_atol,
                        sn_rtol=sn_rtol,
                        learn_p=learn_p,
                    ),
                    n_power_series=n_power_series,
                    n_dist=n_dist,
                    n_samples=n_samples,
                    n_exact_terms=n_exact_terms,
                    neumann_grad=neumann_grad,
                    grad_in_forward=grad_in_forward,
                )
            else:
                ks = list(map(int, kernels.split('-')))
                if learn_p:
                    _domains = [nn.Parameter(torch.tensor(0.)) for _ in range(len(ks))]
                    _codomains = _domains[1:] + [_domains[0]]
                else:
                    _domains = domains
                    _codomains = codomains
                nnet = []
                if not first_resblock and preact:
                    if batchnorm: nnet.append(residual_layers.MovingBatchNorm1d(initial_size[0]))
                    nnet.append(ACT_FNS[activation_fn](False))
                nnet.append(
                    _lipschitz_layer(fc)(
                        initial_size[0], idim, coeff=coeff, n_iterations=n_lipschitz_iters,
                        domain=_domains[0], codomain=_codomains[0], atol=sn_atol, rtol=sn_rtol
                    )
                )
                if batchnorm: nnet.append(residual_layers.MovingBatchNorm1d(idim))
                nnet.append(ACT_FNS[activation_fn](True))
                for i, k in enumerate(ks[1:-1]):
                    nnet.append(
                        _lipschitz_layer(fc)(
                            idim, idim, coeff=coeff, n_iterations=n_lipschitz_iters,
                            domain=_domains[i + 1], codomain=_codomains[i + 1], atol=sn_atol, rtol=sn_rtol
                        )
                    )
                    if batchnorm: nnet.append(residual_layers.MovingBatchNorm1d(idim))
                    nnet.append(ACT_FNS[activation_fn](True))
                if dropout: nnet.append(nn.Dropout1d(dropout, inplace=True))
                nnet.append(
                    _lipschitz_layer(fc)(
                        idim, initial_size[0], coeff=coeff, n_iterations=n_lipschitz_iters,
                        domain=_domains[-1], codomain=_codomains[-1], atol=sn_atol, rtol=sn_rtol
                    )
                )
                if batchnorm: nnet.append(residual_layers.MovingBatchNorm1d(initial_size[0]))
                return residual_layers.iResBlock(
                    nn.Sequential(*nnet),
                    n_power_series=n_power_series,
                    n_dist=n_dist,
                    n_samples=n_samples,
                    n_exact_terms=n_exact_terms,
                    neumann_grad=neumann_grad,
                    grad_in_forward=grad_in_forward,
                )

        if init_layer is not None: chain.append(init_layer)
        if first_resblock and actnorm: chain.append(_actnorm(initial_size, fc))
        if first_resblock and fc_actnorm: chain.append(_actnorm(initial_size, True))

        if squeeze:
            c, w = initial_size
            for i in range(n_blocks):
                if quadratic: chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc, first_resblock=first_resblock and (i == 0)))
                if actnorm: chain.append(_actnorm(initial_size, fc))
                if fc_actnorm: chain.append(_actnorm(initial_size, True))
            chain.append(residual_layers.SqueezeLayer(2))
        else:
            for _ in range(n_blocks):
                if quadratic: chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc))
                if actnorm: chain.append(_actnorm(initial_size, fc))
                if fc_actnorm: chain.append(_actnorm(initial_size, True))
            # Use four fully connected layers at the end.
            if fc_end:
                for _ in range(fc_nblocks):
                    chain.append(_resblock(initial_size, True, fc_idim))
                    if actnorm or fc_actnorm: chain.append(_actnorm(initial_size, True))

        super(StackediResBlocks, self).__init__(chain)


class FCNet(nn.Module):

    def __init__(
        self, input_shape, idim, lipschitz_layer, nhidden, coeff, domains, codomains, n_iterations, activation_fn,
        preact, dropout, sn_atol, sn_rtol, learn_p, div_in=1
    ):
        super(FCNet, self).__init__()
        self.input_shape = input_shape
        
        c, w = self.input_shape
        dim = c * w
        nnet = []
        last_dim = dim // div_in
        if preact: nnet.append(ACT_FNS[activation_fn](False))
        if learn_p:
            domains = [nn.Parameter(torch.tensor(0.)) for _ in range(len(domains))]
            codomains = domains[1:] + [domains[0]]
        for i in range(nhidden):
            nnet.append(
                lipschitz_layer(last_dim, idim) if lipschitz_layer == nn.Linear else lipschitz_layer(
                    last_dim, idim, coeff=coeff, n_iterations=n_iterations, domain=domains[i], codomain=codomains[i],
                    atol=sn_atol, rtol=sn_rtol
                )
            )
            nnet.append(ACT_FNS[activation_fn](True))
            last_dim = idim
        if dropout: nnet.append(nn.Dropout(dropout, inplace=True))
        nnet.append(
            lipschitz_layer(last_dim, dim) if lipschitz_layer == nn.Linear else lipschitz_layer(
                last_dim, dim, coeff=coeff, n_iterations=n_iterations, domain=domains[-1], codomain=codomains[-1],
                atol=sn_atol, rtol=sn_rtol
            )
        )
        self.nnet = nn.Sequential(*nnet)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        y = self.nnet(x)
        return y.reshape(y.shape[0], *self.input_shape)


class FCWrapper(nn.Module):

    def __init__(self, fc_module):
        super(FCWrapper, self).__init__()
        self.fc_module = fc_module

    def forward(self, x, logpx=None):
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        if logpx is None:
            y = self.fc_module(x)
            return y.reshape(*shape)
        else:
            y, logpy = self.fc_module(x, logpx)
            return y.reshape(*shape), logpy

    def reverse(self, y, logpy=None):
        shape = y.shape
        y = y.reshape(y.shape[0], -1)
        if logpy is None:
            x = self.fc_module.reverse(y)
            return x.reshape(*shape)
        else:
            x, logpx = self.fc_module.reverse(y, logpy)
            return x.reshape(*shape), logpx


class StackedCouplingBlocks(residual_layers.SequentialFlow):

    def __init__(
        self,
        initial_size,
        idim,
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        quadratic=False,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_nblocks=4,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
        learn_p=False,
    ):

        # yapf: disable
        class nonloc_scope: pass
        nonloc_scope.swap = True
        # yapf: enable

        chain = []

        def _actnorm(size, fc):
            if fc:
                return FCWrapper(residual_layers.ActNorm1d(size[0] * size[1] * size[2]))
            else:
                return residual_layers.ActNorm1d(size[0])

        def _quadratic_layer(initial_size, fc):
            if fc:
                c, w = initial_size
                dim = c * w
                return FCWrapper(residual_layers.InvertibleLinear(dim))
            else:
                return residual_layers.InvertibleLinear(initial_size[0])

        def _weight_layer(fc):
            return nn.Linear if fc else nn.Conv1d

        def _resblock(initial_size, fc, idim=idim, first_resblock=False):
            if fc:
                nonloc_scope.swap = not nonloc_scope.swap
                return residual_layers.CouplingBlock(
                    initial_size[0],
                    FCNet(
                        input_shape=initial_size,
                        idim=idim,
                        lipschitz_layer=_weight_layer(True),
                        nhidden=len(kernels.split('-')) - 1,
                        activation_fn=activation_fn,
                        preact=preact,
                        dropout=dropout,
                        coeff=None,
                        domains=None,
                        codomains=None,
                        n_iterations=None,
                        sn_atol=None,
                        sn_rtol=None,
                        learn_p=None,
                        div_in=2,
                    ),
                    swap=nonloc_scope.swap,
                )
            else:
                ks = list(map(int, kernels.split('-')))

                if init_layer is None:
                    _block = residual_layers.ChannelCouplingBlock
                    _mask_type = 'channel'
                    div_in = 2
                    mult_out = 1
                else:
                    _block = residual_layers.MaskedCouplingBlock
                    _mask_type = 'checkerboard'
                    div_in = 1
                    mult_out = 2

                nonloc_scope.swap = not nonloc_scope.swap
                _mask_type += '1' if nonloc_scope.swap else '0'

                nnet = []
                if not first_resblock and preact:
                    if batchnorm: nnet.append(residual_layers.MovingBatchNorm1d(initial_size[0]))
                    nnet.append(ACT_FNS[activation_fn](False))
                nnet.append(_weight_layer(fc)(initial_size[0] // div_in, idim))
                if batchnorm: nnet.append(residual_layers.MovingBatchNorm1d(idim))
                nnet.append(ACT_FNS[activation_fn](True))
                for i, k in enumerate(ks[1:-1]):
                    nnet.append(_weight_layer(fc)(idim, idim))
                    if batchnorm: nnet.append(residual_layers.MovingBatchNorm1d(idim))
                    nnet.append(ACT_FNS[activation_fn](True))
                if dropout: nnet.append(nn.Dropout1d(dropout, inplace=True))
                nnet.append(_weight_layer(fc)(idim, initial_size[0] * mult_out))
                if batchnorm: nnet.append(residual_layers.MovingBatchNorm1d(initial_size[0]))

                return _block(initial_size[0], nn.Sequential(*nnet), mask_type=_mask_type)

        if init_layer is not None: chain.append(init_layer)
        if first_resblock and actnorm: chain.append(_actnorm(initial_size, fc))
        if first_resblock and fc_actnorm: chain.append(_actnorm(initial_size, True))

        if squeeze:
            c, w = initial_size
            for i in range(n_blocks):
                if quadratic: chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc, first_resblock=first_resblock and (i == 0)))
                if actnorm: chain.append(_actnorm(initial_size, fc))
                if fc_actnorm: chain.append(_actnorm(initial_size, True))
            chain.append(residual_layers.SqueezeLayer(2))
        else:
            for _ in range(n_blocks):
                if quadratic: chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc))
                if actnorm: chain.append(_actnorm(initial_size, fc))
                if fc_actnorm: chain.append(_actnorm(initial_size, True))
            # Use four fully connected layers at the end.
            if fc_end:
                for _ in range(fc_nblocks):
                    chain.append(_resblock(initial_size, True, fc_idim))
                    if actnorm or fc_actnorm: chain.append(_actnorm(initial_size, True))

        super(StackedCouplingBlocks, self).__init__(chain)
