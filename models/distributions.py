import torch
from torch import nn
import helper_functions
import numpy as np


class Distribution(nn.Module):

    def forward(self, *args):
        raise RuntimeError('Forward method cannot be called for a Distribution object.')

    def log_prob(self, inputs, context=None):
        inputs = torch.as_tensor(inputs)
        if context is not None:
            context = torch.as_tensor(context)
            if inputs.shape[0] != context.shape[0]:
                raise ValueError('Number of input items must be equal to number of context items.')
        return self._log_prob(inputs, context)

    def _log_prob(self, inputs, context):
        raise NotImplementedError()

    def sample(self, num_samples, context=None, batch_size=None):
        if not helper_functions.is_positive_int(num_samples):
            raise TypeError('Number of samples must be a positive integer.')

        if context is not None:
            context = torch.as_tensor(context)

        if batch_size is None:
            return self._sample(num_samples, context)

        else:
            if not helper_functions.is_positive_int(batch_size):
                raise TypeError('Batch size must be a positive integer.')

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context))
            return torch.cat(samples, dim=0)

    def _sample(self, num_samples, context):
        raise NotImplementedError()

    def sample_and_log_prob(self, num_samples, context=None):

        samples = self.sample(num_samples, context=context)

        if context is not None:
            # Merge the context dimension with sample dimension in order to call log_prob.
            samples = helper_functions.merge_leading_dims(samples, num_dims=2)
            context = helper_functions.repeat_rows(context, num_reps=num_samples)
            assert samples.shape[0] == context.shape[0]

        log_prob = self.log_prob(samples, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = helper_functions.split_leading_dim(samples, shape=[-1, num_samples])
            log_prob = helper_functions.split_leading_dim(log_prob, shape=[-1, num_samples])

        return samples, log_prob

    def mean(self, context=None):
        if context is not None:
            context = torch.as_tensor(context)
        return self._mean(context)

class SimpleDistribution:
    def __init__(self, shape):
        self.params = []
        self.shape = shape
    
    def log_prob(self, x):
        return 0
    
    def sample_n(self, batch_size):
        return torch.zeros(batch_size).reshape(batch_size, *self.dimensions)

class Uniform(SimpleDistribution):
    def __init__(self, dimensions, low, high):
        super().__init__(dimensions)
        self.dist = torch.distributions.uniform.Uniform(low, high)
        self.low = low
        self.high = high
        self.params += [["Low", self.low], ["High", self.high]]
        
    def log_prob(self, x, context=None):
        # logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return self.shape*torch.log(torch.ones(x.shape[0])/(self.high - self.low)) 
        
    def sample_n(self, batch_size, context=None):
        x = self.dist.sample((batch_size*self.shape,))
        return x.reshape(batch_size, self.shape)
    
class SimpleNormal(SimpleDistribution):
    def __init__(self, loc, var):
        super().__init__(loc.shape)
        self.dist = torch.distributions.normal.Normal(
        torch.flatten(loc), torch.flatten(var))
        self.params += [["Mu", loc.mean().item()], ["Sigma", var.mean().item()]]
        
    def log_prob(self, x):
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return torch.sum(logp, dim=1)
    
    def sample_n(self, batch_size):
        x = self.dist.sample((batch_size, ))
        return x.reshape(batch_size, *self.shape)

class PlummerModel(SimpleDistribution):
    def __init__(self, plummer_radius = torch.ones(1)):
        super().__init__(plummer_radius.shape)
        self.plummer_radius = plummer_radius
        self.params += [["Radius", self.plummer_radius.mean().item()]]

    def log_prob(self, x):
        
        logp = torch.log(3/(4*torch.pi*(self.plummer_radius**3))) - (2.5) * torch.log(x**2)
        
        return torch.sum(logp, dim=1)

    def sample_n(self, batch_size):
        
        a = self.plummer_radius
        u = torch.rand(batch_size) 

        # Inverse CDF of the Plummer model
        r = (a**2) * (u**(-2/3) - 1)**(-0.5)

        # Generate random angles
        theta = 2 * torch.tensor(np.pi) * torch.rand(batch_size)
        phi = torch.acos(2 * torch.rand(batch_size) - 1)

        # Convert to Cartesian coordinates
        x = r * torch.sin(phi) * torch.cos(theta)
        y = r * torch.sin(phi) * torch.sin(theta)
        z = r * torch.cos(phi)

        samples = torch.stack([x, y, z], dim=1)

        return self.plummer_radius


class BaseNeuralSplineFlow(Distribution):

    def __init__(self, transform, distribution):
        super().__init__()
        self._transform = transform
        self._distribution = distribution

    def _log_prob(self, inputs, context):
        noise, logabsdet = self._transform(inputs, context=context)
        log_prob = self._distribution.log_prob(noise, context=context)
        return log_prob + logabsdet

    def _sample(self, num_samples, context):
        noise = self._distribution.sample(num_samples, context=context)

        if context is not None:
            noise = helper_functions.merge_leading_dims(noise, num_dims=2)
            context = helper_functions.repeat_rows(context, num_reps=num_samples)

        samples, _ = self._transform.inverse(noise, context=context)

        if context is not None:
            samples = helper_functions.split_leading_dim(samples, shape=[-1, num_samples])

        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        noise, log_prob = self._distribution.sample_and_log_prob(num_samples, context=context)

        if context is not None:
            noise = helper_functions.merge_leading_dims(noise, num_dims=2)
            context = helper_functions.repeat_rows(context, num_reps=num_samples)

        samples, logabsdet = self._transform.inverse(noise, context=context)

        if context is not None:
            samples = helper_functions.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = helper_functions.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        noise, _ = self._transform(inputs, context=context)
        return noise
    
    def forward(self, z):
        return self._transform.inverse(z)
    
    def reverse(self, x):
        return self._transform(x)
    
    
