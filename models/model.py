import torch
import wandb
import matplotlib.pyplot as plt
from models import distributions, flows
import helper_functions
import os
from tqdm import tqdm
        
class Model:
    def __init__(self, *, layers = [], dimensions = 2, prior_distribution = None, prior_mu = 0, prior_sigma = 1,  target_distribution = None, target_mu = 0, target_sigma = 1, log_data = False):
        self.log_data = log_data
        self.layers = torch.nn.ModuleList(layers)
        self.dimensions = dimensions

        if prior_distribution:
            self.prior_distribution = prior_distribution
        else:
            self.prior_distribution = distributions.SimpleNormal(torch.zeros(self.dimensions), torch.ones(self.dimensions))
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        
        if target_distribution:
            self.target_distribution = target_distribution
        else:
            self.target_distribution = distributions.SimpleNormal(torch.zeros(self.dimensions), torch.ones(self.dimensions))
        
        self.target_params = f"({self.dimensions}D"
        self.plot_path = f"previous_runs/{self.log_data['layer_names']}/{wandb.run.id}/{self.dimensions}D"
        for param in self.target_distribution.params:
            self.target_params += ", " + param[0] + " = " + str(param[1])
            self.plot_path += "_" + param[0] + "=" + str(param[1])
        self.target_params += ")"
        self.plot_path += "/"
        
        
        self.plot_title = f"{self.log_data['layer_names']} ({wandb.run.id})\n{self.target_params}"
        
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        
    def append_layer(self, flow):
        
        log_params = {"log": self.log_data["parameter"], "target_params": "Layer  " + str(len(self.layers) + 1) + ": " + flow["name"] +  " for " + self.target_params}
        
        if flow["name"] == "box_muller_flow":
            self.layers.append(flows.BoxMullerFlow(log_params=log_params))
        
        elif flow["name"] == "Affine Flow":
            self.layers.append(flows.AffineFlow(self.target_distribution.params[0], self.target_distribution.params[1], log_params=log_params))
        
        elif flow["name"] == "trainable_affine_flow":
            self.layers.append(flows.TrainableAffineFlow(dim=self.dimensions, log_params=log_params))
        
        elif flow["name"] == "simple_coupling_flow":
                self.layers.append(flows.SimpleCouplingFlow(alternate=flow["alternate"], log_params=locals)) 
        
        elif flow["name"] == "affine_coupling_flow":
            self.layers += helper_functions.make_affine_coupling_flow_layers(dimensions=self.dimensions, n_layers=flow["n_layers"], mask_type=flow["mask_type"], mask_shape=(self.dimensions), hidden_sizes=flow["hidden_sizes"], kernel_size=flow["kernel_size"],log_params=log_params)
        
        elif flow["name"] == "masked_autoregressive_flow":
            self.layers += helper_functions.make_masked_autoregressive_flow_layers(dimensions=self.dimensions, n_layers=flow["n_layers"], hidden_sizes=flow["hidden_sizes"], reverse_input=flow["reverse_input"], log_params=log_params)

        elif flow["name"] == "batch_norm_flow":
            self.layers.append(flows.BatchNormFlow(dim=self.dimensions))
            
        elif flow["name"] == "neural_spline_flow":
            self.layers.append(helper_functions.make_neural_spline_flow(splines=flow["splines"], layers=flow["layers"], prior_distribution=self.prior_distribution, dimensions=self.dimensions, tail=flow["tail"]))

        elif flow["name"] == "simple_residual_flow":
            self.layers.append(helper_functions.make_simple_residual_flow(flow["hidden_units"], flow["hidden_layers"], flow["latent_size"], flow["reduce_memory"], flow["exact_trace"], flow["n_dist"], flow["brute_force"]))

        elif flow["name"] == "residual_flow":
            self.layers.append(flows.ResidualFlow(input_size=flow["input_size"], intermediate_dim=flow["intermediate_dim"], fc=flow["fc"], fc_actnorm=flow["fc_actnorm"], fc_idim=flow["fc_idim"], fc_end=flow["fc_end"]))

        else:
            print("Invalid flow type")

    def apply_flow_to_prior(self, batch_size):
        x = self.prior_distribution.sample_n(batch_size)
        logq = self.prior_distribution.log_prob(x)
        
        for layer in self.layers:
            x, logJ = layer.forward(x)
            logq = logq - logJ.squeeze()
        return x, logq
    
    def apply_flow_to_samples(self, x, logq):
        for layer in self.layers:
            x, logJ = layer.forward(x)
            logq = logq - logJ.squeeze()
        
        return x, logq
    
    def apply_reverse_to_target(self, batch_size):
        x = self.target_distribution.sample_n(batch_size)
        logq = self.prior_distribution.log_prob(x)
        
        for layer in reversed(self.layers):
            x, logJ = layer.reverse(x)
            logq = logq + logJ.squeeze()
        return x, logq        
    
    def apply_reverse_to_samples(self, x, logq):
        for layer in reversed(self.layers):
            x, logJ = layer.reverse(x)
            logq = logq + logJ.squeeze()
        
        return x, self.prior_distribution.log_prob(x) + logq

    def compute_ess_from_distributions(self, batch_size):
        x, logq = self.apply_flow_to_prior(batch_size)
        logp = self.target_distribution.log_prob(x)
        
        logw = logp - logq
        log_ess = 2*torch.logsumexp(logw, dim=0) - torch.logsumexp(2*logw, dim=0)
        ess_per_cfg = torch.exp(log_ess) / len(logw)
        return ess_per_cfg
    
    def compute_target_ess_from_distributions(self, batch_size):
        y = self.target_distribution.sample_n(batch_size)
        x, logq = self.apply_reverse_to_samples(y, 0)
        logp = self.target_distribution.log_prob(y)
        
        logw = logp - logq
        log_ess = 2*torch.logsumexp(logw, dim=0) - torch.logsumexp(2*logw, dim=0)
        ess_per_cfg = torch.exp(log_ess) / len(logw)
        return ess_per_cfg

    def compute_forward_kl_divergence(self, batch_size):
        y = self.target_distribution.sample_n(batch_size)
        x, logq = self.apply_reverse_to_samples(y, 0)
        logp = self.target_distribution.log_prob(y)
        return (logp - logq).mean()
    
    def compute_reverse_kl_divergence(self, batch_size):
        
        x, logq = self.apply_flow_to_prior(batch_size)
        logp = self.target_distribution.log_prob(x)
        return (logq - logp).mean()

    def compute_jensen_shanon_divergence(self, batch_size):
        
        reverse_kl = self.compute_reverse_kl_divergence(batch_size)
        forward_kl = self.compute_forward_kl_divergence(batch_size)
        return 0.5*( forward_kl + reverse_kl )

    def visualize_model(self, batch_size, step, forward = True, reverse = True, final = False):
        self.layers.eval()
        
        with torch.no_grad():
            if forward:
                x2, logq2 = self.apply_flow_to_prior(batch_size)
                logp2 = self.target_distribution.log_prob(x2)
                    
                plt.hist(helper_functions.grab(logp2), bins=20, alpha=0.5, label='logp', histtype="step")
                plt.hist(helper_functions.grab(logq2), bins=20, alpha=0.5, label='logq', histtype="step")
                plt.hist(helper_functions.grab(logp2-logq2), bins=20, alpha=0.5, label='logp - logq', histtype="step")
                plt.legend(loc='upper right')
                plt.title(self.plot_title + f", From Posterior, Step {step}")
                plt.savefig(self.plot_path + f"log_prob_posterior_step_{step}.png")
                plt.clf()
            
            if reverse:
                y1 = self.target_distribution.sample_n(batch_size)
                x1, logq1 = self.apply_reverse_to_samples(y1, 0)
                logp1 = self.target_distribution.log_prob(y1)

                plt.hist(helper_functions.grab(logp1), bins=20, alpha=0.5, label='logp', histtype="step")
                plt.hist(helper_functions.grab(logq1), bins=20, alpha=0.5, label='logq', histtype="step")
                plt.hist(helper_functions.grab(logp1-logq1), bins=20, alpha=0.5, label='logp - logq', histtype="step")
                plt.legend(loc='upper right')
                plt.title(self.plot_title + f", From Target, Step {step}")
                plt.savefig(self.plot_path + f"log_prob_target_step_{step}.png")
                plt.clf()

        if final:
            if reverse:
                plt.hist2d(helper_functions.grab(logp1), helper_functions.grab(logq1), bins=30)
                plt.title(self.plot_title + f", Step {step}")
                plt.xlabel = "logp"
                plt.ylabel = "logq"
                plt.savefig(self.plot_path + f"log_prob_target_step_{step}.png")
                plt.clf()
            
            if forward:
                plt.hist2d(helper_functions.grab(logp2), helper_functions.grab(logq1), bins=30)
                plt.title(self.plot_title + f", Step {step}")
                plt.xlabel = "logp"
                plt.ylabel = "logq"
                plt.savefig(self.plot_path + f"log_prob_model_step_{step}.png")
                plt.clf()

        if self.log_data["show_dimensions"]:
            if self.dimensions == 2:
                helper_functions.plot_2d_distribution(helper_functions.grab(x2), self.plot_path + f"2d_posterior_samples_step_{step}.png", self.plot_title + f" 2D, Step {step}")
            helper_functions.plot_nd_distribution(helper_functions.grab(x2), self.plot_path + f"nd_posterior_samples_step_{step}.png", self.plot_title + f" ND, Step {step}", average_dimensions=self.log_data["average_dimensions"])
        
        
    def evaluate_model(self, batch_size, show_target):
        self.layers.eval()
        with torch.no_grad():
            x, logq = self.apply_flow_to_prior(batch_size)
            if self.dimensions == 2:
                helper_functions.plot_2d_distribution(helper_functions.grab(x), self.plot_path + f"trained_2d_posterior_samples.png", self.plot_title + " 2D, Posterior Sample")
                
            target_sample = None
            if show_target:
                target_sample = helper_functions.grab(self.target_distribution.sample_n(batch_size)[:, 0])             
            
            helper_functions.plot_nd_distribution(helper_functions.grab(x), self.plot_path + f"trained_nd_posterior_samples.png", self.plot_title + " ND, Posterior Sample", target_sample=target_sample, average_dimensions=self.log_data["average_dimensions"])      
            
        
    def log_model(self, batch_size, loss):
        self.layers.eval()
        
        with torch.no_grad():
            if self.log_data["model_ess"]:
                model_ess = self.compute_ess_from_distributions(batch_size)
                wandb.log({"Model ESS of " + self.target_params: model_ess})
            if self.log_data["target_ess"]:
                target_ess = self.compute_target_ess_from_distributions(batch_size)
                wandb.log({"Target ESS of " + self.target_params: target_ess})
            if self.log_data["loss"]:
                reverse_kl = self.compute_reverse_kl_divergence(batch_size)
                forward_kl = self.compute_forward_kl_divergence(batch_size)
                wandb.log({"Reverse KL": reverse_kl})
                wandb.log({"Forward KL": forward_kl})
                wandb.log({"Jensen-Shannon": (reverse_kl + forward_kl)/2})
            if self.log_data["grad"]:
                grad = torch.tensor([torch.nn.utils.clip_grad_norm_(self.layers.parameters(), 10000)])
                wandb.log({"Gradient norm of " + self.target_params: grad})
        
    def train_step(self, loss_fn, optimizer, batch_size, scheduler):
        self.layers.train()
        optimizer.zero_grad()
            
        loss = loss_fn(batch_size)
        loss.backward()
        
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        return loss
        
    def train_model(self, optimizer, loss_type, batch_size, training_steps, scheduler=None):

        if loss_type == "Jensen-Shannon Divergence":
            loss_fn = self.compute_jensen_shanon_divergence         
        elif loss_type == "Forward KL Divergence":
            loss_fn = self.compute_forward_kl_divergence 
        elif loss_type == "Reverse KL Divergence":
            loss_fn = self.compute_reverse_kl_divergence 
            
        tbar = tqdm(range(1, training_steps + 1))

        for step in tbar:            
            loss = self.train_step(loss_fn , optimizer, batch_size, scheduler)
            
            if step % self.log_data["log_interval"] == 0:
                self.log_model(batch_size, loss)
                
            if step % self.log_data["visualize_interval"] == 0:
                self.visualize_model(batch_size, training_steps, forward=self.log_data["visualize_forward"], reverse=self.log_data["visualize_reverse"])