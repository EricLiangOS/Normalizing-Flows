import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
import models
from models import distributions, model
import helper_functions
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import argparse
import ast

if torch.cuda.is_available():
    torch_device = 'cuda'
    float_dtype = np.float32 # single
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

else:
    torch_device = 'cpu'
    float_dtype = np.float64 # double
    torch.set_default_tensor_type(torch.DoubleTensor)

parser = argparse.ArgumentParser()
parser.add_argument("--dimensions", help="Number of dimensions",type=int, default=2)
parser.add_argument("--loss", help="Number of dimensions",type=str, default="Reverse KL Divergence")
parser.add_argument("--flows", help="Type of flow", type=str, default="neural_spline_flow_coupling")
parser.add_argument("--target_mus", help="List of target mus", type=ast.literal_eval, default=[250])
parser.add_argument("--target_sigmas", help="List of target sigmas", type=ast.literal_eval, default=[1000])
parser.add_argument("--standard_ess", help="ESS to be used for CAE",type=float, default=1)
args = parser.parse_args()

training_steps = 1
batch_size = 2**23
evaluate_size = 2**23
learning_rate = 0.005

prior_mu = 0
prior_sigma = 1
# target_mu = 1
# target_sigma = 3

loss_type = args.loss
dimensions = args.dimensions
uniform_distribution = distributions.Uniform(dimensions, 0, 1)
standard_normal_distribution = distributions.SimpleNormal(torch.zeros(dimensions), torch.ones(dimensions))
prior = uniform_distribution


target_mus = args.target_mus
target_sigmas = args.target_sigmas

# flows = helper_functions.make_flow_list(args.flows, batch_size, dimensions)
flows = []

standard_ess = args.standard_ess

layer_set = set()
layer_names = ""
for flow in flows:
    name = flow["name"]
    if name == "neural_spline_flow":
        splines = []
        for spline in flow["splines"]:
            splines.append(spline["name"])
        
        splines.sort()
    
        name = "/".join(splines) + "_nsf"
    
    layer_set.add(name)

s = list(layer_set)
s.sort()    
layer_names = "+".join(s)
    
if layer_names == "":
    layer_names = "None"

configurations = {"batch_size": batch_size,
                "learning_rate": learning_rate,
                "loss-type": loss_type,
                "flow_type": layer_names}

helper_functions.initiate_wandb(configurations)

if not os.path.exists(f"previous_runs/{layer_names}"):
    os.makedirs(f"previous_runs/{layer_names}")
if not os.path.exists(f"previous_runs/{layer_names}/{wandb.run.id}"):
    os.makedirs(f"previous_runs/{layer_names}/{wandb.run.id}")

metric_names = ["KL Divergence", "Reverse KL Divergence", "Jensen-Shannon Divergence", "ESS", "Target ESS"]
metrics = [0] * len(metric_names)

ess_accuracies = []
flow_titles = []
flow_samples = []
flow_inversed = []

columns = []

for target_sigma in target_sigmas:
    columns.append(f"Sigma = {target_sigma}")

print(f"Benchmarking {layer_names}")

mu_index = 0

while mu_index < len(target_mus):
    target_mu = target_mus[mu_index]
    sigma_index = 0
    
    ess_row = []
    title_row = []
    sample_row = []
    inverse_row = []
    
    while sigma_index < len(target_sigmas):
        target_sigma = target_sigmas[sigma_index]    
        try:
            target = distributions.SimpleNormal(target_mu*torch.ones(dimensions), target_sigma*torch.ones(dimensions))
            
            flow_model = model.Model(dimensions=dimensions, prior_distribution=prior, prior_mu=prior_mu, prior_sigma=prior_sigma, target_distribution=target, target_mu=target_mu, target_sigma=target_sigma, log_data={"layer_names": layer_names, "visualize_interval": 125, "log_interval": 5, "model_ess": True, "target_ess": True, "loss": True, "grad": True, "parameter": False, "show_dimensions": True, "average_dimensions": False, "visualize_forward": True, "visualize_reverse": False})
            
            for flow in flows:
                flow_model.append_layer(flow)
            
            # optimizer = optim.Adam(flow_model.layers.parameters(), lr=learning_rate)

            # scheduler = lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=0.001, total_iters=30)
            
            # print(f"Now training for {flow_model.target_params}")
            
            # flow_model.train_model(optimizer, loss_type, batch_size, training_steps,
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, training_steps, 0)
            #                     )
            
            # flow_model.layers.eval()
            with torch.no_grad():
                flow_model.evaluate_model(evaluate_size, True)

                ess = flow_model.compute_ess_from_distributions(evaluate_size)
                if target_mu == 0 and target_sigma == 1:
                    standard_ess = ess

                p, logp = flow_model.apply_flow_to_prior(evaluate_size)

                ess_row.append(ess)
                title_row.append(flow_model.target_params)
                sample_row.append(p)

                q, _ = flow_model.apply_reverse_to_samples(p, logp)
                inverse_row.append(q)

                metric1 = flow_model.compute_forward_kl_divergence(evaluate_size)
                metric2 = flow_model.compute_reverse_kl_divergence(evaluate_size)
                metric3 = flow_model.compute_jensen_shanon_divergence(evaluate_size)
                metric4 = ess
                metric5 = flow_model.compute_target_ess_from_distributions(evaluate_size)

                print(metric1.item(), metric2.item(), metric3.item(), metric4.item(), metric5.item())

                metrics[0] += metric1
                metrics[1] += metric2
                metrics[2] += metric3
                metrics[3] += metric4
                metrics[4] += metric5
                
                sigma_index += 1
        except:
            print("something went wrong")
        
    ess_accuracies.append(ess_row)
    flow_titles.append(title_row)
    flow_samples.append(sample_row)
    flow_inversed.append(inverse_row)
    
    mu_index += 1

table = wandb.Table(data=ess_accuracies, columns=columns)
wandb.log({("ESS Accuracies for " + layer_names): table})

for i in range(len(metrics)):
    metrics[i] /= (len(target_mus) * len(target_sigmas))

cae = helper_functions.compute_cae(target_mus, target_sigmas, helper_functions.grab2d(ess_accuracies), standard_ess)
metric_names.append("CAE")
metrics.append(cae)

table = wandb.Table(data=[metrics], columns=metric_names)
wandb.log({("Metrics for  " + layer_names + ", Run " + wandb.run.id): table})

if dimensions == 2:
    helper_functions.plot_multiple_distribution(helper_functions.grab2d(flow_samples), "2D", f"previous_runs/{layer_names}/{wandb.run.id}/2d_posterior_samples_final_grid.png", f"{layer_names}\nAll Target Parameters", plot_labels=flow_titles)

helper_functions.plot_multiple_distribution(helper_functions.grab2d(flow_samples), "ND", f"previous_runs/{layer_names}/{wandb.run.id}/nd_posterior_samples_final_grid.png", f"{layer_names}\nAll Target Parameters ({wandb.run.id})", plot_labels=flow_titles)
