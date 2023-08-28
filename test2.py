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

if torch.cuda.is_available():
    torch_device = 'cuda'
    float_dtype = np.float32 # single
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

else:
    torch_device = 'cpu'
    float_dtype = np.float64 # double
    torch.set_default_tensor_type(torch.DoubleTensor)

training_steps = 250
batch_size = 2**14
evaluate_size = 2**14
learning_rate = 0.005

prior_mu = 0
prior_sigma = 1
# target_mu = 1
# target_sigma = 3

loss_type = "Reverse KL Divergence"
dimensions = 2
uniform_distribution = distributions.Uniform(dimensions, 0, 1)
standard_normal_distribution = distributions.SimpleNormal(torch.zeros(dimensions), torch.ones(dimensions))
plummer_model = distributions.PlummerModel(torch.ones(dimensions))
prior = uniform_distribution

target_mus = [0]
target_sigmas = [1]
target_plummer_radi = [1]

flows = [
    # {"name": "box_muller_flow"},
    # {"name": "affine_flow"),
    # {"name": "trainable_affine_flow"},
    # {"name": "affine_coupling_flow", "n_layers": 32, "batch_size": batch_size, "mask_type": "checker", "hidden_sizes": [8]*16, "kernel_size": 1, "standard_ess": 0.7623},
    
    # {"name": "masked_autoregressive_flow", "n_layers": 32, "batch-size": batch_size, "hidden_sizes": [8]*16, "reverse_input":False, "standard_ess": 0.3375},
    # # {"name": "batch_norm_flow"},
    # {"name": "masked_autoregressive_flow", "n_layers": 32, "batch-size": batch_size, "hidden_sizes": [8]*16, "reverse_input":True, "standard_ess": 0.3375},
    
    {"name": "neural_spline_flow", "tail": 50, "layers": 8, "splines": [
        # {"name": "affine_coupling_transform"},
        {"name": "piecewise_rq_coupling", "standard_ess": 0.7425},
        # {"name": "piecewise_cubic_coupling"},
        # {"name": "masked_piecewise_rq_autoregressive", "standard_ess": 0.008979},
        # {"name": "masked_piecewise_cubic_autoregressive"}
        ]
    },
    
    # # {"name": "simple_residual_flow", "hidden_units": 128, "hidden_layers": 4, "latent_size": dimensions, "reduce_memory": True, "exact_trace": False, "n_dist": "geometric", "brute_force": False},
    # {"name": "residual_flow", "input_size": (batch_size, 1, dimensions), "intermediate_dim": 64, "fc": True, "fc_actnorm": True, "fc_idim":64, "fc_end": True, "standard_ess": 0.7623}
    
]

standard_ess = 0

layer_set = set()
layer_names = ""
for flow in flows:
    name = flow["name"]
    if name == "neural_spline_flow":
        splines = []
        for spline in flow["splines"]:
            splines.append(spline["name"])
            standard_ess = spline["standard_ess"]
        
        splines.sort()
    
        name = "/".join(splines) + "_nsf"
    else:
        standard_ess = flow["standard_ess"]
    
    layer_set.add(name)
s = list(layer_set)
s.sort()    
layer_names = "+".join(s)

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

columns = []

for target_sigma in target_sigmas:
    columns.append(f"Sigma = {target_sigma}")

print(f"Benchmarking {layer_names}")

for plummer_radius in target_plummer_radi:
    
    target = distributions.PlummerModel(plummer_radius*torch.ones(dimensions))

    flow_model = model.Model(dimensions=dimensions, prior_distribution=prior, prior_mu=prior_mu, prior_sigma=prior_sigma, target_distribution=target,  log_data={"layer_names": layer_names, "visualize_interval": 125, "log_interval": 5, "model_ess": True, "target_ess": False, "loss": False, "grad": True, "parameter": False, "show_dimensions": True, "average_dimensions": False, "visualize_forward": True, "visualize_reverse": False})

    for flow in flows:
        flow_model.append_layer(flow)

    optimizer = optim.Adam(flow_model.layers.parameters(), lr=learning_rate)

    # scheduler = lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=0.001, total_iters=30)

    print(f"Now training for {flow_model.target_params}")

    flow_model.train_model(optimizer, loss_type, batch_size, training_steps,
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, training_steps, 0)
                        )

    flow_model.layers.eval()
    with torch.no_grad():
        flow_model.evaluate_model(evaluate_size, False)

        ess = flow_model.compute_ess_from_distributions(evaluate_size)

        p, logp = flow_model.apply_flow_to_prior(evaluate_size)

        ess_accuracies.append([ess])
        flow_titles.append(flow_model.target_params)
        flow_samples.append([p])

        metric1, metric2, metric3, metric4, metric5 = (0,0,0,0,0)

        # metric1 = flow_model.compute_forward_kl_divergence(evaluate_size)
        metric2 = flow_model.compute_reverse_kl_divergence(evaluate_size)
        # metric3 = flow_model.compute_jensen_shanon_divergence(evaluate_size)
        metric4 = ess
        # metric5 = flow_model.compute_target_ess_from_distributions(evaluate_size)

        print(metric1, metric2, metric3, metric4, metric5)

        metrics[0] += metric1
        metrics[1] += metric2
        metrics[2] += metric3
        metrics[3] += metric4
        metrics[4] += metric5
        

table = wandb.Table(data=ess_accuracies, columns=columns)
wandb.log({("ESS Accuracies for " + layer_names): table})

for i in range(len(metrics)):
    metrics[i] /= (len(target_plummer_radi))

# cae = helper_functions.compute_cae(target_mus, target_sigmas, helper_functions.grab2d(ess_accuracies), standard_ess)
# metric_names.append("CAE")
# metrics.append(cae)

table = wandb.Table(data=[metrics], columns=metric_names)
wandb.log({("Metrics for  " + layer_names + ", Run " + wandb.run.id): table})

if dimensions == 2:
    helper_functions.plot_multiple_distribution(helper_functions.grab2d(flow_samples), "2D", f"previous_runs/{layer_names}/{wandb.run.id}/2d_posterior_samples_final_grid.png", f"{layer_names}\nAll Target Parameters", plot_labels=flow_titles)

helper_functions.plot_multiple_distribution(helper_functions.grab2d(flow_samples), "ND", f"previous_runs/{layer_names}/{wandb.run.id}/nd_posterior_samples_final_grid.png", f"{layer_names}\nAll Target Parameters ({wandb.run.id})", plot_labels=flow_titles)
