##################################################################
##################################################################
#### This program solves pre-tech post-damage models
####
##################################################################
##################################################################
##################################################################

import model 
import numpy as np
import tensorflow as tf 
import sys 
import pathlib

#############################################
#############################################
#############################################
#############################################
#############################################

export_folder                                           = sys.argv[1]
post_tech_pre_damage_export_folder                      = sys.argv[2]
pre_tech_post_damage_export_folder                      = sys.argv[3]
steps                                                   = int(round(float(sys.argv[4])))
log_xi_min                                              = float(sys.argv[5])
log_xi_max                                              = float(sys.argv[6])
batch_size                                              = int(sys.argv[7])
num_iterations                                          = int(sys.argv[8])
A_g_prime_min                                           = float(sys.argv[9])
A_g_prime_max                                           = float(sys.argv[10])
pretrained_path                                         = sys.argv[11]
logging_frequency                                       = int(sys.argv[12])
learning_rates                                          = [float(x) for x in sys.argv[13].split(",")]
hidden_layer_activations                                = sys.argv[14].split(",")
output_layer_activations                                = sys.argv[15].split(",")
num_hidden_layers                                       = int(sys.argv[16])
num_neurons                                             = int(sys.argv[17])
learning_rate_schedule_type                             = sys.argv[18]
delta                                                   = float(sys.argv[19])


## Take care of pretrained path
if pretrained_path == "None":
    pretrained_path = None

## Take care of tensorboard
if len(sys.argv) > 20:
    tensorboard = sys.argv[20] == "True"
else:
    tensorboard = False 

## Take care of activation functions 
hidden_layer_activations   = [None if x == "None" else x for x in hidden_layer_activations]
output_layer_activations   = [None if x == "None" else x for x in output_layer_activations]

decay_rate                                              = 0.9
log_xi_list                                             = [float(np.log(xi)) for xi in np.linspace(np.exp(log_xi_min) + 0.02, np.exp(log_xi_max) - 0.02, 10)]

## Load parameters

#############################################
## Part 2
## Solve pre tech pre damage model
#############################################

## This model has three state variables


v_nn_config   = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[0], "dim" : 1, "nn_name" : "v_nn"}
v_nn_config["final_activation"] = output_layer_activations[0]

i_g_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[1], "dim" : 1, "nn_name" : "i_g_nn"}
i_g_nn_config["final_activation"] = output_layer_activations[1]

i_d_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[2], "dim" : 1, "nn_name" : "i_d_nn"}
i_d_nn_config["final_activation"] = output_layer_activations[2]

i_I_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[3], "dim" : 1, "nn_name" : "i_I_nn"}
i_I_nn_config["final_activation"] = output_layer_activations[3]

## Create params struct 
params = {"batch_size" : batch_size, "R_min" : 0.01, \
"R_max" : 0.99, "logK_min" : 4.0,\
"logK_max" : 7.0, "Y_min" : 10e-3, "Y_max" : 3.0, \
"log_I_g_max" : 6.0, "log_I_g_min": 1.0, \
"sigma_d" : 0.15 , "sigma_g" : 0.15, "A_d" : 0.12, "A_g_prime_min" : A_g_prime_min, "A_g_prime_max" : A_g_prime_max, "A_g_prime_length" : 3 , \
"gamma_1" : 0.00017675, "gamma_2" : 2 * 0.0022, "gamma_3_idx" : 0,  "gamma_3_min" : 0.0, "gamma_3_max" : 1.0/3.0, "gamma_3_length" : 5,  \
"y_bar" : 2.0, "beta_f" : 1.86 / 1000, "eta" : 0.17, \
"varsigma" : 1.2 * 1.86 / 1000, "phi_d" : 100.0,  "phi_g" : 100.0, "Gamma" : 0.025,  \
"alpha_d" : -0.0236, "alpha_g" : -0.0236, "delta" : delta, \
"v_nn_config" : v_nn_config, "i_g_nn_config" : i_g_nn_config, "i_d_nn_config" : i_d_nn_config, \
"model_type" : "post_damage_post_tech" , \
"num_iterations" : num_iterations, "logging_frequency": logging_frequency, "verbose": True, "load_parameters" : None,
"r_1": 1.5, "r_2": 2.5, "y_lower_bar" : 1.5, "norm_weight" : 0.9,
"log_xi_min" : log_xi_min, "log_xi_max" : log_xi_max, "pretrained_path" : pretrained_path, 'tensorboard' : tensorboard, "learning_rate_schedule_type" : learning_rate_schedule_type }

params["load_solution"]  = "model_results.json"

if output_layer_activations[1] == "custom" or output_layer_activations[2] == "custom":
    params["i_g_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ params["phi_g"]) / (tf.exp(2 * x) + 1.0)
    params["i_d_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ params["phi_d"]) / (tf.exp(2 * x) + 1.0)



## This model has four state variables. 
## Add in paramters associated with this 4d model 

params["n_dims"]          = 4
params["zeta"]            = 0.0
params["sigma_I"]         = 0.016
params["varrho"]          = 448
params["log_I_g_min"]     = 1.0
params["log_I_g_max"]     = 6.0
params["A_g"]             = 0.10
params["psi_1"]           = 0.5
params["psi_0"]           = 0.10583


if params["learning_rate_schedule_type"] == "None":
    lr_schedulers = learning_rates
    params["optimizers"] = [tf.keras.optimizers.Adam( learning_rate = lr_scheduler) for lr_scheduler in lr_schedulers]
elif params["learning_rate_schedule_type"] == "piecewiseconstant":
    boundaries            = [int(round(x)) for x in np.linspace(0,num_iterations,5)][1:-1]
    values_list           = [[learning_rate / np.power(2,x) for x in range(len(boundaries)+1)] for learning_rate in learning_rates]
    lr_schedulers         = [ tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values) for values in values_list]
    params["optimizers"] = [tf.keras.optimizers.Adam( learning_rate = lr_scheduler) for lr_scheduler in lr_schedulers]
elif params["learning_rate_schedule_type"] == "sgd+piecewiseconstant":
    boundaries            = [int(round(x)) for x in np.linspace(0,num_iterations,5)][1:-1]
    values_list           = [[learning_rate / np.power(2,x) for x in range(len(boundaries)+1)] for learning_rate in learning_rates]
    lr_schedulers         = [ tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values) for values in values_list]
    params["optimizers"] = [tf.keras.optimizers.legacy.SGD( learning_rate = lr_scheduler) for lr_scheduler in lr_schedulers]
elif params["learning_rate_schedule_type"] == "sgd":
    lr_schedulers = learning_rates
    params["optimizers"] = [tf.keras.optimizers.legacy.SGD( learning_rate = lr_scheduler) for lr_scheduler in lr_schedulers]


params["i_I_nn_config"]                   = i_I_nn_config
params["v_pre_tech_post_damage_nn_path"]  = pre_tech_post_damage_export_folder + "/pre_tech_post_damage"  
params["v_post_tech_pre_damage_nn_path"]  = post_tech_pre_damage_export_folder 
params["export_folder"]                   = export_folder
params["model_type"]                      = "pre_tech_pre_damage"

## Uncertainty parameters

params['r_1']   = 1.5
params['r_2']   = 2.5
params['y_lower_bar']  = 1.5

##########
test_model = model.model(params)
test_model.export_parameters()
test_model.train()
# test_model.analyze()

for log_xi_idx in range(len(log_xi_list)):
    test_model.simulate_path(60, 1.0 / 12.0, log_xi_list[log_xi_idx], export_folder + "/log_xi_idx_" + str(log_xi_idx))

