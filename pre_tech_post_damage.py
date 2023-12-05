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

## Load parameters
export_folder                    = sys.argv[1]
log_xi_min                       = float(sys.argv[2])
log_xi_max                       = float(sys.argv[3])
batch_size                       = int(sys.argv[4])
num_iterations                   = int(sys.argv[5])
A_g_prime_min                    = float(sys.argv[6])
A_g_prime_max                    = float(sys.argv[7])
pretrained_path                  = sys.argv[8]
logging_frequency                = int(sys.argv[9])
learning_rates                   = [float(x) for x in sys.argv[10].split(",")]
hidden_layer_activations         = sys.argv[11].split(",")
output_layer_activations         = sys.argv[12].split(",")
num_hidden_layers                = int(sys.argv[13])
num_neurons                      = int(sys.argv[14])
learning_rate_schedule_type      = sys.argv[15]
delta                            = float(sys.argv[16])

'''
export_folder = 'test_grad'
log_xi_min = -3.0
log_xi_max = -1.5
batch_size = 16
num_iterations = 10
A_g_prime_min = 0.1
A_g_prime_max = 0.2
pretrained_path = "None"
logging_frequency = 1
'''
if len(sys.argv) > 17:
    tensorboard = sys.argv[17] == "True"
else:
    tensorboard = False 


## Take care of pretrained path
if pretrained_path == "None":
    pretrained_path_post_tech = None
    pretrained_path_pre_tech = None
else:
    pretrained_path_post_tech = pretrained_path +   "/post_tech_post_damage" 
    pretrained_path_pre_tech  = pretrained_path +   "/pre_tech_post_damage" 

## Take care of activation functions 
hidden_layer_activations   = [None if x == "None" else x for x in hidden_layer_activations]
output_layer_activations   = [None if x == "None" else x for x in output_layer_activations]

#############################################
## Part 1
## Solve post tech post damage model
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
"gamma_1" : 0.00017675, "gamma_2" : 2 * 0.0022, "gamma_3_min" : 0.0, "gamma_3_max" : 1.0/3.0, "gamma_3_length" : 5,  \
"y_bar" : 2.0, "beta_f" : 1.86 / 1000, "eta" : 0.17, \
"varsigma" : 1.2 * 1.86 / 1000, "phi_d" : 100.0,  "phi_g" : 100.0, "Gamma" : 0.025,  \
"alpha_d" : -0.0236, "alpha_g" : -0.0236, "delta" : delta, \
"v_nn_config" : v_nn_config, "i_g_nn_config" : i_g_nn_config, "i_d_nn_config" : i_d_nn_config, \
"n_dims" : 3, "model_type" : "post_damage_post_tech" , \
"num_iterations" : num_iterations, "logging_frequency": logging_frequency, "verbose": True, "load_parameters" : None, "norm_weight" : 0.9,
"log_xi_min" : log_xi_min, "log_xi_max" : log_xi_max, "pretrained_path" : pretrained_path_post_tech, 'tensorboard' : tensorboard, "learning_rate_schedule_type" : learning_rate_schedule_type }

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

params["load_solution"]  = "model_results.json"
params["export_folder"]  = export_folder + "/post_tech_post_damage"

## i_g and i_d activations come after params because we amy want to use phi_g and phi_d
if output_layer_activations[1] == "custom" or output_layer_activations[2] == "custom":
    params["i_g_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ params["phi_g"]) / (tf.exp(2 * x) + 1.0)
    params["i_d_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ params["phi_d"]) / (tf.exp(2 * x) + 1.0)

#######################################################
##### Check if post-tech post-damage is already trained
#######################################################
if not (pathlib.Path(params["export_folder"] + "/v_nn_checkpoint_post_damage_post_tech.index").is_file()):
    ## Model has not yet been trained
    print("Post-tech post-jump model has not yet been trained. Trainning now... ")
    test_model = model.model(params)
    test_model.export_parameters()
    test_model.train()
    test_model.analyze()

    log_xi_list                  = [float(np.log(xi)) for xi in np.linspace(np.exp(log_xi_min) + 0.02, np.exp(log_xi_max) - 0.02, 10)]

    for log_xi_idx in range(len(log_xi_list)):
        print("Simulate Paths at "+ export_folder + "/log_xi_idx_" + str(log_xi_idx))
        test_model.simulate_path_post_tech_post_jump(60, 1.0 / 12.0, test_model.params["gamma_3"], test_model.params["A_g"], log_xi_list[log_xi_idx], export_folder + "/log_xi_idx_" + str(log_xi_idx))


#############################################
## Part 2
## Solve pre tech post damage model
#############################################

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


params["i_I_nn_config"]         = i_I_nn_config
params["v_post_tech_post_damage_nn_path"] = export_folder + "/post_tech_post_damage"
params["export_folder"]         = export_folder + "/pre_tech_post_damage" 
params["model_type"]            = "pre_tech_post_damage"
params["pretrained_path"]       = pretrained_path_pre_tech

test_model = model.model(params)
test_model.export_parameters()
test_model.train()
# test_model.analyze()
