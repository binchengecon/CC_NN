######################################################################
######################################################################
##########       This file defines the model class          ########## 
######################################################################
######################################################################


import numpy as np
import tensorflow as tf
import time
import logging
from tensorflow import keras
import json 
import pathlib
import matplotlib.pyplot as plt
import os 


tf.random.set_seed(11117)


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config):
        super(FeedForwardSubNet, self).__init__(name = config["nn_name"] + ".init_layer")
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5),
                name = config["nn_name"] + ".bn." + str(_)
            )
            for _ in range(len(config["num_hiddens"]) + 1)]
        
        if config['activation'] is not None and "relu" in config['activation']:
            initializer = tf.keras.initializers.HeNormal(seed=0)
        else:
            initializer = tf.keras.initializers.GlorotUniform(seed=0)

        self.dense_layers = [tf.keras.layers.Dense(config["num_hiddens"][i],
                                                   use_bias=config['use_bias'],
                                                   activation=config['activation'],
                                                   kernel_initializer = initializer,
                                                   name = config["nn_name"] + ".dense." + str(i))
                             for i in range(len(config["num_hiddens"]))]
        # final output should be gradient of size dim
        try:
            if config['final_activation'] is None:
                initializer = tf.keras.initializers.GlorotUniform(seed=0)
            elif "relu" in config['final_activation']:
                initializer = tf.keras.initializers.HeNormal(seed=0)
            else:
                initializer = tf.keras.initializers.GlorotUniform(seed=0)
        except:
            initializer = tf.keras.initializers.GlorotUniform(seed=0)

        self.dense_layers.append(tf.keras.layers.Dense(config["dim"], 
        kernel_initializer = initializer, 
        activation=config['final_activation'], use_bias = True, name = config["nn_name"] + ".output" ))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x, training)
        x_inputs = []
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x_inputs.append(x)
        x = tf.keras.layers.Add()(x_inputs)
        x = self.dense_layers[-1](x)
        return x
    
class model:
    ## This class defines a model

    def __init__(self, params):


        ## Load parameters

        self.params  = params 

        if 'tensorboard' not in params.keys():
            print("Tensorboard option not detected; setting to False by default.")
            self.params['tensorboard'] = False 

        print("Tensorboard boolean =", self.params['tensorboard'] )

        if params["load_solution"] is not None:
            ## Parse model solution from a file
            
            self.solution_fd              = json.load(open(params["load_solution"]))

            self.solution_fd['stateSpace'] = np.array(self.solution_fd['grid_tuple'], ndmin=2).transpose()
            self.solution_fd['nK']         = len( np.unique(self.solution_fd['grid_tuple'][0]))
            self.solution_fd['nR']         = len(np.unique(self.solution_fd['grid_tuple'][1]))
            self.solution_fd['nT']         = len(np.unique(self.solution_fd['grid_tuple'][2]))

            self.solution_fd['V']          = np.array(self.solution_fd['V'])
            self.solution_fd['i_G']        = np.array(self.solution_fd['i_G'])
            self.solution_fd['i_B']        = np.array(self.solution_fd['i_B'])

            self.params["A_d"]            = self.solution_fd["A_B"]

            self.params["alpha_d"]        = -self.solution_fd["delta_i"]
            self.params["alpha_g"]        = -self.solution_fd["delta_i"]
            self.params["sigma_d"]        = self.solution_fd["sigma_K"]
            self.params["sigma_g"]        = self.solution_fd["sigma_K"]
            self.params["varsigma"]       = self.solution_fd["sigma_T"]
            self.params["phi_d"]          = self.solution_fd["theta"]
            self.params["phi_g"]          = self.solution_fd["theta"]
            self.params["gamma_1"]        = self.solution_fd["gamma_1"]
            self.params["gamma_2"]        = self.solution_fd["gamma_2"]

            self.params["eta"]            = self.solution_fd["lambda"]
            self.params["beta_f"]         = self.solution_fd["beta"]

            self.params["logK_min"]       = self.solution_fd["K_min"]
            self.params["logK_max"]       = self.solution_fd["K_max"]
            self.params["R_min"]          = self.solution_fd["R_min"]
            self.params["R_max"]          = self.solution_fd["R_max"]
            self.params["Y_min"]          = self.solution_fd["T_min"]
            self.params["Y_max"]          = self.solution_fd["T_max"]

            self.params["gamma_3"]        = self.solution_fd["gamma_3"] ### Need to fix this; shouldn't be hard-coded
            self.params["A_g"]        = self.solution_fd["A_B"] ### Need to fix this; shouldn't be hard-coded
            self.params["log_xi"]         = np.log(self.solution_fd["xi"])

        self.params["A_g_prime_list"]     = np.linspace(self.params["A_g_prime_min"], self.params["A_g_prime_max"], self.params["A_g_prime_length"]).tolist
        self.params["gamma_3_list"]       = np.linspace(self.params["gamma_3_min"], self.params["gamma_3_max"], self.params["gamma_3_length"]).tolist()

        ## Create tensors to store normalizing constants 
        consumption_guess =  ( np.exp(self.params["logK_max"]) + np.exp(self.params["logK_min"]) ) / 2 * 0.1 ## assume consuming 10% of capital

        self.flow_pv_norm                          =  tf.ones(shape = (self.params['batch_size'],1) ) * self.params['delta'] * np.log(consumption_guess)
        self.marginal_utility_of_consumption_norm  =  tf.ones(shape = (self.params['batch_size'],1) ) * self.params['delta'] / consumption_guess

        ## Create neural networks
        self.v_nn    = FeedForwardSubNet(self.params['v_nn_config'])
        self.i_g_nn  = FeedForwardSubNet(self.params['i_g_nn_config'])
        self.i_d_nn  = FeedForwardSubNet(self.params['i_d_nn_config'])
            
        if "pre_" in self.params["model_type"] and "post_" in self.params["model_type"]:

            ## Load post tech post damage model

            self.v_post_tech_post_damage_nn = FeedForwardSubNet(self.params['v_nn_config'])
                
            self.v_post_tech_post_damage_nn.build( (self.params["batch_size"], 6) )  ## need to build network; pre_tech has four state variables so we don't need to do anything (remember gamma_3 is a pseudo state variable)
            ## 5 inputs here: logK, R, Y, gamma_3, (xi as a hidden state variable), + A_g 

            self.v_post_tech_post_damage_nn.load_weights( self.params["v_post_tech_post_damage_nn_path"]  + '/v_nn_checkpoint_post_damage_post_tech')


        if "pre_tech" in self.params["model_type"] and "pre_damage" in self.params["model_type"]:

            self.v_pre_tech_post_damage_nn = FeedForwardSubNet(self.params['v_nn_config'])

            self.v_pre_tech_post_damage_nn.build( (self.params["batch_size"], 6) ) 
            ## 6 inputs here: logK, R, Y, gamma_3, and log_I_g,  (xi as a hidden state variable)

            self.v_pre_tech_post_damage_nn.load_weights( self.params["v_pre_tech_post_damage_nn_path"]  + '/v_nn_checkpoint_pre_tech_post_damage')


            self.v_post_tech_pre_damage_nn = FeedForwardSubNet(self.params['v_nn_config'])

            self.v_post_tech_pre_damage_nn.build( (self.params["batch_size"], 5) ) 
            ## 5 inputs here: logK, R, Y, (xi as a hidden state variable), + A_g

            self.v_post_tech_pre_damage_nn.load_weights( self.params["v_post_tech_pre_damage_nn_path"]  + '/v_nn_checkpoint_pre_damage_post_tech')

        if "pre_tech" in self.params["model_type"]:
            print("Pre tech model detected. Building a neural network for i_I")
            self.i_I_nn  = FeedForwardSubNet(self.params['i_I_nn_config'])

        ## Create folder 
        pathlib.Path(self.params["export_folder"]).mkdir(parents=True, exist_ok=True) 

        ## Create ranges for sampling later 

        self.params["state_intervals"] = {}

        self.params["state_intervals"]["log_xi"] = tf.reshape(tf.linspace(self.params['log_xi_min'], self.params['log_xi_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
        self.params["state_intervals"]["log_xi_interval_size"] =  self.params["state_intervals"]["log_xi"][1] -  self.params["state_intervals"]["log_xi"][0]

        if "post_damage" in self.params["model_type"]:
            self.params["state_intervals"]["gamma_3"] = tf.reshape(tf.linspace(self.params['gamma_3_min'], self.params['gamma_3_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
            self.params["state_intervals"]["gamma_3_interval_size"] =  self.params["state_intervals"]["gamma_3"][1] -  self.params["state_intervals"]["gamma_3"][0]

        if "post_tech" in self.params["model_type"]:
            self.params["state_intervals"]["A_g_prime"] = tf.reshape(tf.linspace(self.params['A_g_prime_min'], self.params['A_g_prime_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
            self.params["state_intervals"]["A_g_prime_interval_size"] =  self.params["state_intervals"]["A_g_prime"][1] -  self.params["state_intervals"]["A_g_prime"][0]
 

        self.params["state_intervals"]["logK"]     =  tf.reshape(tf.linspace(self.params['logK_min'], self.params['logK_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
        self.params["state_intervals"]["logK_interval_size"] =  self.params["state_intervals"]["logK"][1] -  self.params["state_intervals"]["logK"][0]

        self.params["state_intervals"]["R"]        =  tf.reshape(tf.linspace(self.params['R_min'], self.params['R_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
        self.params["state_intervals"]["R_interval_size"] =  self.params["state_intervals"]["R"][1] -  self.params["state_intervals"]["R"][0]

        self.params["state_intervals"]["Y"]        =  tf.reshape(tf.linspace(self.params['Y_min'], self.params['Y_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
        self.params["state_intervals"]["Y_interval_size"] =  self.params["state_intervals"]["Y"][1] -  self.params["state_intervals"]["Y"][0]

        if self.params["n_dims"] == 4:
            self.params["state_intervals"]["log_I_g"]        =  tf.reshape(tf.linspace(self.params['log_I_g_min'], self.params['log_I_g_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
            self.params["state_intervals"]["log_I_g_interval_size"] =  self.params["state_intervals"]["log_I_g"][1] -  self.params["state_intervals"]["log_I_g"][0]


        ## Create objects to generate checkpoints for tensorboard
        pathlib.Path(self.params["export_folder"] + '/logs/train/').mkdir(parents=True, exist_ok=True) 
        pathlib.Path(self.params["export_folder"] + '/logs/test/').mkdir(parents=True, exist_ok=True) 

        self.train_writer = tf.summary.create_file_writer( self.params["export_folder"] + '/logs/train/')
        self.test_writer  = tf.summary.create_file_writer( self.params["export_folder"] + '/logs/test/')

    def custom_tanh(self):
        return 1.0 - (1.0 + 1.0/ self.params['phi_d'] ) / (tf.exp(2 * x) + 1)

    def sample(self):

        ## Sample xi 
        # Always a pseudo state variable
        # Need to sample for four jump state

        offsets = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
        log_xi = tf.random.shuffle(self.params["state_intervals"]["log_xi"][:-1] +  self.params["state_intervals"]["log_xi_interval_size"] * offsets)


        ## Sample gamma_3
        #  In pre damage jump states, the evolution process of climate damage can be viewed as a special case where gamma_3 = 0.
        #  Hence we sample a zero-valued tensor of gamma_3 
        if "post_damage" in self.params["model_type"]:

            offsets      = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
            gamma_3      = tf.random.shuffle(self.params["state_intervals"]["gamma_3"][:-1] + self.params["state_intervals"]["gamma_3_interval_size"] * offsets)


        else:
            gamma_3      = tf.zeros(shape = (self.params['batch_size'],1) )

        ## Sample A_g_prime
        #  In pre tech jump states, the output constraint can be viewed as a special case where A_g_prime = A_g
        #  Hence we sample a constant-valued tensor of A_g_prime
         
        if "post_tech" in self.params["model_type"]:

            offsets      = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
            A_g_prime      = tf.random.shuffle(self.params["state_intervals"]["A_g_prime"][:-1] + self.params["state_intervals"]["A_g_prime_interval_size"] * offsets)


        else:
            A_g_prime      = tf.constant(self.params["A_g"],dtype=float32) * tf.ones(shape = (self.params['batch_size'],1) )



        offsets      = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
        logK         = tf.random.shuffle(self.params["state_intervals"]["logK"][:-1] + self.params["state_intervals"]["logK_interval_size"] * offsets)

        offsets      = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
        R            = tf.random.shuffle(self.params["state_intervals"]["R"][:-1] + self.params["state_intervals"]["R_interval_size"] * offsets)

        offsets      = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
        Y            = tf.random.shuffle(self.params["state_intervals"]["Y"][:-1] + self.params["state_intervals"]["Y_interval_size"] * offsets)

        if self.params["n_dims"] == 4:
            offsets            = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
            log_I_g            = tf.random.shuffle(self.params["state_intervals"]["log_I_g"][:-1] + self.params["state_intervals"]["log_I_g_interval_size"] * offsets)

            return logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g

        else:
            return logK, R, Y, gamma_3, A_g_prime, log_xi

    @tf.function
    def pde_rhs(self, logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g = None, training = True):

        ## This is the RHS of the HJB equation

        ## Transform inputs

        # if "post_damage" in self.params["model_type"] and "post_tech" in self.params["model_type"]:
        #     X = tf.concat([logK, R, Y, gamma_3, log_xi], 1)
    
        if "post_damage" in self.params["model_type"] and "post_tech" in self.params["model_type"]:
            X = tf.concat([logK, R, Y, gamma_3, log_xi, A_g_prime], 1)
            # print("logK dtype:", logK.dtype)
            # print("R dtype:", R.dtype)
            # print("Y dtype:", Y.dtype)
            # print("gamma_3 dtype:", gamma_3.dtype)
            # print("log_xi dtype:", log_xi.dtype)
            # print("A_g_prime dtype:", A_g_prime.dtype)
            # print("A_g_prime dtype:", A_g_prime)
            # print("log_xi dtype:", log_xi)
            
        if "post_damage" in self.params["model_type"] and "pre_tech" in self.params["model_type"]:
            X = tf.concat([logK, R, Y, gamma_3, log_xi, log_I_g], 1)
            i_I                             = self.i_I_nn(X)
            i_I_capped  = tf.reshape( tf.math.maximum( i_I , 0.000000001), [self.params["batch_size"], 1])

        if "pre_damage" in self.params["model_type"] and "post_tech" in self.params["model_type"]:
            X = tf.concat([logK, R, Y, log_xi, A_g_prime], 1)

        if "pre_damage" in self.params["model_type"] and "pre_tech" in self.params["model_type"]:
            X           = tf.concat([logK, R, Y, log_xi, log_I_g], 1)
            i_I         = self.i_I_nn(X)
            i_I_capped  = tf.reshape( tf.math.maximum( i_I , 0.000000001), [self.params["batch_size"], 1])

        xi  = tf.exp(log_xi)

        ## Evalute neural networks 
        v            = self.v_nn(X)
        i_g          = self.i_g_nn(X)
        i_d          = self.i_d_nn(X)

        K = tf.reshape(tf.exp(logK), [self.params['batch_size'], 1])

        ## 
        dv_dlogK                 = tf.reshape(tf.gradients(v, logK, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dv_ddlogK                = tf.reshape(tf.gradients(dv_dlogK, logK, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dv_dlogKdR               = tf.reshape(tf.gradients(dv_dlogK, R, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])

        dv_dR                    = tf.reshape(tf.gradients(v, R, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dv_ddR                   = tf.reshape(tf.gradients(dv_dR, R, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])

        dv_dY                    = tf.reshape(tf.gradients(v, Y, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dv_ddY                   = tf.reshape(tf.gradients(dv_dY, Y, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])

        if self.params["n_dims"] == 4:

            ## Compute terms related to log_I_g
            dv_dI_g                  = tf.reshape(tf.gradients(v, log_I_g, unconnected_gradients='zero')[0], [self.params["batch_size"], 1])
            dv_ddI_g                 = tf.reshape(tf.gradients(dv_dI_g, log_I_g, unconnected_gradients='zero')[0], [self.params["batch_size"], 1])

        ## pre tech and pre damage model
        if "pre_tech" in self.params["model_type"] and "pre_damage" in self.params["model_type"]:
            f_ms = []
            f_m_logs = []
            v_m_vals = []
            
            g_js = []
            g_j_logs = []
            v_j_vals = []
            
            for k in range(self.params["gamma_3_length"]):
                    
                X_pre_tech_post_damage = tf.concat([logK, R, Y, 
                tf.ones(tf.shape(Y)) * self.params["gamma_3_list"][k], log_xi, log_I_g], 1)
                v_m                    = self.v_pre_tech_post_damage_nn(X_pre_tech_post_damage)
                v_m_vals.append( v_m )
                f_m       = tf.exp(-1.0/ xi * (v_m - v))
                f_m_log   = -1.0/ xi * (v_m - v)

                f_ms.append(f_m)
                f_m_logs.append(f_m_log)



            for j in range(self.params["A_g_prime_length"]):

                X_post_tech_pre_damage = tf.concat([logK, R, Y, log_xi, log_I_g, 
                tf.ones(tf.shape(Y)) * self.params["A_g_prime_list"][j]], 1)
                v_j                    = self.v_post_tech_pre_damage_nn(X_post_tech_pre_damage)
                v_j_vals.append( v_j )
                v_diff_temp                        = v_j - v
                g_j                           = tf.exp(-1.0/ xi * (  tf.reshape( tf.math.maximum( v_diff_temp , 0.0000000001), [self.params["batch_size"], 1])  ))
                g_j_log                       = -1.0/ xi * (  tf.reshape( tf.math.maximum( v_diff_temp , 0.0000000001), [self.params["batch_size"], 1])  )
                
                g_js.append(g_j)
                g_j_logs.append(g_j_log)



        ## Compute h which is present in all models
        h                             =   - 1.0 /  xi  * (( dv_dY  - (self.params['gamma_1'] \
        + self.params['gamma_2'] * Y   )   ) *  self.params['varsigma'] * \
        self.params['eta'] * self.params['A_d'] * (1-R) * K   )  

        beta_f   = (self.params["beta_f"] + self.params["varsigma"] * h)    ## beta_f must be adjusted

        #################
        pv           = self.params["delta"] * v

        if self.params["n_dims"] == 4:
            ## Before tech jump, productivity is A_g and planner invests in R&D
            c        = (self.params["A_d"] - i_d) * (1 - R) + (self.params["A_g"] - i_g) * R - tf.exp(-i_I_capped)
        else:
            ## After tech jump, no R&D investment and productivity is A_g_prime
            
            # c        = (self.params["A_d"] - i_d) * (1 - R) + (self.params["A_g_prime"] - i_g) * R 
            c        = (self.params["A_d"] - i_d) * (1 - R) + (A_g_prime - i_g) * R 



        inside_log   =   tf.reshape( tf.math.maximum( c , 0.000000001), [self.params["batch_size"], 1])

        flow           = self.params["delta"] * (tf.math.log( inside_log )  +  logK )
        v_kk_term      = ( tf.pow(self.params["sigma_d"],2) * tf.pow(1-R,2) + tf.pow(self.params["sigma_g"],2) * tf.pow(R,2))/2.0

        inside_log_i_d   =   tf.reshape( tf.math.maximum( 1 + self.params["phi_d"] * i_d , 0.0001), [self.params["batch_size"], 1])
        inside_log_i_g   =   tf.reshape( tf.math.maximum( 1 + self.params["phi_g"] * i_g , 0.0001), [self.params["batch_size"], 1])

        v_k_term       = ( self.params["alpha_d"] + self.params["Gamma"] * tf.math.log( inside_log_i_d ) ) * (1 - R) + ( self.params["alpha_g"] +  self.params["Gamma"] * tf.math.log( inside_log_i_g) ) * R  - v_kk_term
        v_r_term       = ( self.params["alpha_g"] + self.params["Gamma"] * tf.math.log( inside_log_i_g )  - ( self.params["alpha_d"] + self.params["Gamma"] * tf.math.log( inside_log_i_d) ) + tf.pow(self.params["sigma_d"],2) *  (1-R ) - 
                        tf.pow(self.params["sigma_g"], 2) *  R ) * \
        R * (1 - R)
        v_rr_term      = 0.5 * tf.pow(R, 2) * tf.pow( 1- R, 2) *  ( tf.pow(self.params["sigma_g"],2) + tf.pow(self.params["sigma_d"], 2))

        v_logK_r_term  = -R * tf.pow(1-R, 2) * tf.pow(self.params["sigma_d"], 2) + tf.pow(R, 2) * (1.0 - R) * tf.pow(self.params["sigma_g"], 2)

        
        v_y_term       = beta_f * (self.params["eta"] *  self.params["A_d"] * (1-R) * K)
        v_yy_term      = 0.5 * tf.pow( self.params["varsigma"],2) * tf.pow(self.params["eta"] * self.params["A_d"] * (1-R) * K, 2)
        last_term      = -(( self.params["gamma_1"] +  self.params["gamma_2"] * Y  + gamma_3 * (Y - self.params["y_bar"])  ) * v_y_term + \
        (self.params["gamma_2"]+ gamma_3  ) * v_yy_term)

        if self.params["n_dims"] == 4:

            v_I_g_term     = - self.params["zeta"] + self.params["psi_0"] * tf.exp(-self.params["psi_1"] * i_I_capped)  * tf.exp( self.params["psi_1"] * (logK -  log_I_g) ) - 0.5 * tf.pow(self.params["sigma_I"], 2)

            v_I_g_I_g_term = 0.5 * tf.pow(self.params["sigma_I"], 2)
        

        rhs = flow + v_k_term * dv_dlogK + v_kk_term * dv_ddlogK + v_r_term * dv_dR + v_rr_term * dv_ddR \
        + v_y_term * dv_dY + v_logK_r_term * dv_dlogKdR + v_yy_term * dv_ddY + last_term 

        rhs = rhs + xi  * tf.pow(h,2) / 2  ## h appears in all models

        # Entropy computation
        ## post tech and pre damage model
        if "post_tech" in self.params["model_type"] and "pre_damage" in self.params["model_type"]:
            I_d   = self.params['r_1'] * ( tf.exp( self.params['r_2'] / 2 * tf.pow(Y - self.params['y_lower_bar'],2) ) - 1  ) * \
            tf.cast(Y > self.params['y_lower_bar'], tf.float32 )

            f_ms     = []
            f_m_logs = []
            v_m_vals = []

            for k in range(self.params["gamma_3_length"]):
                X_post_tech_post_damage                   = tf.concat([logK, R, tf.ones(tf.shape(Y)) * self.params["y_bar"], tf.ones(tf.shape(Y)) * self.params["gamma_3_list"][k], log_xi, A_g_prime], 1)

                v_m                    =  self.v_post_tech_post_damage_nn(X_post_tech_post_damage) 
                v_m_vals.append( v_m )

                f_m       = tf.exp(-1.0/ xi * (v_m - v))
                f_m_log   = -1.0/ xi * (v_m - v)

                f_ms.append(f_m)
                f_m_logs.append( f_m_log  )

                rhs = rhs + I_d *  (f_ms[k] * ( v_m_vals[k] - v ) + \
                xi * (1.0 - f_ms[k] + f_ms[k] * f_m_logs[k] )) / self.params['gamma_3_length']
                    

        if self.params["n_dims"] == 4:
            rhs = rhs + v_I_g_term * dv_dI_g + v_I_g_I_g_term * dv_ddI_g 

            if "pre_tech" in self.params["model_type"] and "pre_damage" in self.params["model_type"]:
                

                    for j in range(self.params["A_g_prime_length"]):

                        rhs = rhs + tf.exp(log_I_g) / self.params["varrho"] * g_js[j] * (v_j_vals[j] - v) +  \
                        xi * ( tf.exp(log_I_g) / self.params['varrho'] * (1.0 - g_js[j] + g_js[j] * g_j_logs[j]) )


                    for k in range(self.params['gamma_3_length']):

                        I_d   = self.params['r_1'] * ( tf.exp( self.params['r_2'] / 2 * tf.pow(Y - self.params['y_lower_bar'],2) ) - 1  ) * \
                        tf.cast(Y > self.params['y_lower_bar'], tf.float32 )
                        
                        rhs = rhs + I_d *  (f_ms[k] * ( v_m_vals[k] - v ) + \
                        xi * (1.0 - f_ms[k] + f_ms[k] * f_m_logs[k]  )) / self.params['gamma_3_length']
                    
                    
            elif "pre_tech" in self.params["model_type"] and "post_damage" in self.params["model_type"]:
                
                g_js     = []
                g_j_logs = []
                v_j_vals = []



                for j in range(self.params["A_g_prime_length"]):
                    X_post_tech_post_damage                   = tf.concat([logK, R, Y, gamma_3, log_xi, tf.ones(tf.shape(Y)) * self.params["A_g_prime_list"][j]], 1)
    
                    v_j                    =  self.v_post_tech_post_damage_nn(X_post_tech_post_damage) 
                    v_j_vals.append( v_j )

                    g_j       = tf.exp(-1.0/ xi * (v_j - v))
                    g_j_log   = -1.0/ xi * (v_j - v)

                    g_js.append(g_j)
                    g_j_logs.append( g_j_log  )

                    rhs = rhs + tf.exp(log_I_g) / self.params["varrho"] *  (g_js[j] * ( v_j_vals[j] - v ) + \
                    xi * (1.0 - g_js[j] + g_js[j] * g_j_logs[j] ))
                    
        ## FOCs

        marginal_util_c_over_k = self.params["delta"] / inside_log

        FOC_g   = - marginal_util_c_over_k  + self.params["Gamma"] / ( inside_log_i_g ) * self.params["phi_g"] * (dv_dlogK +  (1.0 - R) * dv_dR)
        FOC_d   = - marginal_util_c_over_k  + self.params["Gamma"] / ( inside_log_i_d ) * self.params["phi_d"] * (dv_dlogK - R * dv_dR)
        

        if self.params["n_dims"] == 4:
            FOC_I   = - self.params["delta"] / inside_log * tf.exp(-i_I_capped) + self.params["psi_0"] * self.params["psi_1"] * \
            tf.exp(-i_I_capped  * (self.params["psi_1"])) * tf.exp( self.params["psi_1"] * (logK -  log_I_g) )  * dv_dI_g 

            return rhs, pv, dv_dY, c, 1 + self.params["phi_g"] * i_g, 1 + self.params["phi_d"] * i_d, i_I, v_diff, dv_dI_g, marginal_util_c_over_k, FOC_g, FOC_d, FOC_I
        else:
            return rhs, pv, dv_dY, c, 1 + self.params["phi_g"] * i_g, 1 + self.params["phi_d"] * i_d, marginal_util_c_over_k, FOC_g, FOC_d

    @tf.function
    def objective_fn(self, logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g = None, compute_control = False, training = True):

        ## This is the objective function that stochastic gradient descend will try to minimize
        ## It depends on which NN it is training. Controls and value functions have different
        ## objectives.

        if self.params["n_dims"] == 4:
            rhs, pv, dv_dY, c, inside_log_i_g, inside_log_i_d, i_I, v_diff, dv_dI_g, marginal_utility_of_consumption_norm, FOC_g, FOC_d, FOC_I        = self.pde_rhs(logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g)
        else:
            rhs, pv, dv_dY, c, inside_log_i_g, inside_log_i_d, marginal_utility_of_consumption_norm, FOC_g, FOC_d                       = self.pde_rhs(logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g)

        epsilon = 10e-4
        negative_consumption_boolean = tf.reshape( tf.cast( c < 0.000000001, tf.float32 ),  [self.params["batch_size"], 1])
        loss_c  = - c  * negative_consumption_boolean + epsilon
        
        negative_inside_log_i_g_boolean = tf.reshape( tf.cast( inside_log_i_g < 0.000000001, tf.float32 ),  [self.params["batch_size"], 1])
        loss_inside_log_i_g             = - inside_log_i_g  * negative_inside_log_i_g_boolean + epsilon
        
        negative_inside_log_i_d_boolean = tf.reshape( tf.cast( inside_log_i_d < 0.000000001, tf.float32 ),  [self.params["batch_size"], 1])
        loss_inside_log_i_d             = - inside_log_i_d  * negative_inside_log_i_d_boolean + epsilon

        if self.params['n_dims'] == 4:
            ## i_I cannot be negative
            negative_i_I_boolean            = tf.reshape( tf.cast( i_I < 0.000000001, tf.float32 ),  [self.params["batch_size"], 1])
            loss_i_I                        = - i_I  * negative_i_I_boolean + epsilon
                
        if training:    
            ## Take care of nonsensical controls first

            loss_c_mse = tf.sqrt(tf.reduce_mean(tf.square(loss_c / self.marginal_utility_of_consumption_norm)))
                            
            loss_inside_log_i_g_mse = tf.sqrt(tf.reduce_mean(tf.square(loss_inside_log_i_g / self.marginal_utility_of_consumption_norm)))
            loss_inside_log_i_d_mse = tf.sqrt(tf.reduce_mean(tf.square(loss_inside_log_i_d / self.marginal_utility_of_consumption_norm)))

            control_constraints = tf.reduce_sum(negative_consumption_boolean) + tf.reduce_sum(negative_inside_log_i_g_boolean) + tf.reduce_sum(negative_inside_log_i_d_boolean)

            loss_constraints    = loss_c_mse + loss_inside_log_i_g_mse + loss_inside_log_i_d_mse  

            if self.params['n_dims'] == 4:

                control_constraints     = control_constraints + tf.reduce_sum(negative_i_I_boolean)
                loss_i_I_mse            = tf.sqrt(tf.reduce_mean(tf.square(loss_i_I / self.marginal_utility_of_consumption_norm)))
                loss_constraints        = loss_constraints + loss_i_I_mse
            
            if control_constraints > 0:
                return loss_constraints

            if compute_control:
                
                ## Optimizing all three together
                if self.params['n_dims'] == 4:
                    return -tf.reduce_mean( (rhs - pv ) / self.flow_pv_norm ) + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_g / self.marginal_utility_of_consumption_norm)))  + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_d / self.marginal_utility_of_consumption_norm))) + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_I / self.marginal_utility_of_consumption_norm))) 
                else:
                    return -tf.reduce_mean( (rhs - pv ) / self.flow_pv_norm ) + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_g / self.marginal_utility_of_consumption_norm)))  + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_d / self.marginal_utility_of_consumption_norm))) 
            else:

                ## loss associated with dv/dY > 0
                loss_dv_dY = dv_dY  * tf.reshape( tf.cast(Y > self.params['y_bar'], tf.float32 ),  [self.params["batch_size"], 1]) \
                    * tf.reshape( tf.cast( dv_dY > 0, tf.float32 ),  [self.params["batch_size"], 1]) + 10e-4


                if self.params['n_dims'] == 4:
                    loss_v_diff = -v_diff * tf.reshape( tf.cast( v_diff < 0.000000001, tf.float32 ),  [self.params["batch_size"], 1]) + 10e-4
    
                    loss_dv_dI_g = - dv_dI_g  * tf.reshape( tf.cast( dv_dI_g < 0.0, tf.float32 ),  [self.params["batch_size"], 1]) + 10e-4

                    loss = tf.sqrt(tf.reduce_mean(tf.square( (rhs - pv) / self.flow_pv_norm )))  + tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dY / self.marginal_utility_of_consumption_norm))) + tf.sqrt(tf.reduce_mean(tf.square(FOC_g / self.marginal_utility_of_consumption_norm))) + tf.sqrt(tf.reduce_mean(tf.square(FOC_d / self.marginal_utility_of_consumption_norm))) + tf.sqrt(tf.reduce_mean(tf.square(FOC_I / self.marginal_utility_of_consumption_norm))) + tf.sqrt(tf.reduce_mean(tf.square(loss_v_diff / self.marginal_utility_of_consumption_norm))) + tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dI_g / self.marginal_utility_of_consumption_norm)))

                else:
                    loss = tf.sqrt(tf.reduce_mean(tf.square( (rhs - pv) / self.flow_pv_norm )))  + tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dY / self.marginal_utility_of_consumption_norm))) + tf.sqrt(tf.reduce_mean(tf.square(FOC_g / self.marginal_utility_of_consumption_norm))) + tf.sqrt(tf.reduce_mean(tf.square(FOC_d / self.marginal_utility_of_consumption_norm))) 
                    
                return loss

        else:

            ## loss associated with dv/dY > 0
            loss_dv_dY = dv_dY * tf.reshape( tf.cast(Y > self.params['y_bar'], tf.float32 ),  [self.params["batch_size"], 1]) \
                * tf.reshape( tf.cast( dv_dY > 0.0, tf.float32 ),  [self.params["batch_size"], 1])  + 10e-4

            if self.params["n_dims"] == 4:

                ## loss associated with v_diff
                loss_v_diff = -v_diff  * tf.reshape( tf.cast( v_diff < 0, tf.float32 ),  [self.params["batch_size"], 1]) + 10e-4
                loss_dv_dI_g = - dv_dI_g   * tf.reshape( tf.cast( dv_dI_g < 0, tf.float32 ),  [self.params["batch_size"], 1]) + 10e-4

                return tf.sqrt(tf.reduce_mean(tf.square( (rhs - pv)  / self.flow_pv_norm  ))), -tf.reduce_mean(rhs  / self.flow_pv_norm ), tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dY / self.marginal_utility_of_consumption_norm))), tf.sqrt(tf.reduce_mean(tf.square(loss_c / self.marginal_utility_of_consumption_norm))), tf.sqrt(tf.reduce_mean(tf.square(loss_inside_log_i_g / self.marginal_utility_of_consumption_norm))), tf.sqrt(tf.reduce_mean(tf.square(loss_inside_log_i_d / self.marginal_utility_of_consumption_norm))),  tf.sqrt(tf.reduce_mean(tf.square(FOC_g / self.marginal_utility_of_consumption_norm))), \
                    tf.sqrt(tf.reduce_mean(tf.square(FOC_d / self.marginal_utility_of_consumption_norm))), tf.sqrt(tf.reduce_mean(tf.square(FOC_I / self.marginal_utility_of_consumption_norm))), tf.sqrt(tf.reduce_mean(tf.square(loss_i_I / self.marginal_utility_of_consumption_norm))), tf.sqrt(tf.reduce_mean(tf.square(loss_v_diff / self.marginal_utility_of_consumption_norm))), tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dI_g / self.marginal_utility_of_consumption_norm)))
            else:
                return tf.sqrt(tf.reduce_mean(tf.square((rhs - pv)  / self.flow_pv_norm ))), -tf.reduce_mean(rhs  / self.flow_pv_norm ), tf.sqrt(tf.reduce_mean(tf.square(loss_dv_dY / self.marginal_utility_of_consumption_norm))), tf.sqrt(tf.reduce_mean(tf.square(loss_c / self.marginal_utility_of_consumption_norm))), tf.sqrt(tf.reduce_mean(tf.square(loss_inside_log_i_g / self.marginal_utility_of_consumption_norm))), tf.sqrt(tf.reduce_mean(tf.square(loss_inside_log_i_d / self.marginal_utility_of_consumption_norm))), tf.sqrt(tf.reduce_mean(tf.square(FOC_g / self.marginal_utility_of_consumption_norm))), \
                    tf.sqrt(tf.reduce_mean(tf.square(FOC_d / self.marginal_utility_of_consumption_norm))) 

    def grad(self, logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g = None, compute_control = False, training = True):

        if compute_control:
            with tf.GradientTape(persistent=True) as tape:
                objective = self.objective_fn(logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g, compute_control, training)

            if self.params['n_dims'] == 4:
                trainable_variables = self.i_g_nn.trainable_variables + self.i_d_nn.trainable_variables + self.i_I_nn.trainable_variables
            else:
                trainable_variables = self.i_g_nn.trainable_variables + self.i_d_nn.trainable_variables 

            grad = tape.gradient(objective, trainable_variables)

            del tape

            return grad
        else:
            with tf.GradientTape(persistent=True) as tape:
                objective = self.objective_fn(logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g, compute_control, training)
            grad = tape.gradient(objective, self.v_nn.trainable_variables)
            del tape

            return grad 

    @tf.function
    def train_step(self):
        if "pre_tech" in self.params["model_type"]:
            logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g = self.sample()
        else:
            logK, R, Y, gamma_3, A_g_prime, log_xi = self.sample()
            log_I_g = None 

        ## First, train value function
        
        grad = self.grad(logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g, compute_control= False, training=True)
        self.params["optimizers"][0].apply_gradients(zip(grad, self.v_nn.trainable_variables))

        ## Second, train controls
        grad = self.grad(logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g, compute_control= True, training=True)

        if self.params['n_dims'] == 4:
            self.params["optimizers"][1].apply_gradients(zip(grad, self.i_g_nn.trainable_variables + self.i_d_nn.trainable_variables + self.i_I_nn.trainable_variables ))
        else:
            self.params["optimizers"][1].apply_gradients(zip(grad, self.i_g_nn.trainable_variables + self.i_d_nn.trainable_variables ))

    def train(self):

        start_time = time.time()
        training_history = []

        # Prepare to store best neural networks and initialize networks
        min_loss = float("inf")
        
        if "post_damage" in self.params["model_type"] and "post_tech" in self.params["model_type"]:
            n_inputs = 6
    
        if "post_damage" in self.params["model_type"] and "pre_tech" in self.params["model_type"]:
            n_inputs = 6
        
        if "pre_damage" in self.params["model_type"] and "post_tech" in self.params["model_type"]:
            n_inputs = 5

        if "pre_damage" in self.params["model_type"] and "pre_tech" in self.params["model_type"]:
            n_inputs = 5

        best_v_nn    = FeedForwardSubNet(self.params['v_nn_config'])
        best_v_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.v_nn.build( (self.params["batch_size"], n_inputs) )

        best_i_g_nn  = FeedForwardSubNet(self.params['i_g_nn_config'])
        best_i_g_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.i_g_nn.build( (self.params["batch_size"], n_inputs) )

        best_i_d_nn  = FeedForwardSubNet(self.params['i_d_nn_config'])
        best_i_d_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.i_d_nn.build( (self.params["batch_size"], n_inputs) )

        if "pre_tech" in self.params["model_type"]:
            best_i_I_nn  = FeedForwardSubNet(self.params['i_I_nn_config'])
            best_i_I_nn.build( (self.params["batch_size"], n_inputs) ) 
            self.i_I_nn.build( (self.params["batch_size"], n_inputs) )

        best_v_nn.set_weights(self.v_nn.get_weights())
        best_i_g_nn.set_weights(self.i_g_nn.get_weights())
        best_i_d_nn.set_weights(self.i_d_nn.get_weights())

        if "pre_tech" in self.params["model_type"]:
            best_i_I_nn.set_weights(self.i_I_nn.get_weights())
        
        ## Load pretrained weights
        if self.params['pretrained_path'] is not None:
            if "post_tech" in self.params["model_type"] and "post_damage" in self.params["model_type"]:
                print("Loading pretrained model for post-tech post-damage...")
                self.v_nn.load_weights( self.params["pretrained_path"]  + '/v_nn_checkpoint_post_damage_post_tech')
                self.i_g_nn.load_weights( self.params["pretrained_path"]  + '/i_g_nn_checkpoint_post_damage_post_tech')
                self.i_d_nn.load_weights( self.params["pretrained_path"]  + '/i_d_nn_checkpoint_post_damage_post_tech')
                if self.params["n_dims"] == 4:
                    self.i_I_nn.load_weights( self.params["pretrained_path"]  + '/i_I_nn_checkpoint_post_damage_post_tech')

            if "pre_tech" in self.params["model_type"] and "post_damage" in self.params["model_type"]:
                print("Loading pretrained model for pre-tech post-damage...")
                self.v_nn.load_weights( self.params["pretrained_path"]  + '/v_nn_checkpoint_pre_tech_post_damage')
                self.i_g_nn.load_weights( self.params["pretrained_path"]  + '/i_g_nn_checkpoint_pre_tech_post_damage')
                self.i_d_nn.load_weights( self.params["pretrained_path"]  + '/i_d_nn_checkpoint_pre_tech_post_damage')
                if self.params["n_dims"] == 4:
                    self.i_I_nn.load_weights( self.params["pretrained_path"]  + '/i_I_nn_checkpoint_pre_tech_post_damage')

            if "post_tech" in self.params["model_type"] and "pre_damage" in self.params["model_type"]:
                print("Loading pretrained model for post-tech pre-damage...")
                self.v_nn.load_weights( self.params["pretrained_path"]  + '/v_nn_checkpoint_pre_damage_post_tech')
                self.i_g_nn.load_weights( self.params["pretrained_path"]  + '/i_g_nn_checkpoint_pre_damage_post_tech')
                self.i_d_nn.load_weights( self.params["pretrained_path"]  + '/i_d_nn_checkpoint_pre_damage_post_tech')
                if self.params["n_dims"] == 4:
                    self.i_I_nn.load_weights( self.params["pretrained_path"]  + '/i_I_nn_checkpoint_pre_damage_post_tech')

            if "pre_tech" in self.params["model_type"] and "pre_damage" in self.params["model_type"]:
                print("Loading pretrained model for pre-tech pre-damage...")
                self.v_nn.load_weights( self.params["pretrained_path"]  + '/v_nn_checkpoint_pre_tech_pre_damage')
                self.i_g_nn.load_weights( self.params["pretrained_path"]  + '/i_g_nn_checkpoint_pre_tech_pre_damage')
                self.i_d_nn.load_weights( self.params["pretrained_path"]  + '/i_d_nn_checkpoint_pre_tech_pre_damage')
                if self.params["n_dims"] == 4:
                    self.i_I_nn.load_weights( self.params["pretrained_path"]  + '/i_I_nn_checkpoint_pre_tech_pre_damage')


        # begin sgd iteration
        for step in range(self.params["num_iterations"]):
            if step % self.params["logging_frequency"] == 0:

                ## Sample test data
                if "pre_tech" in self.params["model_type"]:
                    logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g = self.sample()
                else:
                    logK, R, Y, gamma_3, A_g_prime, log_xi = self.sample() 
                    log_I_g = None 

                ## Compute test loss
                test_losses = self.objective_fn(logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g, training = False)

                ## Update normalization constants


                if self.params["n_dims"] == 4:
                    rhs, pv, dv_dY, c, inside_log_i_g, inside_log_i_d, i_I, v_diff, dv_dI_g, marginal_utility_of_consumption_norm, FOC_g, FOC_d, FOC_I        = self.pde_rhs(logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g)
                else:
                    rhs, pv, dv_dY, c, inside_log_i_g, inside_log_i_d, marginal_utility_of_consumption_norm, FOC_g, FOC_d                                = self.pde_rhs(logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g)

                
                self.flow_pv_norm = (1.0 - self.params['norm_weight']) * self.flow_pv_norm + self.params['norm_weight'] * pv 
                self.marginal_utility_of_consumption_norm = (1.0 - self.params['norm_weight']) * self.marginal_utility_of_consumption_norm + self.params['norm_weight'] * marginal_utility_of_consumption_norm

                ## Store best neural networks

                if (test_losses[0] < min_loss):
                    min_loss = test_losses[0]

                    best_v_nn.set_weights(self.v_nn.get_weights())
                    best_i_g_nn.set_weights(self.i_g_nn.get_weights())
                    best_i_d_nn.set_weights(self.i_d_nn.get_weights())

                    if self.params["n_dims"] == 4:
                        best_i_I_nn.set_weights(self.i_I_nn.get_weights())

                ## Generate checkpoints for tensorboard
                if self.params['tensorboard']:
                    grad_v_nn     = self.grad(logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g, compute_control= False, training=True)
                    grad_controls = self.grad(logK, R, Y, gamma_3, A_g_prime, log_xi, log_I_g, compute_control= True, training=True)

                    with self.test_writer.as_default():

                        ## Export learning rates
                        for optimizer_idx in range(len(self.params['optimizers'])):
                            if "sgd" in self.params['learning_rate_schedule_type']:
                                tf.summary.scalar('learning_rate_' + str(optimizer_idx), self.params["optimizers"][optimizer_idx]._decayed_lr(tf.float32), step = step)
                            else:
                                tf.summary.scalar('learning_rate_' + str(optimizer_idx), self.params["optimizers"][optimizer_idx].lr, step = step)

                        ## Export losses
                        tf.summary.scalar('loss_v', test_losses[0], step = step)
                        tf.summary.scalar('loss_negative_mean_rhs', test_losses[1], step = step)
                        tf.summary.scalar('loss_dv_dY', test_losses[2], step = step)
                        tf.summary.scalar('loss_c', test_losses[3], step = step)
                        tf.summary.scalar('loss_inside_log_i_g', test_losses[4], step = step)
                        tf.summary.scalar('loss_inside_log_i_g', test_losses[5], step = step)

                        tf.summary.scalar('loss_FOC_g', test_losses[6], step = step)
                        tf.summary.scalar('loss_FOC_d', test_losses[7], step = step)

                        tf.summary.scalar('pv_norm', tf.reduce_mean(self.flow_pv_norm), step = step)
                        tf.summary.scalar('marginal_util_consumption_norm', 
                                              tf.reduce_mean(self.marginal_utility_of_consumption_norm), step = step)
                        
                        if self.params['n_dims'] == 4:
                            tf.summary.scalar('loss_FOC_I', test_losses[8], step = step)
                            tf.summary.scalar('loss_i_I', test_losses[9], step = step)
                            tf.summary.scalar('loss_v_diff', test_losses[10], step = step)
                            tf.summary.scalar('loss_dv_dI_g', test_losses[11], step = step)

                        ## Export weights and gradients

                        for layer in self.v_nn.layers:
                                                        
                            for W in layer.weights:
                                tf.summary.histogram(W.name + '_weights', W, step = step)

                        for g in range(len(self.v_nn.trainable_variables)):
                            tf.summary.histogram( self.v_nn.trainable_variables[g].name + '_grads', grad_v_nn[g], step = step )

                        for layer in self.i_g_nn.layers:
                            for W in layer.weights:
                                tf.summary.histogram(W.name + '_weights', W, step = step)

                        for g in range(len(self.i_g_nn.trainable_variables)):
                            tf.summary.histogram( self.i_g_nn.trainable_variables[g].name + '_grads', grad_controls[g], step = step )

                        for layer in self.i_d_nn.layers:
                            for W in layer.weights:
                                tf.summary.histogram(W.name + '_weights', W, step = step)

                        for g in range(len(self.i_d_nn.trainable_variables)):
                            tf.summary.histogram( self.i_d_nn.trainable_variables[g].name + '_grads', 
                                                 grad_controls[len(self.i_g_nn.trainable_variables) + g], step = step )

                        if self.params['n_dims'] == 4:             

                            for layer in self.i_I_nn.layers:
                                for W in layer.weights:
                                    tf.summary.histogram(W.name + '_weights', W, step = step)

                            for g in range(len(self.i_I_nn.trainable_variables)):
                                tf.summary.histogram( self.i_I_nn.trainable_variables[g].name + '_grads', 
                                                    grad_controls[len(self.i_d_nn.trainable_variables) + len(self.i_g_nn.trainable_variables) + g], step = step )

                elapsed_time = time.time() - start_time

                ## Appendinging to training history
                entry = [step] + list(test_losses) + [ tf.reduce_mean(self.flow_pv_norm), tf.reduce_mean(self.marginal_utility_of_consumption_norm), elapsed_time]

                training_history.append(entry)

            self.train_step()

        ## Use best neural networks 
        self.v_nn.set_weights(best_v_nn.get_weights())
        self.i_g_nn.set_weights(best_i_g_nn.get_weights())
        self.i_d_nn.set_weights(best_i_d_nn.get_weights())
        if self.params["n_dims"] == 4:
            self.i_I_nn.set_weights(best_i_I_nn.get_weights())
        

        ## Export last check point
        self.v_nn.save_weights( self.params["export_folder"] + '/v_nn_checkpoint_' + self.params["model_type"])
        self.i_g_nn.save_weights( self.params["export_folder"] + '/i_g_nn_checkpoint_' + self.params["model_type"])
        self.i_d_nn.save_weights( self.params["export_folder"] + '/i_d_nn_checkpoint_' + self.params["model_type"])
        if self.params["n_dims"] == 4:
            self.i_I_nn.save_weights( self.params["export_folder"] + '/i_I_nn_checkpoint_' + self.params["model_type"])


        ## Save training history
        if self.params["n_dims"] == 4:
            header = 'step,loss_v,loss_negative_mean_rhs,loss_dv_dY,loss_c,loss_inside_log_i_g,loss_inside_log_i_d,loss_FOC_g,loss_FOC_d,loss_FOC_I,loss_i_I,loss_v_diff,loss_dv_dI_g,pv_norm,marginal_util_consumption_norm,elapsed_time'
        else:
            header = 'step,loss_v,loss_negative_mean_rhs,loss_dv_dY,loss_c,loss_inside_log_i_g,loss_inside_log_i_d,loss_FOC_g,loss_FOC_d,pv_norm,marginal_util_consumption_norm,elapsed_time'

        np.savetxt(  self.params["export_folder"] + '/training_history.csv',
            training_history,
            fmt= ['%d'] + ['%.5e'] * len(test_losses) + ['%.5e','%.5e','%d'], # ['%d', '%.5e', '%.5e', '%.5e', '%d'],
            delimiter=",",
            header=header,
            comments='')

        ## Plot losses

        loss_v_history                   = [history_record[1] for history_record in training_history]
        loss_negative_mean_rhs_history   = [history_record[2] for history_record in training_history]
        loss_dv_dY_history               = [history_record[3] for history_record in training_history]
        loss_c_history                   = [history_record[4] for history_record in training_history]
        loss_inside_log_i_g_history      = [history_record[5] for history_record in training_history]
        loss_inside_log_i_d_history      = [history_record[6] for history_record in training_history]
        loss_FOC_g_history               = [history_record[7] for history_record in training_history]
        loss_FOC_d_history               = [history_record[8] for history_record in training_history]

        if self.params['n_dims'] == 4:
            loss_FOC_I_history                                    = [history_record[9] for history_record in training_history]
            loss_i_I_history                                      = [history_record[10] for history_record in training_history]
            loss_v_diff_history                                   = [history_record[11] for history_record in training_history]
            loss_dv_dI_g_history                                  = [history_record[12] for history_record in training_history]
            pv_norm_history                                       = [history_record[13] for history_record in training_history]
            marginal_util_consumption_norm_history                = [history_record[14] for history_record in training_history]
        else:
            pv_norm_history                                       = [history_record[9] for history_record in training_history]
            marginal_util_consumption_norm_history                = [history_record[10] for history_record in training_history]

        plt.figure()
        plt.title("Test loss: value function")
        plt.plot(loss_v_history)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_v_history.png")

        plt.figure()
        plt.title("Test loss: controls")
        plt.plot(loss_negative_mean_rhs_history)
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_controls_v.png")

        plt.figure()
        plt.title("Test loss: dvdY")
        plt.plot(loss_dv_dY_history)
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_controls_dv_dY.png")

        plt.figure()
        plt.title("Test loss: c")
        plt.plot(loss_c_history)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_controls_c.png")

        plt.figure()
        plt.title("Test loss: inside_log_i_g")
        plt.plot(loss_inside_log_i_g_history)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_controls_inside_log_i_g.png")

        plt.figure()
        plt.title("Test loss: inside_log_i_d")
        plt.plot(loss_inside_log_i_d_history)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_controls_inside_log_i_d.png")

        plt.figure()
        plt.title("Test loss: FOC_g")
        plt.plot(loss_FOC_g_history)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_FOC_g_history.png")

        plt.figure()
        plt.title("Test loss: FOC_d")
        plt.plot(loss_FOC_d_history)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_FOC_d_history.png")

        plt.figure()
        plt.title("pv_norm")
        plt.plot(pv_norm_history)
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/pv_norm_history.png")

        plt.figure()
        plt.title("marginal_util_consumption_norm")
        plt.plot(marginal_util_consumption_norm_history)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/marginal_util_consumption_norm_history.png")

        if self.params['n_dims'] == 4:
            plt.figure()
            plt.title("Test loss: FOC_I")
            plt.plot(loss_FOC_I_history)
            plt.yscale('log')
            plt.xscale('log')
            plt.savefig( self.params["export_folder"] + "/loss_FOC_I_history.png")

            plt.figure()
            plt.title("Test loss: i_I")
            plt.plot(loss_i_I_history)
            plt.yscale('log')
            plt.xscale('log')
            plt.savefig( self.params["export_folder"] + "/loss_i_I_history.png")

            plt.figure()
            plt.title("Test loss: v_diff")
            plt.plot(loss_v_diff_history)
            plt.yscale('log')
            plt.xscale('log')
            plt.savefig( self.params["export_folder"] + "/loss_v_diff_history.png")

            plt.figure()
            plt.title("Test loss: dvdI_g")
            plt.plot(loss_dv_dI_g_history)
            plt.xscale('log')
            plt.savefig( self.params["export_folder"] + "/loss_dv_dI_g_history.png")
        return np.array(training_history)

    def export_parameters(self):

        ## Export parameters

        with open(self.params["export_folder"] + '/params.txt', 'a') as the_file:
            for key in self.params.keys():
                if "nn_config" not in key:
                    the_file.write( str(key) + ": " + str(self.params[key]) + '\n')
        nn_config_keys = [x for x in self.params.keys() if "nn_config" in x]

        for nn_config_key in nn_config_keys:
            with open(self.params["export_folder"] + '/params_' + nn_config_key + '.txt', 'a') as the_file:
                for key in self.params[nn_config_key].keys():
                    the_file.write( str(key) + ": " + str(self.params[nn_config_key][key]) + '\n')

                
    def analyze(self):

        ## Analyze results
        n_points = 100

        if "post_damage" in self.params["model_type"] and "post_tech" in self.params["model_type"]:
            Y_vector       = np.unique(self.solution_fd['stateSpace'][:,2])
            point_Y        = Y_vector[np.abs(Y_vector - 2.5).argmin()]
            mid_point_logK = np.unique(self.solution_fd['stateSpace'][:,0])[ round(self.solution_fd['nK']/2)]

            idx            = (self.solution_fd['stateSpace'][:,2] == point_Y) & (self.solution_fd['stateSpace'][:,0] == mid_point_logK)
            X              = self.solution_fd['stateSpace'][idx]
            log_xi             = tf.ones( (X.shape[0],1) ) * self.params["log_xi"]

            X              = tf.cast(X ,dtype= "float32")
            gamma_3        = tf.ones( (X.shape[0],1) ) * self.params["gamma_3"]
            A_g_prime      = tf.ones( (X.shape[0],1) ) * self.params["A_d"]
            
            X              = tf.concat([X, gamma_3, A_g_prime, log_xi], axis=1)
            v = self.v_nn(X); i_g = self.i_g_nn(X); 
            i_d = self.i_d_nn(X)


            ## Generate plots
            f, ax = plt.subplots(1,3, figsize=(20,5))

            ax[0].plot(X[:,1], v, label = "Neural network")
            if self.params["n_dims"] < 4:
                ax[0].plot(X[:,1], self.solution_fd['V'].flatten(order = 'F')[idx], label = "Finite difference")
            ax[0].set_xlabel(r'$R$')
            ax[0].set_title(r'$v$')
            ax[0].legend()

            ax[1].plot(X[:,1], i_g)
            if self.params["n_dims"] < 4:
                ax[1].plot(X[:,1], self.solution_fd['i_G'].flatten(order = 'F')[idx])
            ax[1].set_xlabel(r'$R$')
            ax[1].set_title(r'$i_g$')

            ax[2].plot(X[:,1], i_d)
            if self.params["n_dims"] < 4:
                ax[2].plot(X[:,1], self.solution_fd['i_B'].flatten(order = 'F')[idx])
            ax[2].set_xlabel(r'$R$')
            ax[2].set_title(r'$i_d$')

            plt.savefig(self.params["export_folder"] + "/compare_results_fd.png")

            ## Vary gamma_3
            for gamma_3_idx in range(self.params["gamma_3_length"]):

                idx            = (self.solution_fd['stateSpace'][:,2] == point_Y) & (self.solution_fd['stateSpace'][:,0] == mid_point_logK)
                X              = self.solution_fd['stateSpace'][idx]
                X              = tf.cast(X ,dtype= "float32")
                gamma_3        = tf.ones( (X.shape[0],1) ) * self.params["gamma_3_list"][gamma_3_idx]
                A_g_prime      = tf.ones( (X.shape[0],1) ) * self.params["A_d"]

                X              = tf.concat([X, gamma_3, A_g_prime, log_xi], axis=1)
                v = self.v_nn(X); i_g = self.i_g_nn(X); 
                i_d = self.i_d_nn(X)


                ## Generate plots
                f, ax = plt.subplots(1,3, figsize=(20,5))

                ax[0].plot(X[:,1], v, label = "Neural network; gamma_3 = " + str( round(self.params["gamma_3_list"][gamma_3_idx],2) ))
                if self.params["n_dims"] < 4:
                    ax[0].plot(X[:,1], self.solution_fd['V'].flatten(order = 'F')[idx], label = "Finite difference")
                ax[0].set_xlabel(r'$R$')
                ax[0].set_title(r'$v$')
                ax[0].legend()

                ax[1].plot(X[:,1], i_g)
                if self.params["n_dims"] < 4:
                    ax[1].plot(X[:,1], self.solution_fd['i_G'].flatten(order = 'F')[idx])
                ax[1].set_xlabel(r'$R$')
                ax[1].set_title(r'$i_g$')

                ax[2].plot(X[:,1], i_d)
                if self.params["n_dims"] < 4:
                    ax[2].plot(X[:,1], self.solution_fd['i_B'].flatten(order = 'F')[idx])
                ax[2].set_xlabel(r'$R$')
                ax[2].set_title(r'$i_d$')

                plt.savefig(self.params["export_folder"] + "/compare_results_" + str(gamma_3_idx) + ".png")

            ## Vary xi
            log_xi_list                                             = [float(np.log(xi)) for xi in np.linspace(np.exp(self.params['log_xi_min']) + 0.02, 
                                                                                                               np.exp(self.params['log_xi_max']) - 0.02, 5)]
            ## Generate plots
            f, ax = plt.subplots(1,3, figsize=(20,5))

            for log_xi_idx in range(len(log_xi_list)):

                idx            = (self.solution_fd['stateSpace'][:,2] == point_Y) & (self.solution_fd['stateSpace'][:,0] == mid_point_logK)
                X              = self.solution_fd['stateSpace'][idx]
                X              = tf.cast(X ,dtype= "float32")
                gamma_3        = tf.ones( (X.shape[0],1) ) * self.params["gamma_3"]
                A_g_prime      = tf.ones( (X.shape[0],1) ) * self.params["A_d"]

                log_xi         = tf.ones( (X.shape[0],1) ) * log_xi_list[log_xi_idx]
                X              = tf.concat([X, gamma_3, A_g_prime, log_xi], axis=1)
                
                v   = self.v_nn(X); i_g = self.i_g_nn(X); 
                i_d = self.i_d_nn(X)




                ax[0].plot(X[:,1], v, label = "Neural network; xi = " + str( round( np.exp( log_xi_list[log_xi_idx] ),2) ))
                ax[0].set_xlabel(r'$R$')
                ax[0].set_title(r'$v$')
                ax[0].legend()

                ax[1].plot(X[:,1], i_g)
                ax[1].set_xlabel(r'$R$')
                ax[1].set_title(r'$i_g$')

                ax[2].plot(X[:,1], i_d)
                ax[2].set_xlabel(r'$R$')
                ax[2].set_title(r'$i_d$')

                
            plt.savefig(self.params["export_folder"] + "/compare_results_xi.png")
   
        if "pre_damage" in self.params["model_type"] and "pre_tech" in self.params["model_type"]:

            logK      = (self.params["logK_max"] + self.params["logK_min"]) / 2 * np.ones(n_points).reshape(n_points, 1)
            Y         = 2.5 * np.ones(n_points).reshape(n_points, 1)
            log_I_g   = (self.params["log_I_g_max"] + self.params["log_I_g_min"]) / 2 * np.ones(n_points).reshape(n_points, 1)

            R         = np.linspace(self.params["R_min"], self.params["R_max"], n_points)


            logK = tf.reshape(tf.cast(logK ,dtype= "float32"), [n_points,1]) 
            Y    = tf.reshape(tf.cast(Y ,dtype= "float32"), [n_points,1]) 
            log_I_g    = tf.reshape(tf.cast(log_I_g ,dtype= "float32"), [n_points,1]) 

            R   = tf.reshape(tf.cast(R ,dtype= "float32"), [n_points,1]) 

            log_xi             = tf.ones( (n_points, 1) ) * self.params["log_xi"]
            X = tf.concat([logK, R, Y, log_xi, log_I_g], 1)

            v = self.v_nn(X); i_g = self.i_g_nn(X); 
            i_d = self.i_d_nn(X); i_I = self.i_I_nn(X)
            

            f, ax = plt.subplots(1,4, figsize=(20,5))

            ax[0].plot(X[:,1], v, label = "Neural network")
            ax[0].set_xlabel(r'$R$')
            ax[0].set_title(r'$v$')
            ax[0].legend()

            ax[1].plot(X[:,1], i_g)
            ax[1].set_xlabel(r'$R$')
            ax[1].set_title(r'$i_g$')

            ax[2].plot(X[:,1], i_d)
            ax[2].set_xlabel(r'$R$')
            ax[2].set_title(r'$i_d$')

            ax[3].plot(X[:,1], tf.exp(-i_I))
            ax[3].set_xlabel(r'$R$')
            ax[3].set_title(r'$i_I$')

            plt.savefig(self.params["export_folder"] + "/compare_results.png")

    def simulate_path(self, T, dt, log_xi, export_folder):

        ## Create folder
        pathlib.Path(export_folder).mkdir(parents=True, exist_ok=True) 
        
        ## Initial state 
        init_logK     = tf.math.log(739.0)
        init_R        = 0.5  
        init_I_g      = tf.math.log(11.2)
        init_Y        = 1.1

        state         = tf.convert_to_tensor( [[ init_logK,  init_R, init_Y, log_xi,  init_I_g]] )
        state         = tf.reshape(state, (1,5))

        # state_pre     = tf.convert_to_tensor( [[ init_logK,  init_R, init_Y, log_xi, A_g_prime]] )
        # state_pre     = tf.reshape(state_pre, (1,5)) 

        state_list       = [state]
        # state_pre_list   = [state_pre]

        i_g_list      = [self.i_g_nn(state)]
        i_d_list      = [self.i_d_nn(state)]
        i_I_list      = [self.i_I_nn(state)]

        # v_post_tech_pre_damage        = self.v_post_tech_pre_damage_nn(state_pre)
        v                             = self.v_nn(state)

        f_ms = []
        v_m_vals = []

        g_js = []
        v_j_vals = []

        for k in range(self.params["gamma_3_length"]):

            state_pre_tech_post_damage    = tf.convert_to_tensor( [[ init_logK,  init_R, init_Y, self.params["gamma_3_list"][k], log_xi, init_I_g]] )
            state_pre_tech_post_damage        = tf.reshape(state_pre_tech_post_damage, (1,6))
            v_m                           = self.v_pre_tech_post_damage_nn(state_pre_tech_post_damage)
            v_m_vals.append( v_m )
            f_m       = tf.exp(-1.0/ np.exp(log_xi) * (v_m - v))
            f_ms.append(f_m)

        f_ms_list         = [f_ms]

        for j in range(self.params["A_g_prime_length"]):
            
            state_post_tech_pre_damage    = tf.convert_to_tensor( [[ init_logK,  init_R, init_Y, log_xi, self.params["A_g_prime_list"][j]]] )
            state_post_tech_pre_damage        = tf.reshape(state_post_tech_pre_damage, (1,5))
            
            v_j                           = self.v_post_tech_pre_damage_nn(state_post_tech_pre_damage)
            v_j_vals.append( v_j )

            g_j      = tf.exp(-1.0/  np.exp(log_xi) * (v_j - v))
            g_js.append(g_j)


        g_js_list            = [g_js]


        for t in range(T):


            ## Find investments
            i_g                         = self.i_g_nn(state_list[t])
            i_d                         = self.i_d_nn(state_list[t])
            i_I                         = self.i_I_nn(state_list[t])
            v                           = self.v_nn(state_list[t])

            f_ms              = []

            for k in range(self.params["gamma_3_length"]):

                state_pre_tech_post_damage    = tf.convert_to_tensor( [[ state_list[t][0,0], state_list[t][0,1], state_list[t][0,2],
                    self.params["gamma_3_list"][k],  state_list[t][0,3], state_list[t][0,4]]] )
                state_pre_tech_post_damage        = tf.reshape(state_pre_tech_post_damage, (1,6))
                
                v_m                           = self.v_pre_tech_post_damage_nn(state_pre_tech_post_damage)
                f_m       = tf.exp(-1.0/  np.exp(log_xi) * (v_m - v))
                f_ms.append(f_m)

            f_ms_list.append(f_ms)

            g_js              = []

            for j in range(self.params["A_g_prime_length"]):

                state_post_tech_pre_damage    = tf.convert_to_tensor( [[ state_list[t][0,0], state_list[t][0,1], state_list[t][0,2],
                    state_list[t][0,3], self.params["A_g_prime_list"][j]]] )
                state_pre_tech_post_damage        = tf.reshape(state_pre_tech_post_damage, (1,5))
                
                v_j                           = self.v_pre_tech_post_damage_nn(state_pre_tech_post_damage)
                g_j       = tf.exp(-1.0/  np.exp(log_xi) * (v_j - v))
                g_js.append(g_j)

            g_js_list.append(g_js)



            logK      = state_list[t][0,0]; R = state_list[t][0,1]; 
            Y         = state_list[t][0,2]; log_I_g = state_list[t][0,4]
            K         = tf.exp(logK)

            # g         = tf.exp(-1.0/  np.exp(log_xi) * (v_post_tech_pre_damage  - v))
            # g_list.append(g)


            i_g_list.append(i_g)
            i_d_list.append(i_d)
            i_I_list.append(i_I)

            ## Transition
            v_kk_term      = ( tf.pow(self.params["sigma_d"],2) * tf.pow(1-R,2) + tf.pow(self.params["sigma_g"],2) * tf.pow(R,2))/2.0

            inside_log_i_d   =     tf.math.maximum( 1 + self.params["phi_d"] * i_d , 0.0001) 
            inside_log_i_g   =     tf.math.maximum( 1 + self.params["phi_g"] * i_g , 0.0001)

            v_k_term       = ( self.params["alpha_d"] + self.params["Gamma"] * tf.math.log( inside_log_i_d ) ) * (1 - R) + ( self.params["alpha_g"] +  self.params["Gamma"] * tf.math.log( inside_log_i_g) ) * R  - v_kk_term
            v_r_term       = ( self.params["alpha_g"] + self.params["Gamma"] * tf.math.log( inside_log_i_g )  - ( self.params["alpha_d"] + self.params["Gamma"] * tf.math.log( inside_log_i_d) ) + tf.pow(self.params["sigma_d"],2) *  (1-R ) - 
                            tf.pow(self.params["sigma_g"], 2) *  R ) * \
            R * (1 - R)
            
            v_I_g_term     = - self.params["zeta"] + self.params["psi_0"] * tf.exp(-i_I * self.params["psi_1"]) * tf.exp( self.params["psi_1"] * (logK -  log_I_g) ) - 0.5 * tf.pow(self.params["sigma_I"], 2)
    
            new_logK       = logK + v_k_term * dt 
            new_R          = R + v_r_term * dt  
            new_log_I_g    = log_I_g + v_I_g_term * dt  
            new_Y          = Y + self.params["beta_f"] * ( self.params["eta"] * self.params["A_d"] * (1-R) * tf.exp( logK )) * dt 

            state          = tf.concat([new_logK, new_R, tf.reshape(new_Y, [1,1]), tf.reshape(log_xi, [1,1]), new_log_I_g ], axis=1)
            state_list.append(state)

            # state_pre          = tf.concat([new_logK, new_R, tf.reshape(new_Y, [1,1]),  tf.reshape(log_xi, [1,1])], axis=1)
            # state_pre_list.append(state_pre)

        state_matrix = tf.concat(state_list, axis = 0)

        with tf.GradientTape() as tape:
            tape.watch(state_matrix)
            output_array = self.v_nn(state_matrix)
        derivative = tape.gradient(output_array, state_matrix) 
        dv_dY      = derivative[:,2] 


        ################################################################
        ################################################################
        ################################################################
        ## Make plots
        ################################################################
        ################################################################
        ################################################################


        plt.figure()
        plt.plot([state.numpy()[0,0] for state in state_list])
        plt.xlabel("month")
        plt.title(r'$\log K$')
        plt.savefig(export_folder + "/logK_simulation.png")
        np.savetxt(export_folder + "/log_K_simulation.txt", np.array([state.numpy()[0,0] for state in state_list]))


        plt.figure()
        plt.plot([state.numpy()[0,1] for state in state_list])
        plt.xlabel("month")
        plt.title(r'$R$')
        plt.savefig(export_folder +  "/R_simulation.png")
        np.savetxt(export_folder + "/R_simulation.txt", np.array([state.numpy()[0,1] for state in state_list]))


        plt.figure()
        plt.plot([state.numpy()[0,4] for state in state_list])
        plt.xlabel("month")
        plt.title(r'$\log I_g$')
        plt.savefig(export_folder + "/log_I_g_simulation.png")
        np.savetxt(export_folder +  "/log_I_g_simulation.txt", np.array([state.numpy()[0,4] for state in state_list]))


        plt.figure()
        plt.plot([state.numpy()[0,2] for state in state_list])
        plt.xlabel("month")
        plt.title(r'$T$')
        plt.savefig(export_folder + "/T_simulation.png")
        np.savetxt(export_folder + "/T_simulation.txt", np.array([state.numpy()[0,2] for state in state_list]))


        plt.figure()
        plt.plot([i_g.numpy()[0,0] for i_g in i_g_list])
        plt.xlabel("month")
        plt.title(r'$i_g$')
        plt.savefig(export_folder + "/i_g_simulation.png")
        np.savetxt(export_folder + "/i_g_simulation.txt", np.array([i_g.numpy()[0,0] for i_g in i_g_list]))


        plt.figure()
        plt.plot([i_d.numpy()[0,0] for i_d in i_d_list])
        plt.xlabel("month")
        plt.title(r'$i_d$')
        plt.savefig(export_folder + "/i_d_simulation.png")
        np.savetxt(export_folder + "/i_d_simulation.txt", np.array([i_d.numpy()[0,0] for i_d in i_d_list]))


        plt.figure()
        plt.plot([np.exp(-i_I.numpy()[0,0]) for i_I in i_I_list])
        plt.xlabel("month")
        plt.title(r'$i_I$')
        plt.savefig(export_folder + "/i_I_simulation.png")
        np.savetxt(export_folder + "/i_I_simulation.txt", np.array([ np.exp(-i_I.numpy()[0,0]) for i_I in i_I_list]))


        Y      =  np.array([state.numpy()[0,2] for state in state_list])
        R      =  np.array([state.numpy()[0,1] for state in state_list])
        K      =  np.exp(np.array([state.numpy()[0,0] for state in state_list]))

        h      = -1.0 / tf.exp(log_xi) * ((dv_dY.numpy() - \
        (self.params["gamma_1"] + self.params["gamma_2"] * Y)) * self.params["varsigma"] * \
                self.params["eta"] * self.params["A_d"] * (1 - R) * K)

        plt.figure()
        plt.plot(h)
        plt.xlabel("month")
        plt.title(r'$h$')
        plt.savefig(export_folder + "/h_simulation.png")
        np.savetxt(export_folder +  "/h_simulation.txt", h)


        # plt.figure()
        # plt.plot([g.numpy()[0,0] for g in g_list])
        # plt.xlabel("month")
        # plt.title(r'$g$')
        # plt.savefig(export_folder +  "/g_simulation.png")
        # np.savetxt(export_folder + "/g_simulation.txt", np.array([g.numpy()[0,0] for g in g_list]))
        # ## Distorted probability

        I_g        =  np.exp(np.array( [state.numpy()[0,4] for state in state_list] ))
        g          = [g.numpy()[0,0] for g in g_list]
        integrand  =  I_g * g * dt 
        integral   = -np.cumsum(integrand)
        distorted_probability = 1.0 - np.exp(integral)

        plt.figure()
        plt.plot(distorted_probability)
        plt.xlabel("month")
        plt.title(r'Distorted probability - tech jump')
        plt.savefig(export_folder + "/distort_prob_tech_simulation.png")



        plt.figure()
        plt.plot(dv_dY.numpy())
        plt.xlabel("month")
        plt.title(r'$dv dY$')
        plt.savefig(export_folder + "/dv_dY_simulation.png")
        np.savetxt(export_folder + "/dv_dY_simulation.txt", dv_dY.numpy())


        for i in range(self.params["gamma_3_length"]):

            plt.figure()
            plt.plot([f_ms[i].numpy()[0,0] for f_ms in f_ms_list])
            plt.xlabel("month")
            plt.title("f_m " + str(i+1))
            plt.savefig(export_folder + "/f_m_" + str(i+1) + "_simulation.png")
            np.savetxt(export_folder + "/f_m_" + str(i+1) + "_simulation.txt", [f_ms[i].numpy()[0,0] for f_ms in f_ms_list])
            
        for i in range(self.params["A_g_prime_length"]):

            plt.figure()
            plt.plot([g_js[i].numpy()[0,0] for g_js in g_js_list])
            plt.xlabel("month")
            plt.title("g_j " + str(i+1))
            plt.savefig(export_folder + "/g_j_" + str(i+1) + "_simulation.png")
            np.savetxt(export_folder + "/g_j_" + str(i+1) + "_simulation.txt", [g_js[i].numpy()[0,0] for g_js in g_js_list])
            
            
    def simulate_path_post_tech_post_jump(self, T, dt, gamma_3, A_g_prime, log_xi, export_folder):

        ## Create folder
        pathlib.Path(export_folder).mkdir(parents=True, exist_ok=True) 
        
        ## Initial state 
        init_logK     = tf.math.log(739.0)
        init_R        = 0.5  
        init_Y        = 1.1

        state         = tf.convert_to_tensor( [[ init_logK,  init_R, init_Y, gamma_3, log_xi, A_g_prime]] )
        state         = tf.reshape(state, (1,6))

        state_list       = [state]

        i_g_list      = [self.i_g_nn(state)]
        i_d_list      = [self.i_d_nn(state)]


        for t in range(T):


            ## Find investments
            i_g                         = self.i_g_nn(state_list[t])
            i_d                         = self.i_d_nn(state_list[t])

            logK      = state_list[t][0,0]; R = state_list[t][0,1]; 
            Y         = state_list[t][0,2]; 
            K         = tf.exp(logK)

            i_g_list.append(i_g)
            i_d_list.append(i_d)

            ## Transition
            v_kk_term      = ( tf.pow(self.params["sigma_d"],2) * tf.pow(1-R,2) + tf.pow(self.params["sigma_g"],2) * tf.pow(R,2))/2.0

            inside_log_i_d   =     tf.math.maximum( 1 + self.params["phi_d"] * i_d , 0.0001) 
            inside_log_i_g   =     tf.math.maximum( 1 + self.params["phi_g"] * i_g , 0.0001)

            v_k_term       = ( self.params["alpha_d"] + self.params["Gamma"] * tf.math.log( inside_log_i_d ) ) * (1 - R) + ( self.params["alpha_g"] +  self.params["Gamma"] * tf.math.log( inside_log_i_g) ) * R  - v_kk_term
            v_r_term       = ( self.params["alpha_g"] + self.params["Gamma"] * tf.math.log( inside_log_i_g )  - ( self.params["alpha_d"] + self.params["Gamma"] * tf.math.log( inside_log_i_d) ) + tf.pow(self.params["sigma_d"],2) *  (1-R ) - 
                            tf.pow(self.params["sigma_g"], 2) *  R ) * \
            R * (1 - R)
            
    
            new_logK       = logK + v_k_term * dt 
            new_R          = R + v_r_term * dt  
            new_Y          = Y + self.params["beta_f"] * ( self.params["eta"] * self.params["A_d"] * (1-R) * tf.exp( logK )) * dt 

            state          = tf.concat([new_logK, new_R, tf.reshape(new_Y, [1,1]), tf.reshape(gamma_3, [1,1]),  tf.reshape(log_xi, [1,1]),  tf.reshape(A_g_prime, [1,1]) ], axis=1)
            state_list.append(state)

        ################################################################
        ################################################################
        ################################################################
        ## Make plots
        ################################################################
        ################################################################
        ################################################################


        plt.figure()
        plt.plot([state.numpy()[0,0] for state in state_list])
        plt.xlabel("month")
        plt.title(r'$\log K$')
        plt.savefig(export_folder + "/logK_simulation.png")
        np.savetxt(export_folder + "/log_K_simulation.txt", np.array([state.numpy()[0,0] for state in state_list]))


        plt.figure()
        plt.plot([state.numpy()[0,1] for state in state_list])
        plt.xlabel("month")
        plt.title(r'$R$')
        plt.savefig(export_folder +  "/R_simulation.png")
        np.savetxt(export_folder + "/R_simulation.txt", np.array([state.numpy()[0,1] for state in state_list]))

        plt.figure()
        plt.plot([state.numpy()[0,2] for state in state_list])
        plt.xlabel("month")
        plt.title(r'$T$')
        plt.savefig(export_folder + "/T_simulation.png")
        np.savetxt(export_folder + "/T_simulation.txt", np.array([state.numpy()[0,2] for state in state_list]))


        plt.figure()
        plt.plot([i_g.numpy()[0,0] for i_g in i_g_list])
        plt.xlabel("month")
        plt.title(r'$i_g$')
        plt.savefig(export_folder + "/i_g_simulation.png")
        np.savetxt(export_folder + "/i_g_simulation.txt", np.array([i_g.numpy()[0,0] for i_g in i_g_list]))


        plt.figure()
        plt.plot([i_d.numpy()[0,0] for i_d in i_d_list])
        plt.xlabel("month")
        plt.title(r'$i_d$')
        plt.savefig(export_folder + "/i_d_simulation.png")
        np.savetxt(export_folder + "/i_d_simulation.txt", np.array([i_d.numpy()[0,0] for i_d in i_d_list]))


        Y      =  np.array([state.numpy()[0,2] for state in state_list])
        R      =  np.array([state.numpy()[0,1] for state in state_list])

#### Test initialization 

"""
import model
import tensorflow as tf
v_nn_config   = {"num_hiddens" : [4,4,4], "use_bias" : True, "activation" : "tanh", "dim" : 1, "nn_name" = "v_nn"}

i_g_nn_config = {"num_hiddens" : [4,4,4], "use_bias" : True, "activation" : "tanh", "dim" : 1, "nn_name" = "i_g_nn"}

i_d_nn_config = {"num_hiddens" : [4,4,4], "use_bias" : True, "activation" : "tanh", "dim" : 1, "nn_name" = "i_d_nn"}

i_I_nn_config = {"num_hiddens" : [4,4,4], "use_bias" : True, "activation" : "tanh", "dim" : 1, "nn_name" = "i_I_nn"}


## 4D 
params = {"batch_size" : 32, "R_min" : 0.01, \
"gamma_3" : 0.0, "R_max" : 0.99, "logK_min" : 4.0,\
"logK_max" : 7.0, "Y_min" : 10e-3, "Y_max" : 3.0, \
"log_I_g_max" : 6.0, "log_I_g_min": 1.0, \
"sigma_d" : 0.15 , "sigma_g" : 0.15, "A_d" : 0.12, "A_g_prime" : 0.15, \
"gamma_1" : 0.00017675, "gamma_2" : 2 * 0.0022, "gamma_3" : 0.15 , \
"y_bar" : 2.0, "beta_f" : 1.86 / 1000, "eta" : 0.17, \
"varsigma" : 1.2 * 1.86 / 1000, "phi_d" : 100.0,  "phi_g" : 100.0, "Gamma" : 0.025,  \
"alpha_d" : -0.0236, "alpha_g" : -0.0236, "delta" : 0.025, \
"v_nn_config" : v_nn_config, "i_g_nn_config" : i_g_nn_config, "i_d_nn_config" : i_d_nn_config, \
"i_I_nn_config" : i_I_nn_config, "n_dims" : 4, "model_type" : "post_damage_pre_tech", \
"num_iterations" : 100, "logging_frequency": 10, "verbose": True, "load_parameters" : None  }
params["optimizers"] = [tf.keras.optimizers.Adam(), tf.keras.optimizers.Adam()]


test_model = model.model(params)

logK, R, Y, log_I_g = test_model.sample()

test_model.train_step()


test_model.grad(logK, R, Y, log_I_g, True, True)
test_model.objective_fn(logK, R, Y, log_I_g)

######################

import model
import tensorflow as tf

v_nn_config   = {"num_hiddens" : [32 for _ in range(4)], "use_bias" : True, "activation" : "swish", "dim" : 1, "nn_name" = "v_nn"}
v_nn_config["final_activation"] = None

i_g_nn_config = {"num_hiddens" : [32 for _ in range(4)], "use_bias" : True, "activation" : "swish", "dim" : 1, "nn_name" = "i_g_nn"}
i_d_nn_config = {"num_hiddens" : [32 for _ in range(4)], "use_bias" : True, "activation" : "swish", "dim" : 1, "nn_name" = "i_d_nn"}

i_I_nn_config = {"num_hiddens" : [32 for _ in range(4)], "use_bias" : True, "activation" : "swish", "dim" : 1, "nn_name" = "i_I_nn"}
i_I_nn_config["final_activation"] = "tanh"



## 3D 
params = {"batch_size" : 32, "R_min" : 0.01, \
"R_max" : 0.99, "logK_min" : 4.0,\
"logK_max" : 7.0, "Y_min" : 10e-3, "Y_max" : 3.0, \
"log_I_g_max" : 6.0, "log_I_g_min": 1.0, \
"sigma_d" : 0.15 , "sigma_g" : 0.15, "A_d" : 0.12, "A_g_prime" : 0.15, \
"gamma_1" : 0.00017675, "gamma_2" : 2 * 0.0022, "gamma_3" : 0.15 , \
"y_bar" : 2.0, "beta_f" : 1.86 / 1000, "eta" : 0.17, \
"varsigma" : 1.2 * 1.86 / 1000, "phi_d" : 100.0,  "phi_g" : 100.0, "Gamma" : 0.025,  \
"alpha_d" : -0.0236, "alpha_g" : -0.0236, "delta" : 0.025, \
"v_nn_config" : v_nn_config, "i_g_nn_config" : i_g_nn_config, "i_d_nn_config" : i_d_nn_config, \
"n_dims" : 3, "model_type" : "post_damage_post_tech" , \
"num_iterations" : 10000, "logging_frequency": 1000, "verbose": True, "load_parameters" : None  }
params["optimizers"] = [tf.keras.optimizers.Adam( learning_rate = 10e-5, beta_1=0.8, beta_2=0.999  ), \
tf.keras.optimizers.Adam( learning_rate = 10e-5, beta_1=0.8, beta_2=0.999 ), tf.keras.optimizers.Adam( learning_rate = 10e-5, beta_1=0.8, beta_2=0.999 )]
params["load_solution"]  = "model_results.json"
params["export_folder"]   = "test_model"


params["i_g_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ params["phi_g"]) / (tf.exp(2 * x) + 1.0)
params["i_d_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ params["phi_d"]) / (tf.exp(2 * x) + 1.0)


test_model = model.model(params)
test_model.export_parameters()
test_model.train()
test_model.analyze()



test_model.train_step()


logK, R, Y = test_model.sample()




test_model.grad(logK, R, Y, True, True)
test_model.objective_fn(logK, R, Y)

"""