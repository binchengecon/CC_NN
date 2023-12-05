#!/bin/sh

pre_tech_pre_damage_export_folder="/scratch/midway3/bincheng/pre_tech_pre_damage_models_12052023_tensorboard_version_iteration_500000"
pre_tech_post_damage_export_folder="${pre_tech_pre_damage_export_folder}/pre_tech_post_damage"
post_tech_pre_damage_export_folder="${pre_tech_pre_damage_export_folder}/post_tech_pre_damage"

pretrained_pre_tech_pre_damage_export_folder="None"
pretrained_pre_tech_post_damage_export_folder="None"
pretrained_post_tech_pre_damage_export_folder="None"

log_xi_min="-3.0"
log_xi_max="-1.5"
batch_size="32"
# num_iterations="2000000"
num_iterations="500000"
# num_iterations="100"
# num_iterations="100"
A_g_prime_min="0.1"
A_g_prime_max="0.2"
logging_frequency="1000"
learning_rates="10e-5,10e-5,10e-5,10e-5"
hidden_layer_activations="swish,tanh,tanh,softplus"
output_layer_activations="None,custom,custom,softplus"
num_hidden_layers="4"
num_neurons="32"
learning_rate_schedule_type="None"
delta="0.025"

tensorboard='True'
touch main_train.sh

tee main_train.sh <<EOF
#!/bin/sh

######## login

#SBATCH --job-name=modification
#SBATCH --output=runtd_short_${num_iterations}.out
#SBATCH --error=runtd_short_${num_iterations}.err
#SBATCH --time=0-36:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
# module load tensorflow/2.1
# module unload cuda
# module unload python
# module load cuda/11.2
module load python/anaconda-2021.05


# job_file="pre_tech_pre_damage_simulation_${model_num}.job"

echo "Export folder: $pre_tech_pre_damage_export_folder"
echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task

python pre_tech_post_damage.py $pre_tech_post_damage_export_folder $log_xi_min $log_xi_max $batch_size $num_iterations $A_g_prime_min $A_g_prime_max  $pretrained_pre_tech_post_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $delta $tensorboard
python post_tech_pre_damage.py $post_tech_pre_damage_export_folder $pre_tech_post_damage_export_folder $log_xi_min $log_xi_max $batch_size $num_iterations  $A_g_prime_min $A_g_prime_max   $pretrained_post_tech_pre_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $delta $tensorboard
python pre_tech_pre_damage.py $pre_tech_pre_damage_export_folder $post_tech_pre_damage_export_folder $pre_tech_post_damage_export_folder -10 $log_xi_min $log_xi_max $batch_size $num_iterations  $A_g_prime_min $A_g_prime_max   $pretrained_pre_tech_pre_damage_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $delta $tensorboard

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF

sbatch main_train.sh
