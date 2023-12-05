#!/bin/sh

######## login

#SBATCH --job-name=modification
#SBATCH --output=runtd_short_500000.out
#SBATCH --error=runtd_short_500000.err
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


# job_file="pre_tech_pre_damage_simulation_.job"

echo "Export folder: /scratch/midway3/bincheng/pre_tech_pre_damage_models_12052023_tensorboard_version_iteration_500000"
echo "$SLURM_JOB_NAME"

echo "Program starts $(date)"
start_time=$(date +%s)
# perform a task

python pre_tech_post_damage.py /scratch/midway3/bincheng/pre_tech_pre_damage_models_12052023_tensorboard_version_iteration_500000/pre_tech_post_damage -3.0 -1.5 32 500000 0.1 0.2  None 1000 10e-5,10e-5,10e-5,10e-5 swish,tanh,tanh,softplus None,custom,custom,softplus 4 32 None 0.025 True
python post_tech_pre_damage.py /scratch/midway3/bincheng/pre_tech_pre_damage_models_12052023_tensorboard_version_iteration_500000/post_tech_pre_damage /scratch/midway3/bincheng/pre_tech_pre_damage_models_12052023_tensorboard_version_iteration_500000/pre_tech_post_damage -3.0 -1.5 32 500000  0.1 0.2   None 1000 10e-5,10e-5,10e-5,10e-5 swish,tanh,tanh,softplus None,custom,custom,softplus 4 32 None 0.025 True
python pre_tech_pre_damage.py /scratch/midway3/bincheng/pre_tech_pre_damage_models_12052023_tensorboard_version_iteration_500000 /scratch/midway3/bincheng/pre_tech_pre_damage_models_12052023_tensorboard_version_iteration_500000/post_tech_pre_damage /scratch/midway3/bincheng/pre_tech_pre_damage_models_12052023_tensorboard_version_iteration_500000/pre_tech_post_damage -10 -3.0 -1.5 32 500000  0.1 0.2   None 1000 10e-5,10e-5,10e-5,10e-5 swish,tanh,tanh,softplus None,custom,custom,softplus 4 32 None 0.025 True

echo "Program ends $(date)"
end_time=$(date +%s)

# elapsed time with second resolution
elapsed=$((end_time - start_time))

eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"

