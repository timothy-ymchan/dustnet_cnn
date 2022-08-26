#!/bin/bash

#SBATCH --job-name=mldust-valid
#SBATCH --output=%x-%j.out
#SBATCH --exclusive
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32gb:4
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL --mail-user=timothychan29475600@gmail.com

echo "ML Dust validation using map2map"
echo "Configured using 4 gpus"
echo "Training with placeholder learning rate 1e-4 and batch size 1."
echo '#SBATCH --exclusive'

hostname; pwd; date


# set computing environment, e.g. with module or anaconda

module load gcc
which gcc
#module list

source $HOME/ceph/miniconda3/bin/activate ml
#source $HOME/anaconda3/bin/activate pytorch_env
#conda info

PREP_PATH="$HOME/ceph/mldust-ver2/data"
CALLBACK="$HOME/map2map/map2map/callback"

srun python $HOME/map2map/m2m.py test \
    --test-in-patterns "$PREP_PATH/npy/TurbPar.out2.00[0-9][0-9][0-9].hdf5-vel.npy","$PREP_PATH/npy/TurbPar.out2.00[0-9][0-9][0-9].hdf5-rho.npy" \
    --test-tgt-patterns "$PREP_PATH/npy/TurbPar.out2.00[0-9][0-9][0-9].hdf5-rhop.npy","$PREP_PATH/npy/TurbPar.out2.00[0-9][0-9][0-9].hdf5-vrel.npy" \
    --load-state "./checkpoint.pt" \
    --in-norms "dust.identity,dust.simpleLog" --tgt-norms "dust.lnExp2,dust.identity" \
    --crop 128 --pad 20 \
    --model VNet --batch-size 1 --loader-workers 8 --misc-kwargs '{"lnexp-rho0":0.14,"log-eps":1e-8}' \
    --callback-at "$HOME/map2map/map2map/callback" --stats-callback "test_stat.TestStat" 
#    --augment --aug-shift 4 --crop 128 --pad 20 \
date
