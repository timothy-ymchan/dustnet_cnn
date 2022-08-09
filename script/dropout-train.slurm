#!/bin/bash

#SBATCH --job-name=mldust
#SBATCH --output=%x-%j.out
#SBATCH --exclusive
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32gb:4
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL --mail-user=timothychan29475600@gmail.com

echo "ML Dust prediction using map2map"
echo "Configured using 4 gpus"
echo "Training with placeholder learning rate 1e-4 and batch size 1."
echo '#SBATCH --exclusive'

hostname; pwd; date


# set computing environment, e.g. with module or anaconda

#module load gcc
#module load python3
#module list
module load cuda
module load cudnn

which gcc

#conda activate ml
source $HOME/ceph/miniconda3/bin/activate ml2
#source $HOME/anaconda3/bin/activate
#source $HOME/anaconda3/bin/activate pytorch_env
#conda info

#export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:/usr/lib64
echo "$LD_LIBRARY_PATH"

PREP_PATH="$HOME/ceph/mldust-ver2/data"
CALLBACK="$HOME/map2map/map2map/callback"

srun python $HOME/map2map/m2m.py train \
    --train-in-patterns "$PREP_PATH/train-in/TurbPar.out2.00[0-9][1-9][0-9].hdf5-vel.npy","$PREP_PATH/train-in/TurbPar.out2.00[0-9][1-9][0-9].hdf5-rho.npy" \
    --train-tgt-patterns "$PREP_PATH/train-out/TurbPar.out2.00[0-9][1-9][0-9].hdf5-rhop.npy","$PREP_PATH/train-out/TurbPar.out2.00[0-9][1-9][0-9].hdf5-vrel.npy" \
    --val-in-patterns "$PREP_PATH/valid-in/TurbPar.out2.00[0-9][1-9][0-9].hdf5-vel.npy","$PREP_PATH/valid-in/TurbPar.out2.00[0-9][1-9][0-9].hdf5-rho.npy" \
    --val-tgt-patterns "$PREP_PATH/valid-out/TurbPar.out2.00[0-9][1-9][0-9].hdf5-rhop.npy","$PREP_PATH/valid-out/TurbPar.out2.00[0-9][1-9][0-9].hdf5-vrel.npy" \
    --in-norms "dust.identity,dust.simpleLog" --tgt-norms "dust.lnExp2,dust.identity" \
    --crop 128 --pad 20 --aug-shift 4 --criterion MSELoss \
    --model DustNet --callback-at "$CALLBACK" --load-state checkpoint.pt\
    --lr 1e-4 --reduce-lr-on-plateau --optimizer-args '{"weight_decay": 1e-4}' \
    --batch-size 1 --loader-workers 8 --div-data --div-shuffle-dist 8 \
    --epochs 1024 --misc-kwargs '{"lnexp-rho0":0.14,"log-eps":1e-8,"mc_dropout_p":0.1}' 

#    --augment --aug-shift 4 --crop 128 --pad 20 \
date
