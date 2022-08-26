#!/bin/bash

#SBATCH --job-name=mldust-valid
#SBATCH --output=%x-%j.out
#SBATCH --exclusive
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL --mail-user=timothychan29475600@gmail.com

echo "ML Dust validation using map2map"
echo "Configured using 4 gpus"
echo "Training with placeholder learning rate 1e-4 and batch size 1."
echo '#SBATCH --exclusive'

hostname; pwd; date


# set computing environment, e.g. with module or anaconda

module load cuda
module load cudnn
#module list

source $HOME/ceph/miniconda3/bin/activate ml2
#source $HOME/anaconda3/bin/activate pytorch_env
#conda info

PREP_PATH="$HOME/ceph/mldust-ver2/data"
CALLBACK="$HOME/map2map/map2map/callback"

for i in {1..20..1}
    do
        scale=0
        echo "$i,$scale"
        srun python $HOME/map2map/m2m.py test \
            --train-in-patterns "$PREP_PATH/train-in/TurbPar.out2.00[0-9][1-9][0-9].hdf5-vel.npy","$PREP_PATH/train-in/TurbPar.out2.00[0-9][1-9][0-9].hdf5-rho.npy" \
            --train-tgt-patterns "$PREP_PATH/train-out/TurbPar.out2.00[0-9][1-9][0-9].hdf5-rhop.npy","$PREP_PATH/train-out/TurbPar.out2.00[0-9][1-9][0-9].hdf5-vrel.npy" \
            --test-in-patterns "$PREP_PATH/valid-in/TurbPar.out2.0010[0-9].hdf5-vel.npy","$PREP_PATH/valid-in/TurbPar.out2.0010[0-9].hdf5-rho.npy" \
            --test-tgt-patterns "$PREP_PATH/valid-out/TurbPar.out2.0010[0-9].hdf5-rhop.npy","$PREP_PATH/valid-out/TurbPar.out2.0010[0-9].hdf5-vrel.npy" \
            --load-state "./swag_state_445.pt" --swag_sample_scale $scale\
            --in-norms "dust.identity,dust.simpleLog" --tgt-norms "dust.lnExp2,dust.identity" \
            --crop 128 --pad 20 --valid-npy-out-dir "./with-bn-$i-scale-$scale/"\
            --model VNet --batch-size 1 --loader-workers 8 --misc-kwargs '{"lnexp-rho0":0.14,"log-eps":1e-8}' \
            --callback-at "$HOME/map2map/map2map/callback" --stats-callback "test_stat.TestStat" --swag --swag_use_conv  

    done
#    --augment --aug-shift 4 --crop 128 --pad 20 \
date
