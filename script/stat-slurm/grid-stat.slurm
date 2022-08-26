#!/bin/bash

#SBATCH --job-name=gridstats
#SBATCH --output=%x-%j.out
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=ALL --mail-user=timothychan29475600@gmail.com

hostname; pwd; date

module load gcc
source $HOME/ceph/miniconda3/bin/activate ml


CODE_DIR="$HOME/ceph/grid-stats"
DATA_DIR="$HOME/ceph/grid-stats"

python "$CODE_DIR/particleRDFvelMulti-grid.py" --nsamples 65536 --vp "$DATA_DIR/TurbPar.out2.00040.hdf5-vp.npy" --rhop "$DATA_DIR/TurbPar.out2.00040.hdf5-rhop.npy" 
