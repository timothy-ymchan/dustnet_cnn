#!/bin/bash

#SBATCH --job-name=gridstats
#SBATCH --output=%x-%j.out
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL --mail-user=timothychan29475600@gmail.com

hostname; pwd; date

module load gcc
source $HOME/ceph/miniconda3/bin/activate ml


CODE_DIR="$HOME/ceph/rdf-grid"
DATA_DIR="$HOME/ceph/mldust-ver2/runs/wvelc-stats/all-out/"

python "$CODE_DIR/dust-rdf.py" --rhop "$DATA_DIR/*rhop_out.npy" --outname "unetv-normal" 
