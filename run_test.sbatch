#!/bin/bash
#SBATCH --job-name=gea_gqap_test
#SBATCH --error=gea_gqap_test-%j.err
#SBATCH --output=gea_gqap_test-%j.log
#SBATCH --time=336:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gpus=0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUM_WORKERS=16

module load Python
source activate gea_gqap_env

poetry run python gea_gqap_adaptive_python/tests/test_20250301/run_comparison_test.py
