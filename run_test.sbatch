#!/bin/bash
#SBATCH --job-name=gea_gqap_test
#SBATCH --error=gea_gqap_test-%j.err
#SBATCH --output=gea_gqap_test-%j.log
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus=0

module load Python
source activate gea_gqap_env

poetry run python gea_gqap_adaptive_python/tests/test_20250301/run_comparison_test.py
