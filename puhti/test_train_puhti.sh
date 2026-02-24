#!/bin/bash
#SBATCH --account=project_2017985
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=flood_diffusion
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

mkdir -p logs

module load pytorch
source /scratch/project_2017985/SateliteData/venv/bin/activate

cd /scratch/project_2017985/SateliteData

python run.py -p train -c config/flood_satellite.json