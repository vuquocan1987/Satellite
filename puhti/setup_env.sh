
module load pytorch

python -m venv --system-site-packages /scratch/project_2017985/SateliteData/venv

source /scratch/project_2017985/SateliteData/venv/bin/activate

pip3 install -r /scratch/project_2017985/SateliteData/requirements.txt

echo "Environment ready!"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
