rsync -avz --exclude 'experiments' --exclude 'datasets' --exclude '__pycache__' --exclude 'logs' --exclude '.git'\
        --exclude 'checkpoints' \
    ~/Workspace/SateliteData/ \
    vuan1111@puhti.csc.fi:/scratch/project_2017985/SateliteData/