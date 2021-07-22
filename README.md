# DeepForest-pytorch
## Branch: bbox_size_tune

* this branch uses ray tune for hyperparameter tuning

* files come from https://github.com/NERSC/slurm-ray-cluster

* 3 files needed
    start-head.sh <br>
    start-worker.sh <br>
    submit-ray-cluster.sbatch <br>

* place the following line inside code for it to run without this error

 File "/home/iharmon1/.conda/envs/deepforest1/lib/python3.7/signal.py", line 47, in signal
    handler = _signal.signal(_enum_to_int(signalnum), _enum_to_int(handler))
    ValueError: signal only works in main thread

    `os.environ["SLURM_JOB_NAME"] = "bash"`

* edit the number of samples in tune1.py

* edit submit-ray-cluster.sbatch to load conda environment

* job parameters in submit-ray-cluster.sbatch

* to run interactively: 
1) srun -p gpu --gpus=geforce:2 --mem=64gb --time=01:00:00  --pty -u bash -i
2) ml git conda
3) make sure ray.init() includes number of cpus and gpus in tune1.py
4) conda activate deepforest1
5) python3 tune1.py 

* to run as a batch job: `sbatch submit-ray-cluster.sbatch` 
