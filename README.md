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

* to run: `sbatch submit-ray-cluster.sbatch` 

