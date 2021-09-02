# DeepForest-pytorch
## Branch: bbox_size

* tries to train the neural network to prefer bounding boxes of a reduced area (< 400 square pixels)

* uses hu FOL rules

* this version is for hipergator

* everything works

* conda environment causes core dump on hipergator at the end of the run

* alterations made in main.py

* results stored on comet.ml

* tags "bbox_2big_rle", "big_ds"

* trained on entire ds


## to run

srun -p gpu --gpus=1 --mem=64gb --time=02:00:00  --pty -u bash -i
ml conda git
conda activate deepforest1
python3 test1.py
