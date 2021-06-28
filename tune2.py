# for debugging
#RAY_RAYLET_GDB=1 RAY_RAYLET_TMUX=1 python

# load modules
import os
import time
import numpy as np
import torch
from pygit2 import Repository
from pytorch_lightning.callbacks import ModelCheckpoint
from deepforest import main
from deepforest import get_data
from deepforest import utilities
from deepforest import preprocess
from deepforest import callbacks
from deepforest.fol import FOL_green, FOL_competition, FOL_bbox_2big

# packages needed for tuning
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback


def train_deepforest_tune(config, num_epochs=10, num_gpus=0):
   print("here 0")
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   np.random.seed(42)
   n_classes = 1
   rules = [FOL_competition(device, 1, None, None), ]   #[FOL_green(device, 2, None, None), ]
   rule_lambdas = [1]
   pi_params = [config['pi_0'], config['pi_1']] # [0.9, 0]
   batch_size = 1
   C = config['C'] # 6

   # directory with image and annotation data
   data_dir = "/blue/daisyw/iharmon1/data/DeepForest-pytorch/train_data_folder2"

   train_csv = os.path.join(data_dir, "train.csv")
   val_csv = os.path.join(data_dir, "val.csv")
   test_csv = os.path.join(data_dir, "test_small.csv")

   """## Training & Evaluating Using GPU"""

   # initialize the model and change the corresponding config file
   m = main.deepforest(rules, rule_lambdas, pi_params, C, num_classes=n_classes).to(device)
   m.config['gpus'] = '-1' #move to GPU and use all the GPU resources
   m.config["train"]["csv_file"] = train_csv
   m.config["train"]["root_dir"] = data_dir
   m.config["train"]["lr"] = config["lr"]
   m.config["validation"]["csv_file"] = val_csv
   m.config["train"]['epochs'] = num_epochs
   m.config["validation"]["root_dir"] = data_dir
   m.config["score_thresh"] = config['score_thresh']    # old value 0.4
   m.config["nms_thresh"] =  config['nms_thresh']       # old value 0.05

   training_data = m.train_dataloader()
   n_train_batches = len(training_data) / batch_size
   m.config["train"]["n_train_batches"] = n_train_batches

   # create a pytorch lighting trainer
   #checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='./checkpoints', filename='deepforest_chkpt-{epoch:02d}-{val_loss:.2f}', save_top_k=1, mode='min',)
   raytune_callback = TuneReportCallback(metrics={"val_class_loss": "class_loss", "val_reg_loss": 'reg_loss', "val_avg_loss":'avg_loss'}, on="validation_end")

   m.create_trainer(callbacks=[raytune_callback])
   m.trainer.fit(m)


def tune_deepforest_asha(num_samples=10, num_epochs=10, gpus_per_trial=0):
    config = {
        "pi_0": tune.uniform(0.50, 0.99),
        "pi_1": 0,
        "C": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "score_thresh": 0.4,
        "nms_thresh": 0.05
    }

    scheduler = ASHAScheduler(
        max_t=3,
        grace_period=2,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["pi_0", "pi_1", "lr", "C", "score_thresh", "nms_thresh"],
        metric_columns=["val_class_loss", "val_reg_loss", "val_avg_loss", "training_iteration"])

    analysis = tune.run(train_deepforest_tune, config=config, verbose=3, resume=False)

    print("Best hyperparameters found were: ", analysis.best_config)

tune_deepforest_asha(3, 2, 1)


