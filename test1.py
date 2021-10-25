# load modules
import comet
import os
import time
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pygit2 import Repository
from deepforest import main
from deepforest import get_data
from deepforest import utilities
from deepforest import preprocess
from deepforest import callbacks
from deepforest.fol import FOL_green, FOL_competition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
n_classes = 1
rules = [FOL_competition(device, 1, None, None), ]   #[FOL_green(device, 2, None, None), ]
rule_lambdas = [1]  # default 0.1
pi_params = [0.6, 0.40]
batch_size = 1
C = 1 # default 9

# directory with image and annotation data
#data_dir = "/blue/daisyw/iharmon1/data/DeepForest-pytorch/train_data_folder2"
train_dir = "/blue/daisyw/iharmon1/data/DeepForest-pytorch/training4"
eval_dir = "/blue/daisyw/iharmon1/data/DeepForest-pytorch/evaluation4"

train_csv = os.path.join(train_dir, "NIWO-train.csv")  #os.path.join(data_dir, "train.csv")
val_csv = os.path.join(train_dir, "NIWO-val.csv")      #os.path.join(data_dir, "val.csv")
test_csv = os.path.join(eval_dir, "NIWO-test.csv")     #os.path.join(data_dir, "test_small.csv")

"""## Training & Evaluating Using GPU"""

# initialize the model and change the corresponding config file
m = main.deepforest(rules, rule_lambdas, pi_params, C, num_classes=n_classes, batch_size=batch_size).to(device)
m.config['gpus'] = '-1' #move to GPU and use all the GPU resources
m.config["train"]["csv_file"] = train_csv
m.config["train"]["root_dir"] = train_dir
m.config["score_thresh"] = 0.46 # default 0.4
m.config["train"]['epochs'] = 15
m.config["validation"]["csv_file"] = val_csv
m.config["validation"]["root_dir"] = train_dir
m.config["nms_thresh"] = 0.05
m.config["train"]["lr"] = 0.0017997179587414414  # default 0.001
m.config["train"]["beg_incr_pi"] = 78e6 # 385 #480
m.config["batch_size"] = batch_size

print("Training csv: {}".format(m.config["train"]["csv_file"]))

training_data = m.train_dataloader()
n_train_batches = len(training_data) / batch_size
m.config["train"]["n_train_batches"] = n_train_batches

# create a pytorch lighting trainer used to training
#checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='./checkpoints', filename='deepforest_chkpt-{epoch:02d}-{val_loss:.2f}', save_top_k=1, mode='min',)
#m.create_trainer(callbacks=[checkpoint_callback])
m.create_trainer()

# load the lastest release model
#m.use_release()

start_time = time.time()
m.trainer.fit(m)
print(f"--- Training on GPU: {(time.time() - start_time):.2f} seconds ---")

#save the prediction result to a prediction folder
save_dir = os.path.join(os.getcwd(),'pred_result2')

try:
   os.mkdir(save_dir)
except OSError as error:
   pass
#results = m.evaluate(test_csv, data_dir, iou_threshold = 0.4, show_plot = False, savedir= save_dir)
results = m.evaluate(test_csv, eval_dir, iou_threshold = 0.4, show_plot = False, savedir= save_dir)

file_list = [f for f in os.listdir(save_dir) if (f.split(".")[1] == 'png') or (f.split(".")[1] =='tif')]

for f in file_list[:33]:
   comet.experiment.log_image('./pred_result2/' + f)

comet.experiment.add_tags(["chm", "niwo"])
comet.experiment.log_others(results)
comet.experiment.log_parameter('pi_params', pi_params)
comet.experiment.log_parameter('m.config', m.config)
comet.experiment.log_parameter("m.config['train']", m.config['train'])
repo = Repository('.git')
last = repo[repo.head.target]
comet.experiment.log_parameter('git branch', Repository('.').head.shorthand)
comet.experiment.log_parameter('last_commit', last.id)
comet.experiment.log_table('./pred_result/predictions.csv')
comet.experiment.log_code(file_name='deepforest/main.py')
comet.experiment.log_code(file_name='deepforest/fol.py')
comet.experiment.log_code(file_name='logic_nn.py')
