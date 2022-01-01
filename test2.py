# load modules
import comet
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
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
n_classes = 1
rules = [FOL_bbox_2big(device, 1, None, None), ]   #[FOL_green(device, 2, None, None), ]
rule_lambdas = [1e2]
pi_params = [0.80, 0.5]
batch_size = 1
C = 0.01


parser = argparse.ArgumentParser()
parser.add_argument('--site', type=str, required=True, help='name of site')
parser.add_argument('--train_dir', type=str, required=True, help='training directory')
parser.add_argument('--test_dir', type=str, required=True, help='test directory')
parser.add_argument('--train_ann', type=str, required=True, help='training annotations')
parser.add_argument('--test_ann', type=str, required=True, help='testing annotations')
args = parser.parse_args()


# directory with image and annotation data
train_dir = args.train_dir
eval_dir = args.test_dir

train_csv = os.path.join(train_dir, args.train_ann)  
val_csv = os.path.join(train_dir, "TEAK-val.csv")     
test_csv = os.path.join(eval_dir, args.test_ann) 

"""## Training & Evaluating Using GPU"""

# initialize the model and change the corresponding config file
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
m = main.deepforest(rules, rule_lambdas, pi_params, C, num_classes=n_classes).to(device)
m.config['gpus'] = '-1' #move to GPU and use all the GPU resources
m.config["train"]["csv_file"] = train_csv
m.config["train"]["root_dir"] = train_dir
m.config["score_thresh"] = 0.46  # default 0.4
m.config["train"]['epochs'] = 4
m.config["validation"]["csv_file"] = val_csv
m.config["validation"]["root_dir"] = train_dir
m.config["nms_thresh"] = 0.57  # default 0.05
m.config["train"]["lr"] = 0.0017997179587414414  # default 0.001

print("Training csv: {}".format(m.config["train"]["csv_file"]))

training_data = m.train_dataloader()
n_train_batches = len(training_data) / batch_size
m.config["train"]["n_train_batches"] = n_train_batches
m.config["train"]["beg_incr_pi"] = round(len(training_data) * 1)

# create a pytorch lighting trainer used to training
#checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='./checkpoints', filename='deepforest_chkpt-{epoch:02d}-{val_loss:.2f}', save_top_k=1, mode='min',)
#m.create_trainer(callbacks=[checkpoint_callback])
m.create_trainer()

# load the lastest release model
m.use_release()

start_time = time.time()
m.trainer.fit(m)
print(f"--- Training on GPU: {(time.time() - start_time):.2f} seconds ---")

#save the prediction result to a prediction folder
save_dir = os.path.join(os.getcwd(),'pred_result2')

try:
   os.mkdir(save_dir)
except OSError as error:
   pass

results = m.evaluate(test_csv, eval_dir, iou_threshold = 0.5, show_plot = False, savedir= save_dir)

#######################################################################################################################################################################################################
# log data to locally
log_fname = "df_teak_bbox_ht_log.csv".format(args.site)
if not (os.path.isfile(log_fname)):
   f = open(log_fname, "w")
   f.write("site,train,test,bbox_prec,bbox_rec,class_prec,class_rec\n")
   f.close()

f = open(log_fname, "a")
f.write("{},{},{},{},{},{},{}\n".format(args.site, args.train_ann, args.test_ann, results['box_precision'], results['box_recall'], results['class_recall']['precision'].item(), results['class_recall']['recall'].item()))
f.close()
######################################################################################################################################################################################################

file_list = [f for f in os.listdir(save_dir) if (f.split(".")[1] == 'png') or (f.split(".")[1] =='tif')]

for f in file_list[:34]:
   comet.experiment.log_image('./pred_result2/' + f)

comet.experiment.add_tags([os.path.basename(test_csv).split('-')[0].lower()])
comet.experiment.log_others(results)
comet.experiment.log_parameter('pi_params', pi_params)
comet.experiment.log_parameter('m.config', m.config)
comet.experiment.log_parameter("m.config['train']", m.config['train'])

repo = Repository('.git')
last_commit = repo[repo.head.target]
comet.experiment.log_parameter('git branch', Repository('.').head.shorthand)
comet.experiment.log_parameter('last_commit', last_commit.id)
comet.experiment.log_table('./pred_result2/predictions.csv')
comet.experiment.log_table('./pred_result2/matches.csv')
comet.experiment.log_code(file_name='test1.py')
comet.experiment.log_code(file_name='deepforest/main.py')
comet.experiment.log_code(file_name='deepforest/fol.py')
comet.experiment.log_code(file_name='deepforest/logic_nn.py')
comet.experiment.log_code(file_name='deepforest/predict.py')

