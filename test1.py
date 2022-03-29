# load modules
import comet
import my_parse as pars
import my_log as mlg
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

n_classes = 1
rules = [FOL_competition(device, 1, None, None), ]   #[FOL_green(device, 2, None, None), ]
rule_lambdas = [1]  # default 0.1
pi_params = [pars.args.pi_0, pars.args.pi_1]
batch_size = 1
C = pars.args.C

# directory with image and annotation data
train_dir = pars.args.train_dir
eval_dir = pars.args.test_dir

train_csv = os.path.join(train_dir, pars.args.train_ann)
val_csv = os.path.join(train_dir, pars.args.val_ann)
test_csv = os.path.join(eval_dir, pars.args.test_ann)

"""## Training & Evaluating Using GPU"""

# initialize the model and change the corresponding config file
torch.manual_seed(pars.args.seed)
torch.cuda.manual_seed_all(pars.args.seed)
np.random.seed(pars.args.seed)

m = main.deepforest(rules, rule_lambdas, pi_params, C, num_classes=n_classes, batch_size=batch_size).to(device)
m.config['gpus'] = '-1' #move to GPU and use all the GPU resources
m.config["train"]["csv_file"] = train_csv
m.config["train"]["root_dir"] = train_dir
m.config["score_thresh"] = 0.46 # default 0.4
m.config["train"]['epochs'] = pars.args.epochs
m.config["validation"]["csv_file"] = val_csv
m.config["validation"]["root_dir"] = train_dir
m.config["nms_thresh"] = 0.57
m.config["train"]["lr"] = 0.0017997179587414414  # default 0.001
m.config["batch_size"] = batch_size

print("Training csv: {}".format(m.config["train"]["csv_file"]))

training_data = m.train_dataloader()
m.config["train"]["beg_incr_pi"] = round(len(training_data) * pars.args.pi_start)
n_train_batches = len(training_data) / batch_size
m.config["train"]["n_train_batches"] = n_train_batches

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
csv_wr_obj = mlg.Writer(pars.args.log, ["site", "seed", "epochs", "chm", "train", "test", "bbox_prec", "bbox_rec", "class_prec", "class_rec"])
csv_wr_obj.write_data([pars.args.site, str(pars.args.seed), str(pars.args.epochs), pars.args.chm, train_csv, test_csv, results['box_precision'], results['box_recall'], results['class_recall']['precision'].item(), results['class_recall']['recall'].item()])
print("bbox prec: {}".format(results['box_precision']))
print("bbox rec: {}".format(results['box_recall']))
print("class prec: {}".format(results['class_recall']['precision'].item()))
print("class rec: {}".format(results['class_recall']['recall'].item()))

file_list = [f for f in os.listdir(save_dir) if (f.split(".")[1] == 'png') or (f.split(".")[1] =='tif')]

for f in file_list[:34]:
   comet.experiment.log_image('./pred_result2/' + f)

comet.experiment.add_tags([os.path.basename(test_csv).split('-')[0].lower(), "reg"])
comet.experiment.log_others({'box_precision': results['box_precision'], 
			     'box_recall': results['box_recall'], 
                             'class_precision': results['class_recall']['precision'].item(), 
                             'class_recall': results['class_recall']['recall'].item(), 
                             'size': results['class_recall']['size'].item()})
comet.experiment.log_parameter('pi_params', pi_params)
comet.experiment.log_parameter('m.config', m.config)
comet.experiment.log_parameter("m.config['train']", m.config['train'])
repo = Repository('.git')
last = repo[repo.head.target]
comet.experiment.log_parameter('git branch', Repository('.').head.shorthand)
comet.experiment.log_parameter('last_commit', last.id)
comet.experiment.log_table('./pred_result2/predictions.csv')
comet.experiment.log_table('./pred_result2/matches.csv')
comet.experiment.log_code(file_name='test1.py')
comet.experiment.log_code(file_name='deepforest/main.py')
comet.experiment.log_code(file_name='deepforest/fol.py')
comet.experiment.log_code(file_name='deepforest/logic_nn.py')
comet.experiment.log_code(file_name='deepforest/predict.py')
comet.experiment.log_code(file_name='deepforest/evaluate.py')
