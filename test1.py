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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
n_classes = 1
batch_size = 1

# directory with image and annotation data
train_dir = "/blue/daisyw/iharmon1/data/DeepForest-pytorch/training3"
eval_dir = "/blue/daisyw/iharmon1/data/DeepForest-pytorch/evaluation3"

train_csv = os.path.join(train_dir, "SJER-train.csv")  
val_csv = os.path.join(train_dir, "SJER-val.csv")    
test_csv = os.path.join(eval_dir, "SJER-test.csv") 

"""## Training & Evaluating Using GPU"""

# initialize the model and change the corresponding config file
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
m = main.deepforest()
m.config['gpus'] = '-1' #move to GPU and use all the GPU resources
m.config["train"]["csv_file"] = train_csv
m.config["train"]["root_dir"] = train_dir
m.config["score_thresh"] = 0.46  # default 0.4
m.config["train"]['epochs'] = 5
m.config["validation"]["csv_file"] = val_csv
m.config["validation"]["root_dir"] = train_dir
m.config["nms_thresh"] = 0.57  # default 0.05
m.config["train"]["lr"] = 0.0017997179587414414  # default 0.001

print("Training csv: {}".format(m.config["train"]["csv_file"]))

training_data = m.train_dataloader()
n_train_batches = len(training_data) / batch_size
m.config["train"]["n_train_batches"] = n_train_batches

# create a pytorch lighting trainer used to training
checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='./checkpoints', filename='deepforest_chkpt-{epoch:02d}-{val_loss:.2f}', save_top_k=1, mode='min',)
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

results = m.evaluate(test_csv, eval_dir, iou_threshold = 0.5, show_plot = False, savedir = save_dir)
print("bbox prec: {}".format(results['box_precision']))
print("bbox rec: {}".format(results['box_recall']))
print("class prec: {}".format(results['class_recall']['precision'].item()))
print("class rec: {}".format(results['class_recall']['recall'].item()))

file_list = [f for f in os.listdir(save_dir) if (f.split(".")[1] == 'png') or (f.split(".")[1] =='tif')]

for f in file_list[:34]:
   comet.experiment.log_image('./pred_result2/' + f)

comet.experiment.add_tags([os.path.basename(test_csv).split('-')[0].lower(), "baseline"])
comet.experiment.log_others({'box_precision': results['box_precision'], 
			     'box_recall': results['box_recall'], 
                             'class_precision': results['class_recall']['precision'].item(), 
                             'class_recall': results['class_recall']['recall'].item(), 
                             'size': results['class_recall']['size'].item()})
comet.experiment.log_parameter('m.config', m.config)
comet.experiment.log_parameter("m.config['train']", m.config['train'])

repo = Repository('.git')
last_commit = repo[repo.head.target]
comet.experiment.log_parameter('git branch', Repository('.').head.shorthand)
comet.experiment.log_parameter('last_commit', last_commit.id)
comet.experiment.log_table('./pred_result2/predictions.csv')
comet.experiment.log_table('./pred_result2/matches.csv')
comet.experiment.log_code(file_name='./test1.py')
comet.experiment.log_code(file_name='deepforest/main.py')
comet.experiment.log_code(file_name='deepforest/fol.py')
comet.experiment.log_code(file_name='deepforest/logic_nn.py')
comet.experiment.log_code(file_name='deepforest/predict.py')
