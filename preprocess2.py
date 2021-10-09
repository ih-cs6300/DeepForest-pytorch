# This file converts the deepforest benchmark dataset annotations from .xml to .csv files. 
# Then the .tif files over 500 x 500 pixles are split into smaller files.
# Benchmark dataset repo: https://github.com/weecology/NeonTreeEvaluation.
# Large tifs stored on zenodo: https://zenodo.org/record/4746605.

import re
from os.path import join
from os import mkdir
from os import remove
from glob import glob
from os.path import basename
from deepforest import utilities
from shutil import copyfile
from deepforest.preprocess import split_raster
from PIL import Image

ann_dir = "/home/iharmon1/data/NeonTreeEvaluation/annotations"
eval_dir = "./evaluation3"
train_dir = "./training3"
rgb_dir = "/home/iharmon1/data/NeonTreeEvaluation/evaluation/RGB"

####################################################################################################################################################################
# functions

def sep_train_eval_ann(files, regex):
   training = []
   evaluation = []
   for file_name in files:
      if re.search(regex, file_name) is not None:
         training.append(file_name)
      else:
         evaluation.append(file_name)
   return training, evaluation




def sep_train_eval_img(img_files, training_list, eval_list):
   train_img_list = []
   eval_img_list = []

   train_ann_rem = []
   eval_ann_rem = []

   # loop over training annotations
   for f in training_list:
      # see if an annotation file has a corresponding image file
      img_filename = f + ".tif"
      if img_filename in image_files:
         # add image file to training_img_list
         train_img_list.append(img_filename)
      else:
         # remove from annotations if no corresponding image
         train_ann_rem.append(f)

   # loop over evaluation annotations
   for f in eval_list:
      img_filename = f + ".tif"
      if img_filename in image_files:
         eval_img_list.append(img_filename)
      else:
         # remove .csv from list since it has no corresponding image file
         eval_ann_rem.append(f)

   return train_img_list, eval_img_list, train_ann_rem, eval_ann_rem




def xml_2_csv(annot_path, file_name, storage_dir):
   # converts xml to csv and saves in storage directory
   annot_df = utilities.xml_to_annotations(join(annot_path, file_name))
   new_name = f.split(".")[0] +	".csv"
   annot_df.to_csv(join(storage_dir, new_name), header=True, index=False)




def copy_files(source_path, dest_path, file_list):
   for f in file_list:
      copyfile(join(source_path, f), join(dest_path, f))




def clean_annotations(train_dir, eval_dir, train_rem_list, eval_rem_list):
   # deletes .csv files that don't have corresponding image files
   train_remove_count = len(train_rem_list)
   for ann in train_rem_list:
      ann_name = ann + ".csv"
      remove(join(train_dir, ann_name))
  
   eval_remove_count = len(eval_rem_list)
   for ann in eval_rem_list:
      ann_name = ann + ".csv"
      remove(join(eval_dir, ann_name))

   print("   Removed {} training annotations...".format(train_remove_count))
   print("   Removed {} evaluation annotations...".format(eval_remove_count))



def split_large_images(working_dir, image_files):
    for f in image_files:
       img_name = f + ".tif"
       ann_name = f + ".csv"
       print("Image: {}".format(img_name))
       img = Image.open(join(working_dir, img_name))

       # split files larger than 500 pixels in either dimension
       # the image file is saved as a PNG
       # the annotations file is saved as image_file_name.csv
       if (img.size[0] > 500 or img.size[1] > 500):

          # if a file is sent to split_raster but it has no annotations in train_df it causes an error
          # check if in train_df before sending to split_raster
          print("   Spliting...")
          ann_df = split_raster(join(working_dir, img_name), join(working_dir, ann_name), working_dir, 400, 0.05)

          # remove original tif and csv
          print("   Removing {}...".format(join(working_dir, img_name)))
          print("   Removing {}...".format(join(working_dir, ann_name)))
          remove(join(working_dir, img_name))

          # don't remove .csv file because the original is overwritten by split_raster()
          #remove(join(working_dir, ann_name))
       img.close()
   
#####################################################################################################################################################################
# driver code

#import pdb; pdb.set_trace()
files = glob(join(ann_dir, "*.xml")) 
files = [basename(x) for x in files]      
training_regex = "[\d]{4}_[\w]{4}_[\d]{1}_[\d]{6}_[\d]{7}_image.*\.xml$"

# annotations are divided into training and evaluation based on file name
# example training file name: 2018_TEAK_3_315000_4094000_image_crop.xml
# example evaluation file name: SJER_045_2018.xml
training_ann_list, evaluation_ann_list = sep_train_eval_ann(files, training_regex)

# using the training and evaluation list of annotations make list of train and eval images

print("Creating directories...")
# make new directory if it doesn't exist
try:
    mkdir(eval_dir)
    mkdir(train_dir)
except OSError as error:
    print(error)

#import pdb; pdb.set_trace()
# convert xml annotations to csv file
print("Converting training xml to csv...")
for f in training_ann_list:
   # saves csv file in train_dir
   xml_2_csv(ann_dir, f, train_dir)

print("Converting evaluation xml to csv...")
for f in evaluation_ann_list:
   # saves csv file in eval_dir
   xml_2_csv(ann_dir, f, eval_dir)
   
train_img_list = []
eval_img_list = []

# get list of .tif files
image_files = glob(join(rgb_dir, "*.tif"))

# remove directory from file names
image_files = [basename(x) for x in image_files]

# create a training_list with names with no extension; can be used to find images
training_list = [x.split(".")[0] for x in training_ann_list]

# create a eval_list with names with no extension; can be used to find images
eval_list = [x.split(".")[0] for x in evaluation_ann_list]

# create a list of training images and a list of evaluation images
print("Separating training and evaluation images...")
train_img_list, eval_img_list, train_ann_rem, eval_ann_rem  = sep_train_eval_img(image_files, training_list, eval_list)

# remove annotations without corresponding images from training and eval annotation list
print("Cleaning training and eval annotation lists...")
clean_annotations(train_dir, eval_dir, train_ann_rem, eval_ann_rem)

# copy images to new directories
print("Copying training images...")
copy_files(rgb_dir, train_dir, train_img_list)

print("Copying annotation images...")
copy_files(rgb_dir, eval_dir, eval_img_list)

#import pdb; pdb.set_trace()
# split images larger than 500 pixels in either dimension

filtered_train_basename_list = glob(join(train_dir, "*.tif"))
filtered_eval_basename_list = glob(join(eval_dir, "*.tif"))

# get basenames
filtered_train_basename_list = [basename(x).split(".")[0] for x in filtered_train_basename_list]
filtered_eval_basename_list =  [basename(x).split(".")[0] for x in filtered_eval_basename_list]

# split large training images
print("\n")
print("Checking training images sizes...")
split_large_images(train_dir, filtered_train_basename_list)

# split large evaluation images
print("\n\n")
print("Checking evaluation image sizes...")
split_large_images(eval_dir, filtered_eval_basename_list)
