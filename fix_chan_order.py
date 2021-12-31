# this script attempts to fix the channel order error in SJER, NIWO, and TEAK rgb+chm files in directory 
# evaluation4.  The error occurs becomes the rgb raster is sometimes saved as BGR rather than RGB.
# This reduces performance.  use preprocess4.py to generate the rgb+chm files correctly.

from glob import glob
from os.path import join
from os.path import basename
from skimage import io
import re
from os import system
import numpy as np

dir2cmp_tr = "training3_bak"
dir2cmp_ev = "evaluation3_bak"

dir_list = [("training4", "tr"), ("evaluation4", "ev")]
regex_tr = r'^2018_(NIWO|SJER|TEAK)_.+\.(png|tif)'
regex_ev = r'^(NIWO|TEAK|SJER)_.{3}_201\d\.(tif|png)'

def fix_img(direct, img1, img2):
   #cmp img1 and img2
   all_zero = np.all(img1 - img2[:, :, :3] == 0)

   if not all_zero:
      print("      Bad image...")
      # reverse channel order: rgb to bgr or bgr to rgb
      temp_img = img2[:, :, :3][:, :, ::-1]
      
      all_zero = np.all((img1 - temp_img[:, :, :3]) == 0)
      if (all_zero):
         # replace original img2 with flipped channels
         img2[:, :, :3] = temp_img
            
         # store img2
         print("      Flipping image and saving...")
         io.imsave(join(direct, img_name), img2)
   else:
      print("      Image ok") 


#########################################################################################
# driver code
system("clear")
for direct in dir_list:
   print("Checking directory {}...".format(direct[0]))
   # look for training images
   all_list = glob(join(direct[0], "*.*"))
   all_list = [basename(x) for x in all_list]

   if direct[1] == "tr":
      cleaned_list = [f for f in all_list if re.search(regex_tr, f)]
   elif direct[1] == "ev":
      cleaned_list = [f for f in all_list if re.search(regex_ev, f)]
   else:
      cleaned_list = []
 
   # make sure list not empty
   assert len(cleaned_list) > 0, "Print error 1"
 
   print("  Checking {} files...".format(len(cleaned_list)))

   for img_name in cleaned_list:
      print("    {}".format(img_name))
      if direct[1] == "tr":
         drcty = dir2cmp_tr
      elif direct[1] == "ev":
         drcty = dir2cmp_ev
      else:
         drcty = ""

      assert len(drcty) > 0, "Error 2"

      img1 = io.imread(join(drcty, img_name))
      img2 = io.imread(join(direct[0], img_name))
      fix_img(direct[0], img1, img2)      
   print("")

print("Done!")
