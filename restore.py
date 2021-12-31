from os.path import join
from os.path import basename
from os import mkdir
from glob import glob
from shutil import copyfile
from skimage import io
import sys
import rasterio

source_dirs = ["../training4_bak", "../evaluation4_bak"]
dest_dirs = ["training3_bak", "evaluation3_bak"]
count_csv = 0
count_png = 0
count_tif = 0

def write_geotiff(working_dir, rasterio_obj, new_fname, data):
    with rasterio.open(
            join(working_dir, new_fname),
            'w',
            driver=rasterio_obj.driver,
            height=data.shape[1],
            width=data.shape[2],
            count=data.shape[0],
            dtype=data.dtype,
            crs=rasterio_obj.crs,
            transform=rasterio_obj.transform,
    ) as dst:
        dst.write( data)
        dst.close()

for idx, dir in enumerate(source_dirs):
   print("Directory {}...".format(dir))
   all_files = glob(join(dir, "*"))

   try:
      mkdir(dest_dirs[idx])
   except OSError as error:
      print(error)

   for fle in all_files:
      fle_basename = basename(fle)
      if fle_basename.split(".")[-1] == "csv":
         copyfile(fle, join(dest_dirs[idx], fle_basename))
         print("  copying {} ...".format(fle, join(dest_dirs[idx], fle_basename)))
         count_csv += 1

      elif fle_basename.split(".")[-1] == "png":
         img = io.imread(fle)
         if "MLBS" not in fle_basename:
            io.imsave(join(dest_dirs[idx], fle_basename), img[:, :, :3][:, :, ::-1])
         else:
            io.imsave(join(dest_dirs[idx], fle_basename), img[:, :, :3])
         print("  saving {} ...".format(join(dest_dirs[idx], fle_basename)))
         count_png += 1

      
      elif fle_basename.split(".")[-1] == "tif":
         #img = io.imread(fle)
         #io.imsave(join(dest_dirs[idx], fle_basename), img[:, :, :3])        
   
         geo_tif = rasterio.open(fle)
         arr = geo_tif.read()
         arr_3chan = arr[:3, :, :]
         write_geotiff(dest_dirs[idx], geo_tif, fle_basename, arr_3chan)  
         geo_tif.close()
         
         print("  saving {} ...".format(fle, join(dest_dirs[idx], fle_basename)))
         count_tif += 1
         
      else:
         print("ERROR")
         sys.exit()

   print("\n")

print("Total files copied: {}".format(str(count_csv + count_png + count_tif)))
print("Done!")
