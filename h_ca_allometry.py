import pandas as pd
import numpy as np
import torch
import cv2
from os.path import join
from itertools import combinations

def has_competition(preds):
    """
    Description: finds trees with touching or intersecting crowns
    images - list of images in batch
    preds - list of prediction dictionaries with keys boxes, scores, and labels
    trees_competing - list of competing trees; 0 => not competing, 1 => competing
    """

    trees_competing = [0] * preds[0]['boxes'].shape[0]

    # dist = L2 norm of difference
    # produces a tensor representing the radius of each bounding circle that contains the bounding box
    bb_rads = torch.linalg.norm(preds[0]['boxes'][:, :2] - preds[0]['boxes'][:, 2:], ord=2, dim=1) / 2.

    # calculate the centroid of each bounding box
    bb_centroid = (preds[0]['boxes'][:, :2] + preds[0]['boxes'][:, 2:]) / 2.

    # calculate distance between each set of bounding box centers
    for pair in combinations(range(0, preds[0]['boxes'].shape[0]), 2):
        dist = torch.linalg.norm(bb_centroid[pair[0]] - bb_centroid[pair[1]], ord = 2)

        #if dist between two BB centroids is less than or equal to the sum of the radii of their crown bounding cirlces, then consider trees to be touching
        if dist <= (bb_rads[pair[0]] + bb_rads[pair[1]]):
            trees_competing[pair[0]] = 1
            trees_competing[pair[1]] = 1

    res = torch.where(torch.tensor(trees_competing) == 1.)[0].tolist()
    return res


train_dir = "./training4"
res = 0.1  # 0.1 m/pixel

fname = join(train_dir, "NIWO-train.csv")
print("\n\n\n")
print("Reading csv...")
df = pd.read_csv(fname)

print("Calculating area...")
df['area'] = ((df['xmax'] - df['xmin']) * res) * ((df['ymax'] - df['ymin']) * res)

ht_list = []

print("Getting heights...")
for idx, row in df.iterrows():
   img = cv2.imread(join(train_dir, row['image_path']), cv2.IMREAD_UNCHANGED)
   
   ht = np.max(img[row['ymin']:row['ymax'], row['xmin']:row['xmax'], 3])
   ht_list.append(ht)

df['ht'] = ht_list

img_groups = df.groupby("image_path") 
#img_groups.get_group("2018_NIWO_2_450000_4426000_image_crop_44.png")


indices_to_keep = []
for gname, group in img_groups:
   tensr = torch.from_numpy(group.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']].to_numpy())
   tensr = tensr.type(torch.float)
   dict = {"boxes": tensr, 'scores': [0] * len(tensr), "labels": [0] * len(tensr)}
   boxes_to_keep = has_competition([dict])
   indices_to_keep += group.iloc[boxes_to_keep, :].index.to_list()

comp_df = df.iloc[indices_to_keep, :]
print("Saving csv...")
comp_df.to_csv("area_ht.csv", header=True, index=False)

print("Done!")
