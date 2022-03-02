"""
Dataset model

https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection

During training, the model expects both the input tensors, as well as a targets (list of dictionary), containing:

boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

labels (Int64Tensor[N]): the class label for each ground-truth box

https://colab.research.google.com/github/benihime91/pytorch_retinanet/blob/master/demo.ipynb#scrollTo=0zNGhr6D7xGN

"""
import os
import pandas as pd
import cv2
import numpy as np
import torch
import my_parse as pars
from skimage import io
from torch.utils.data import Dataset
from deepforest import transforms as T
from deepforest.utilities import check_image


def get_transform(augment):
    transforms = []
    transforms.append(T.ToTensor())
    if augment:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def make_chm_mask(chm):
    _, mask = cv2.threshold(chm, 2, 1, cv2.THRESH_BINARY)  # x < 2 ==> 0; x > 2 ==> 1
    kernel = np.ones((7, 7),np.uint8)
    mask_morph = cv2.dilate(mask, kernel, iterations=1)
    return mask_morph   
 
def apply_chm_transform(image, mask):
    masked = np.transpose(image, (2, 0, 1)) * mask
    image = np.transpose(masked, (1, 2, 0))
    return masked, image

class TreeDataset(Dataset):

    def __init__(self, csv_file, root_dir, transforms, label_dict = {"Tree": 0}):
        """
        Args:
            csv_file (string): Path to a single csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            label_dict: a dictionary where keys are labels from the csv column and values are numeric labels "Tree" -> 0
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms
        self.image_names = self.annotations.image_path.unique()
        self.label_dict = label_dict

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = io.imread(img_name)
       
        if (pars.args.chm.lower() == 'true'):
            chm = image[:, :, 3]
            image = image[:, :, :3]
            mask = make_chm_mask(chm)
            masked, image = apply_chm_transform(image, mask)
            chm = torch.from_numpy(chm)
            chm = chm.type(torch.float32)
        else:
            image = image[:, :, :3]

        image = image / 255
        
        try:
            check_image(image)
        except Exception as e:
            raise Exception("dataloader failed with exception for image: {}",format(img_name))

        # select annotations
        image_annotations = self.annotations[self.annotations.image_path ==
                                             self.image_names[idx]]
        targets = {}
        targets["boxes"] = image_annotations[["xmin", "ymin", "xmax",
                                              "ymax"]].values.astype(float)

        # Labels need to be encoded? 0 or 1 indexed?, ALl tree for the moment.
        targets["labels"] = image_annotations.label.apply(
            lambda x: self.label_dict[x]).values.astype(int)

        if self.transform:
            image, targets = self.transform(image, targets)

        return self.image_names[idx], image, targets
