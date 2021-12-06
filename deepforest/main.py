# entry point for deepforest model
import os
import pandas as pd
import numpy as np
from skimage import io
import torch
from torch.nn import functional as F

from datetime import datetime
#import cv2
from torchvision.ops import boxes

import pytorch_lightning as pl
from torch import optim

from deepforest import utilities
from deepforest import dataset
from deepforest import get_data
from deepforest import model
from deepforest import predict
from deepforest import evaluate as evaluate_iou
from deepforest.logic_nn import LogicNN
import comet
from itertools import combinations

class deepforest(pl.LightningModule):
    """Class for training and predicting tree crowns in RGB images
    """

    def __init__(self, rules, rule_lambdas, pi_params, C, num_classes=1, label_dict = {"Tree":0}, batch_size = 1):
        """
        Args:
            num_classes (int): number of classes in the model
        Returns:
            self: a deepforest pytorch ligthning module
        """
        super().__init__()

        # Read config file - if a config file exists in local dir use it,
        # if not use installed.
        if os.path.exists("deepforest_config.yml"):
            config_path = "deepforest_config.yml"
        else:
            try:
                config_path = get_data("deepforest_config.yml")
            except Exception as e:
                raise ValueError(
                    "No deepforest_config.yml found either in local "
                    "directory or in installed package location. {}".format(e))

        print("Reading config file: {}".format(config_path))
        self.config = utilities.read_config(config_path)

        # release version id to flag if release is being used
        self.__release_version__ = None

        self.num_classes = num_classes
        self.create_model()
        
        #Label encoder and decoder
        if not len(label_dict) == num_classes:
            raise ValueError("label_dict {} does not match requested number of classes {}, please supply a label_dict argument {'label1':0, 'label2':1, 'label3':2 ... etc} for each label in the dataset".format(label_dict, num_classes))
        
        self.label_dict = label_dict
        self.numeric_to_label_dict = {v: k for k, v in label_dict.items()}
        self.batch_size = batch_size
        self.logic_nn = LogicNN(self.device, network=model, rules=rules, rule_lambda=rule_lambdas, C=C)
        self.pi = 0
        self.n_train_batches = -1
        self.pi_params = pi_params
        self.batch_cnt = 0
        

    def use_release(self):
        """Use the latest DeepForest model release from github and load model.
        Optionally download if release doesn't exist.
        Returns:
            model (object): A trained keras model
        """
        # Download latest model from github release
        release_tag, self.release_state_dict = utilities.use_release()
        self.model.load_state_dict(torch.load(self.release_state_dict, map_location=self.device))

        # load saved model and tag release
        self.__release_version__ = release_tag
        print("Loading pre-built model: {}".format(release_tag))

    def create_model(self):
        """Define a deepforest retinanet architecture"""
        now = datetime.now()
        curr_time = now.strftime("%H:%M:%S")
        print("Time: {}".format(curr_time))
        self.model = model.create_model(self.num_classes, self.config["nms_thresh"], self.config["score_thresh"])

    def create_trainer(self, logger=None, callbacks=None, **kwargs):
        """Create a pytorch ligthning training by reading config files
        Args:
            callbacks (list): a list of pytorch-lightning callback classes
        """

        self.trainer = pl.Trainer(logger=logger,
                                  max_epochs=self.config["train"]["epochs"],
                                  gpus=self.config["gpus"],
                                  checkpoint_callback=True,
                                  distributed_backend=self.config["distributed_backend"],
                                  fast_dev_run=self.config["train"]["fast_dev_run"],
                                  callbacks=callbacks,
                                  **kwargs)

    def save_model(self, path):
        """
        Save the trainer checkpoint in user defined path, in order to access in future
        Args:
            Path: the path located the model checkpoint

        """
        self.trainer.save_checkpoint(path)

    def load_dataset(self,
                     csv_file,
                     root_dir=None,
                     augment=False,
                     shuffle=False,
                     batch_size=1):
        """Create a tree dataset for inference
        Csv file format is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax" for the image name and bounding box position.
        Image_path is the relative filename, not absolute path, which is in the root_dir directory. One bounding box per line.

        Args:
            csv_file: path to csv file
            root_dir: directory of images. If none, uses "image_dir" in config
            augment: Whether to create a training dataset, this activates data augmentations
        Returns:
            ds: a pytorch dataset
        """

        ds = dataset.TreeDataset(csv_file=csv_file,
                                 root_dir=root_dir,
                                 transforms=dataset.get_transform(augment=augment),
                                 label_dict=self.label_dict)

        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=utilities.collate_fn,
            num_workers=self.config["workers"],
        )

        return data_loader

    def train_dataloader(self):
        """
        Train loader using the configurations
        Returns: loader

        """
        loader = self.load_dataset(csv_file=self.config["train"]["csv_file"],
                                   root_dir=self.config["train"]["root_dir"],
                                   augment=True,
                                   shuffle=False,
                                   batch_size=self.config["batch_size"])

        return loader

    def val_dataloader(self):
        """
        Create a val data loader only if specified in config
        Returns: loader or None

        """
        loader = None
        if self.config["validation"]["csv_file"] is not None:
            loader = self.load_dataset(csv_file=self.config["validation"]["csv_file"],
                                       root_dir=self.config["validation"]["root_dir"],
                                       augment=False,
                                       shuffle=False,
                                       batch_size=self.config["batch_size"])

        return loader

    def predict_image(self, image=None, path=None, return_plot=False):
        """Predict an image with a deepforest model

        Args:
            image: a numpy array of a RGB image ranged from 0-255
            path: optional path to read image from disk instead of passing image arg
            return_plot: Return image with plotted detections
        Returns:
            boxes: A pandas dataframe of predictions (Default)
            img: The input with predictions overlaid (Optional)
        """
        if isinstance(image, str):
            raise ValueError(
                "Path provided instead of image. If you want to predict an image from disk, is path ="
            )

        if path:
            if not isinstance(path, str):
                raise ValueError("Path expects a string path to image on disk")
            image = io.imread(path)

            # Load on GPU is available
        if torch.cuda.is_available:
            self.model.to(self.device)

        self.model.eval()

        # Check if GPU is available and pass image to gpu
        result = predict.predict_image(model=self.model,
                                       image=image,
                                       return_plot=return_plot,
                                       device=self.device,
                                       iou_threshold=self.config["nms_thresh"])
        
        #Set labels to character from numeric if returning boxes df
        if not return_plot:
            if not result is None:
                result["label"] = result.label.apply(lambda x: self.numeric_to_label_dict[x])
        
        return result

    def predict_file(self, csv_file, root_dir, savedir=None):
        """Create a dataset and predict entire annotation file

        Csv file format is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax" for the image name and bounding box position.
        Image_path is the relative filename, not absolute path, which is in the root_dir directory. One bounding box per line.

        Args:
            csv_file: path to csv file
            root_dir: directory of images. If none, uses "image_dir" in config
            savedir: Optional. Directory to save image plots.
        Returns:
            df: pandas dataframe with bounding boxes, label and scores for each image in the csv file
        """
        self.model.eval()
        result = predict.predict_file(model=self.model,
                                      csv_file=csv_file,
                                      root_dir=root_dir,
                                      savedir=savedir,
                                      device=self.device,
                                      iou_threshold=self.config["nms_thresh"])

        #Set labels to character from numeric
        result["label"] = result.label.apply(lambda x: self.numeric_to_label_dict[x])
            
        return result

    def predict_tile(self,
                     raster_path=None,
                     image=None,
                     patch_size=400,
                     patch_overlap=0.05,
                     iou_threshold=0.15,
                     return_plot=False,
                     use_soft_nms=False,
                     sigma=0.5,
                     thresh=0.001):
        """For images too large to input into the model, predict_tile cuts the
        image into overlapping windows, predicts trees on each window and
        reassambles into a single array.

        Args:
            raster_path: Path to image on disk
            image (array): Numpy image array in BGR channel order
                following openCV convention
            patch_size: patch size default400,
            patch_overlap: patch overlap default 0.15,
            iou_threshold: Minimum iou overlap among predictions between
                windows to be suppressed. Defaults to 0.5.
                Lower values suppress more boxes at edges.
            return_plot: Should the image be returned with the predictions drawn?
            use_soft_nms: whether to perform Gaussian Soft NMS or not, if false, default perform NMS.
            sigma: variance of Gaussian function used in Gaussian Soft NMS
            thresh: the score thresh used to filter bboxes after soft-nms performed

        Returns:
            boxes (array): if return_plot, an image.
            Otherwise a numpy array of predicted bounding boxes, scores and labels
        """

        self.model.eval()

        result = predict.predict_tile(model=self.model,
                                      raster_path=raster_path,
                                      image=image,
                                      patch_size=patch_size,
                                      patch_overlap=patch_overlap,
                                      iou_threshold=iou_threshold,
                                      return_plot=return_plot,
                                      use_soft_nms=use_soft_nms,
                                      sigma=sigma,
                                      thresh=thresh,
                                      device=self.device)

        #Set labels to character from numeric if returning boxes df
        if not return_plot:
            result["label"] = result.label.apply(lambda x: self.numeric_to_label_dict[x])
            
        return result

    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        path: tuple
        images: tuple of images.  image shape is [C, H, W].  image values [0, 1].  channel order RGB.
        targets: tuple of dictionaries.  dictinary has keys 'boxes' and 'labels'. values are tensors of torch.float64
        """

        path, images, targets = batch
        curr_iter = self.batch_cnt * 1. / self.config["train"]["n_train_batches"]
        #self.batch_cnt += 1
        if self.global_step > self.config['train']["beg_incr_pi"]:
           self.batch_cnt += 1

        #calculate pi
        pi = self.get_pi(curr_iter)

        # make sure model is in training mode
        self.model.train()
        loss_dict = self.model.forward(images, targets)

        # put model in eval mode
        self.model.eval()

        # preds a list of dictionaries
        # one dictionary per image
        # each dictionary has keys 'boxes', 'scores', and 'labels'
        # each value is a tensor
        preds = self.model.forward(images)          #targets must be included in training mode

        # get special features
        eng_fea = []
        for img, img_dict in zip(images, preds):
            # generate special features
            mask = self.has_competition(images, preds)
            mask = torch.tensor(mask, dtype=torch.float, requires_grad=True).reshape(-1, 1)
            mask = mask.to(self.device)
            eng_fea = self.bbox_2big(images, preds)
            eng_fea = (mask, eng_fea)
        
        if (len(preds[0]['scores']) > 0):
            q_y_pred = self.logic_nn.predict(preds[0]['scores'], images, [eng_fea]).to(self.device)
            huLoss = F.binary_cross_entropy(torch.tensor(1.) - preds[0]['scores'].float(), q_y_pred.float())
        else:
            huLoss = torch.tensor(0., requires_grad=True).to(self.device)

        losses = (1 - pi) * sum([loss for loss in loss_dict.values()]) + pi * (huLoss + loss_dict['bbox_regression'])

        self.log('pi', pi, prog_bar=True, on_step=True)
        self.log('num_preds', len(preds[0]['labels']), prog_bar=True, on_step=True)
        self.log('num_comp', len(eng_fea), prog_bar=True, on_step=True)
        self.log('hu_loss', huLoss, prog_bar=True, on_step=True)

        with comet.experiment.train():
            comet.experiment.log_metric("pi", pi)
            comet.experiment.log_metric("hu_loss", huLoss)

        return losses

    def validation_step1(self, batch, batch_idx):
        """Train on a loaded dataset

        """
        path, images, targets = batch

        with torch.no_grad():
            self.model.train()
            loss_dict = self.model.forward(images, targets)

            # sum of regression and classification loss
            losses = sum([loss for loss in loss_dict.values()])

            # Log loss
            for key, value in loss_dict.items():
                self.log("val_{}".format(key), value, prog_bar=True, on_epoch=True)
                with comet.experiment.validate():
                    comet.experiment.log_metric("val_{}".format(key), value, step=batch_idx)

        self.log("val_loss", losses)                
        return losses

    def validation_end1(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        comet_logs = {'val_loss': avg_loss}

        with comet.experiment.validate():
            comet.experiment.log_metric("val_loss", comet_logs)
        return {'avg_val_loss': avg_loss, 'log': comet_logs}

    def configure_optimizers(self):
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.config["train"]["lr"],
                                   momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.1,
                                                                    patience=10,
                                                                    verbose=True,
                                                                    threshold=0.0001,
                                                                    threshold_mode='rel',
                                                                    cooldown=0,
                                                                    min_lr=0,
                                                                    eps=1e-08)
        return self.optimizer

    def evaluate(self,
                 csv_file,
                 root_dir,
                 iou_threshold=None,
                 show_plot=False,
                 savedir=None):
        """Compute intersection-over-union and precision/recall for a given iou_threshold

        Args:
            df: a pandas-type dataframe (geopandas is fine) with columns "name","xmin","ymin","xmax","ymax","label", each box in a row
            root_dir: location of files in the dataframe 'name' column.
            iou_threshold: float [0,1] intersection-over-union union between annotation and prediction to be scored true positive
            show_plot: open a blocking matplotlib window to show plot and annotations, useful for debugging.
            savedir: optional path dir to save evaluation images
        Returns:
            results: dict of ("results", "precision", "recall") for a given threshold
        """
        self.model.eval()

        if not self.device.type == "cpu":
            self.model = self.model.to(self.device)

        predictions = predict.predict_file(self,
                                           model=self.model,
                                           csv_file=csv_file,
                                           root_dir=root_dir,
                                           savedir=savedir,
                                           device=self.device,
                                           iou_threshold=self.config["nms_thresh"])

        predictions["label"] = predictions.label.apply(lambda x: self.numeric_to_label_dict[x])
        ground_df = pd.read_csv(csv_file)

        # if no arg for iou_threshold, set as config
        if iou_threshold is None:
            iou_threshold = self.config["validation"]["iou_threshold"]

        results = evaluate_iou.evaluate(predictions=predictions,
                                        ground_df=ground_df,
                                        root_dir=root_dir,
                                        iou_threshold=iou_threshold,
                                        show_plot=show_plot,
                                        savedir=savedir)

        return results

    def get_pi(self, cur_iter, pi=None):
        """ exponential decay: pi_t = max{1 - k^t, lb} """
        k, lb = self.pi_params[0], self.pi_params[1]
        pi = 1. - max([k ** cur_iter, lb])
        return pi

    def translate_box(self, box):
        box[:, 2] = box[:, 2] - box[:, 0]
        box[:, 0] = box[:, 0] - box[:, 0]

        box[:, 3] = box[:, 3] - box[:, 1]
        box[:, 1] = box[:, 1] - box[:, 1]

        return box

    def has_competition(self, images, preds):
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
            if dist <= bb_rads[pair[0]] + bb_rads[pair[1]]:
                trees_competing[pair[0]] = 1
                trees_competing[pair[1]] = 1

        #res = torch.where(torch.tensor(trees_competing) == 1.)[0].tolist()

        return trees_competing

    def scaleBB(self, coords, scaleX, scaleY, device):
        # takes in bounding box coordinates as [x1, y1, x2, y2] and returns a scaled bounding box with the same centroid
        coords2 = coords.view(-1, 2).to(device)

        # transpose coordinates and make them homogenous
        coordsMatrix = torch.vstack([coords2.T, torch.ones([1, coords2.shape[0]], requires_grad=True).to(device)]).to(device)

        # calculate coordinates of centroid
        # centroid = np.mean(coordsNp[:-1, :], axis=0)
        centroid = torch.mean(coords2, 0)

        # transform to translate to origin, scale, and translate back to centroid
        trans = torch.tensor(
            [[scaleX, 0, centroid[0] * (1 - scaleX)],
             [0, scaleY, centroid[1] * (1 - scaleY)],
             [0, 0, 1]]).to(device)

        # multiply matrices
        res = torch.mm(trans, coordsMatrix)[:2, :].T
        res = res.contiguous()

        # return data to original format of a list of tuples
        res = res.view(1, 4)

        return res


    def bbox_2big(self, images, preds):
        """
        Description: Find predictions where the bounding box is infeasibily large
        images - list of images in batch
        preds - list of prediction dictionaries withkeys boxes, scores, and labels
        """   

        sigma = torch.nn.Sigmoid()

        # calculate the area of each bounding box
        x_len = preds[0]['boxes'][:, 2] - preds[0]['boxes'][:, 0]
        y_len = preds[0]['boxes'][:, 3] - preds[0]['boxes'][:, 1]
        bb_area = x_len * y_len

        #return the index of bboxes with areas greater than X
        res = sigma(0.5 * (400 - bb_area))
        return res
