# Model
import torchvision
import numpy as np
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.retinanet import AnchorGenerator
import torch

def load_backbone():
    """A torch vision retinanet model"""
    #backbone = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    backbone = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)

    # load the model onto the computation device
    return backbone


def create_anchor_generator(sizes=((8, 16, 32, 64, 128, 256, 400),),
                            aspect_ratios=((0.5, 1.0, 2.0),)):
    """
    Create anchor box generator as a function of sizes and aspect ratios
    Documented https://github.com/pytorch/vision/blob/67b25288ca202d027e8b06e17111f1bcebd2046c/torchvision/models/detection/anchor_utils.py#L9
    let's make the network generate 5 x 3 anchors per spatial
    location, with 5 different sizes and 3 different aspect
    ratios. We have a Tuple[Tuple[int]] because each feature
    map could potentially have different sizes and
    aspect ratios
    Args:
        sizes:
        aspect_ratios:

    Returns: anchor_generator, a pytorch module

    """
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

    return anchor_generator


def create_model(num_classes, nms_thresh, score_thresh):
    """Create a retinanet model
    Args:
        num_classes (int): number of classes in the model
        nms_thresh (float): non-max suppression threshold for intersection-over-union [0,1]
        score_thresh (float): minimum prediction score to keep during prediction  [0,1]
    Returns:
        model: a pytorch nn module
    """
    backbone = load_backbone()

    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = (tuple(np.linspace(0.25, 2.5, 12)),) * len(anchor_sizes)
    anchor_generator = create_anchor_generator(anchor_sizes, aspect_ratios)

    
    model = RetinaNet(backbone.backbone, num_classes=num_classes)
    #model = RetinaNet(backbone.backbone, num_classes=num_classes, anchor_generator=anchor_generator)

    in_channels = model.head.classification_head.cls_logits.in_channels
    out_channels = model.head.classification_head.cls_logits.out_channels
    kernel_size = model.head.classification_head.cls_logits.in_channels
    stride = model.head.classification_head.cls_logits.stride
    padding = model.head.classification_head.cls_logits.padding
    #model.head.classification_head.cls_logits = torch.nn.Conv2d(in_channels, 36, kernel_size=kernel_size, stride=stride, padding=padding)

    in_channels = model.head.regression_head.bbox_reg.in_channels
    kernel_size = model.head.regression_head.bbox_reg.kernel_size
    stride = model.head.regression_head.bbox_reg.stride
    padding = model.head.regression_head.bbox_reg.padding
    #model.head.regression_head.bbox_reg = torch.nn.Conv2d(in_channels, 144, kernel_size=kernel_size, stride=stride, padding=padding)

    model.nms_thresh = nms_thresh
    model.score_thresh = score_thresh

    # Optionally allow anchor generator parameters to be created here
    # https://pytorch.org/vision/stable/_modules/torchvision/models/detection/retinanet.html

    return model
