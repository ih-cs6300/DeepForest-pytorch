from torchvision.ops import boxes
import torch


box1 = torch.tensor([5, 10, 15, 30]).unsqueeze(0)
box2 = torch.tensor([100, 35, 120, 45]).unsqueeze(0)
box2 = box1

def translate_box(box):
   box[:, 2] = box[:, 2] - box[:, 0]
   box[:, 0] = box[:, 0] - box[:, 0]

   box[:, 3] = box[:, 3] - box[:, 1]
   box[:, 1] = box[:, 1] - box[:, 1]

   return box

box1 = translate_box(box1)
box2 = translate_box(box2)


print(boxes.box_iou(box1, box2))
 
