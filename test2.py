import cv2
import torch

img = cv2.imread("./train_data_folder/2019_YELL_2_528000_4978000_image_crop2_10.png")
img = img.astype(np.float32)
normImg = np.zeros(img.shape)
img2 = cv2.normalize(img,  normImg, 0, 1, cv2.NORM_MINMAX)
img2 = torch.tensor(img2)
img2 = img2.permute(2, 0, 1)

m.model.eval()
m.model.forward([img2])
