import os
import json

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import skimage

# Dataset has a folder structure of <class_label>/image/<id>.jpg, <class_label>/segmentation/<class>/*.json
class CustomClassificaitonDataset(Dataset):
  def __init__(self, root_dir, infos, label_map, transforms=None):
    self.transforms = transforms
    self.image_infos = []
    self.label_map = label_map
    for id, info in infos.items():
      for label, details in info.items():
        self.image_infos.append({
          'pid': id,
          'filepath': f"{root_dir}/{details['relpath']}",
          'label': label
        })

  def __len__(self):
    return len(self.image_infos)
  
  def __getitem__(self, index):
    info = self.image_infos[index]
    image = cv2.imread(info['filepath'])
    if self.transforms is not None:
      image = self.transforms(image)
    return image, self.label_map[info['label']]
  

def load_mask(height, width, segmentations, mask_map):
  # outline, liver, mass
  mask_np = np.zeros((height, width), dtype=np.int8)
  for label, segs in segmentations.items():
    mask = mask_map[label]
    segs = np.array(segs)
    rr, cc = skimage.draw.polygon(segs[:, 1], segs[:, 0], mask_np.shape)
    mask_np[rr, cc] = np.maximum(mask_np[rr, cc], mask)
  return mask_np
 
class CustomSegmentationDataset(Dataset):
  def __init__(self, root_dir, infos, mask_map, transforms=None):
    self.transforms = transforms
    self.mask_map = mask_map
    self.image_infos = []
    for id, info in infos.items():
      for label, details in info.items():
        self.image_infos.append({
          'pid': id,
          'filepath': f"{root_dir}/{details['relpath']}",
          'segmentations': details['segmentations'],
          'label': label
        })

  def __len__(self):
    return len(self.image_infos)
  
  def __getitem__(self, index):
    info = self.image_infos[index]
    image = cv2.imread(info['filepath'])
    masks = load_mask(image.shape[0], image.shape[1], info['segmentations'], self.mask_map)
    if self.transforms is None:
      return image, masks
    return self.transforms(image, masks)