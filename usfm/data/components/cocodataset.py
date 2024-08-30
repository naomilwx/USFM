import os
import json

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import skimage

def load_mask(image_info, annotations):
  mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.int8)

  for ann in annotations:
    mask_type = ann['category_id']
    # Extract segmentation polygon
    for seg in ann['segmentation']:
      # Convert polygons to a binary mask and add it to the main mask
      rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
      mask_np[rr, cc] = mask_type
    return mask_np
      
class CocoSegDataset(Dataset):
  def __init__(self, folder, transforms, annotations_file='_annotations.coco.json'):
    self.transforms = transforms
    self.folder = folder
    self._update_annotations_info(f"{folder}/{annotations_file}")

  def _update_annotations_info(self, json_file):
    with open(json_file, 'r') as f:
      self.annotations_info = json.load(f)
      self.image_info = self.annotations_info['images']
      self.annotations_map = {}
      annotations = self.annotations_info['annotations']
      
      for ann in annotations:
        img_id = ann['image_id']
        if img_id not in self.annotations_map:
          self.annotations_map[img_id] = []
        self.annotations_map[img_id].append(ann)

  def __len__(self):
    return len(self.image_info)
  
  def __getitem__(self, index):
    image_info = self.image_info[index]
    image_file = f"{self.folder}/{image_info['file_name']}"
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask = load_mask(image_info, self.annotations_map[image_info['id']])
    return self.transforms(image=image, mask=mask)
   