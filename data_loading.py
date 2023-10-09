import os
from skimage import io, transform, color,img_as_ubyte
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor

def Normalization():
   return A.Compose(
       [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ])


#Dataset Loader
#-------------------Class for Training and Evaluating The Network-----------------------------#
class binary_class(Dataset):
        def __init__(self,path,data, transform=None):
            self.path = path
            self.folders = data
            self.transforms = transform
            self.histogram_equalization = A.CLAHE(p=1)
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_path = os.path.join(self.path,'images/',self.folders[idx])
            mask_path = os.path.join(self.path,'masks/',self.folders[idx])
            mask = cv2.imread(mask_path, 0)
            mask = (mask == 255).astype('uint8')
            image_id = self.folders[idx]
            histogram_equalized = self.histogram_equalization(image=img, mask=mask)
            img, mask = histogram_equalized['image'], histogram_equalized['mask']
            img = cv2.imread(image_path)[:,:,:3].astype('float32')
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
            return img, mask, image_id
#-----------------------------------------------------------------------------#

#----------------------Class for Predicting The Mask------------------------------#
class binary_class_eval(Dataset):
        def __init__(self,path,data, transform=None):
            self.path = path
            self.folders = data
            self.transforms = transform
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_path = os.path.join(self.path,'images/',self.folders[idx])
            image = cv2.imread(image_path)[:,:,:3].astype('float32')
            image_id = self.folders[idx]
            
            augmented = self.transforms(image=image)
            img = augmented['image']
            
            return img, image, image_id
        

        
