import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from data_loading import binary_class_eval
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
from torchmetrics import Accuracy, Precision, Recall
from torchmetrics import F1Score as F1
import argparse
import time
import pandas as pd
import cv2
import os
from skimage import io, transform
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import xlwt
from xlwt import Workbook


"""
Class for wrapping the output of model to track gradients for GradCAM
"""
class SemanticTarget:
        def __init__(self, category, mask):
            self.category = category
            self.mask = torch.from_numpy(mask)
            if torch.cuda.is_available():
                self.mask = self.mask.cuda()
            self.loss = torch.nn.BCEWithLogitsLoss()

        def __call__(self, model_output):
            return self.loss(model_output[0, :, :], self.mask)


class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU

class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice

def get_transform():
   return A.Compose(
       [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/',type=str, help='the path of dataset')
    parser.add_argument('--csvfile', default='src/test_train_data.csv',type=str, help='two columns [image_id,category(train/test)]')
    parser.add_argument('--model',default='save_models/epoch_last.pth', type=str, help='the path of model')
    parser.add_argument('--debug',default=True, type=bool, help='plot mask')
    args = parser.parse_args()
    
    os.makedirs('debug/',exist_ok=True)
    
    df = pd.read_csv(args.csvfile)
    df = df[df.category=='test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_files = list(df.image_id)
    test_dataset = binary_class_eval(args.dataset,test_files, get_transform())
    model = torch.load(args.model)

    model = model.cuda()
    target_layers = []
    for name, layer in model.named_modules():
        target_layers.append(layer)
        break

    time_cost = []
    pseudo_mask = np.ones((224, 224), dtype=np.float32)
    targets = [SemanticTarget(1, pseudo_mask)]

    since = time.time()
    """
    *************Excel sheet for Classification*******************
    # for excel
    #wb = Workbook()
    #sheet1 = wb.add_sheet('Sheet 1')
    #sheet1.write(0, 0, 'Image Name')
    #sheet1.write(0,1,'Predicted Class Label')
    #cnt = 1
    #bleed = 0
    #nonbleed = 0
    ***************************************************************
    """

    #---------------------storing the endoscopy images--------------------------#
    for image_id in test_files:
        img = cv2.imread(os.path.join(args.dataset, 'images', image_id))
        img = cv2.resize(img, (224,224))
        img_id = list(image_id.split('.'))[0]
        cv2.imwrite(f'debug/{img_id}.png',img) # change the directory here
    #---------------------------------------------------------------------------#
    
    for img, real_image, img_id in test_dataset:
        
        img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).cuda()           
        torch.cuda.synchronize()
        start = time.time()
        pred = model(img)
        torch.cuda.synchronize()
        end = time.time()
        time_cost.append(end-start)

        pred = torch.sigmoid(pred)

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        
        #-------------------writing the classification to Excel Sheet--------------------#
        #if len(torch.unique(pred)) == 1:
            #nonbleed += 1
            #sheet1.write(cnt,0,img_id)
            #sheet1.write(cnt,1,'Non-Bleeding')
        #else:
            #bleed += 1            
            #sheet1.write(cnt,0,img_id)
            #sheet1.write(cnt,1,'Bleeding')
        #cnt += 1
        #--------------------------------------------------------------------------------#
        
        #-----------------saving the attention map of neural network---------------------#
        pred_draw = pred.clone().detach()
        real_image = (real_image/255).astype(np.float32)
        with GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam = cam(input_tensor=img, targets=targets)[0, :]
            cam_image = show_cam_on_image(real_image, grayscale_cam, use_rgb=True)
        cv2.imwrite(f'debug/{img_id}_attention.png', cam_image) # change the directory here
        #--------------------------------------------------------------------------------#
        
        #-----------------printing the predicted mask------------------------------------#
        if args.debug:
            img_id = list(img_id.split('.'))[0]
            img_numpy = pred_draw.cpu().detach().numpy()[0][0]
            img_numpy[img_numpy==1] = 255 
            cv2.imwrite(f'debug/{img_id}_pred.png',img_numpy) # change the directory here
        #--------------------------------------------------------------------------------#
            
        pred = pred.view(-1)
        torch.cuda.empty_cache()
            
    time_elapsed = time.time() - since
    """********Saving The Excel Sheet**********"""
    #wb.save('./ans.xls')
    #print(bleed,nonbleed)
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


