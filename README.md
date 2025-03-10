# AutoWCEBleed-Challenge-DeepNet
This a submission to the contest conducted by MISAHUB titled AutoWCEBleed-Challenge

# Auto-WCEBleedGen Challenge
Automatic Classification, Segmentation and Detection of Bleeding and Non-Bleeding Frames in a Wireless Capsule Endoscopy Images.

## Achieved Evaluation Metrics 

### Classification Metrics

|               |          |
|---------------|----------|
| Accuracy      | 0.9647   |
| Precision     | 0.9513   |
| Recall        | 0.9841   |
| F1 Score      | 0.9674   |

### Segmentation Metrics
|                           |   Mean | Std    |
|---------------------------|--------|--------|
| Mean IoU (threshold 0.5)  | 0.7492 | 0.3098 |
| Average Precision         | 0.3723 | 0.4096 |
| Mean Average Precision    | 0.3731 |    -   |


## Some Predicted Image of Validation Dataset

    Image                Ground Truth           Prediction              CAM plot    
**1)**<img src="best_predict_on_val/0.png" alt="OriginalImage" width="150"/> <img src="best_predict_on_val/0_ground.png" alt="Ground Truth" width="150"/> <img src="best_predict_on_val/0_pred.png" alt="Bleeding Prediction" width="150"/> <img src="best_predict_on_val/0.png_attention.png" alt="CAM_PLOT" width="150"/> 


**2)**<img src="best_predict_on_val/3.png" alt="OriginalImage" width="150"/> <img src="best_predict_on_val/3_ground.png" alt="Ground Truth" width="150"/> <img src="best_predict_on_val/3_pred.png" alt="Bleeding Prediction" width="150"/> <img src="best_predict_on_val/3.png_attention.png" alt="CAM_PLOT" width="150"/> 


**3)**<img src="best_predict_on_val/45.png" alt="OriginalImage" width="150"/> <img src="best_predict_on_val/45_ground.png" alt="Ground Truth" width="150"/> <img src="best_predict_on_val/45_pred.png" alt="Bleeding Prediction" width="150"/> <img src="best_predict_on_val/45.png_attention.png" alt="CAM_PLOT" width="150"/> 


**4)**<img src="best_predict_on_val/50.png" alt="OriginalImage" width="150"/> <img src="best_predict_on_val/50_ground.png" alt="Ground Truth" width="150"/> <img src="best_predict_on_val/50_pred.png" alt="Bleeding Prediction" width="150"/> <img src="best_predict_on_val/50.png_attention.png" alt="CAM_PLOT" width="150"/> 


**5)**<img src="best_predict_on_val/65.png" alt="OriginalImage" width="150"/> <img src="best_predict_on_val/65_ground.png" alt="Ground Truth" width="150"/> <img src="best_predict_on_val/65_pred.png" alt="Bleeding Prediction" width="150"/> <img src="best_predict_on_val/65.png_attention.png" alt="CAM_PLOT" width="150"/> 


**6)**<img src="best_predict_on_val/71.png" alt="OriginalImage" width="150"/> <img src="best_predict_on_val/71_ground.png" alt="Ground Truth" width="150"/> <img src="best_predict_on_val/71_pred.png" alt="Bleeding Prediction" width="150"/> <img src="best_predict_on_val/71.png_attention.png" alt="CAM_PLOT" width="150"/> 


**7)**<img src="best_predict_on_val/72.png" alt="OriginalImage" width="150"/> <img src="best_predict_on_val/72_ground.png" alt="Ground Truth" width="150"/> <img src="best_predict_on_val/72_pred.png" alt="Bleeding Prediction" width="150"/> <img src="best_predict_on_val/72.png_attention.png" alt="CAM_PLOT" width="150"/>


**8)**<img src="best_predict_on_val/140.png" alt="OriginalImage" width="150"/> <img src="best_predict_on_val/140_ground.png" alt="Ground Truth" width="150"/> <img src="best_predict_on_val/140_pred.png" alt="Bleeding Prediction" width="150"/> <img src="best_predict_on_val/140.png_attention.png" alt="CAM_PLOT" width="150"/> 


**9)**<img src="best_predict_on_val/183.png" alt="OriginalImage" width="150"/> <img src="best_predict_on_val/183_ground.png" alt="Ground Truth" width="150"/> <img src="best_predict_on_val/183_pred.png" alt="Bleeding Prediction" width="150"/> <img src="best_predict_on_val/183.png_attention.png" alt="CAM_PLOT" width="150"/> 


**10)**<img src="best_predict_on_val/208.png" alt="OriginalImage" width="150"/> <img src="best_predict_on_val/208_ground.png" alt="Ground Truth" width="150"/> <img src="best_predict_on_val/208_pred.png" alt="Bleeding Prediction" width="150"/> <img src="best_predict_on_val/208.png_attention.png" alt="CAM_PLOT" width="150"/> 


**11)**<img src="best_predict_on_val/268.png" alt="OriginalImage" width="150"/> <img src="best_predict_on_val/268_ground.png" alt="Ground Truth" width="150"/> <img src="best_predict_on_val/268_pred.png" alt="Bleeding Prediction" width="150"/> <img src="best_predict_on_val/268.png_attention.png" alt="CAM_PLOT" width="150"/> 


## Some Predicted Images of Test 1 Dataset
    Image                     Prediction                  CAM plot
**1)**<img src="best_predict_on_test1/A0036.png" alt="OriginalImage" width="200"/> <img src="best_predict_on_test1/A0036_pred.png" alt="Bleeding Prediction" width="200"/> <img src="best_predict_on_test1/A0036.png_attention.png" alt="CAM_PLOT" width="200"/>


**2)**<img src="best_predict_on_test1/A0037.png" alt="OriginalImage" width="200"/> <img src="best_predict_on_test1/A0037_pred.png" alt="Bleeding Prediction" width="200"/> <img src="best_predict_on_test1/A0037.png_attention.png" alt="CAM_PLOT" width="200"/>


**3)**<img src="best_predict_on_test1/A0038.png" alt="OriginalImage" width="200"/> <img src="best_predict_on_test1/A0038_pred.png" alt="Bleeding Prediction" width="200"/> <img src="best_predict_on_test1/A0038.png_attention.png" alt="CAM_PLOT" width="200"/>


**4)**<img src="best_predict_on_test1/A0039.png" alt="OriginalImage" width="200"/> <img src="best_predict_on_test1/A0039_pred.png" alt="Bleeding Prediction" width="200"/> <img src="best_predict_on_test1/A0039.png_attention.png" alt="CAM_PLOT" width="200"/>


**5)**<img src="best_predict_on_test1/A0047.png" alt="OriginalImage" width="200"/> <img src="best_predict_on_test1/A0047_pred.png" alt="Bleeding Prediction" width="200"/> <img src="best_predict_on_test1/A0047.png_attention.png" alt="CAM_PLOT" width="200"/>


## Some Predicted Images of Test 2 Dataset
    Image                     Prediction                  CAM plot
**1)**<img src="best_predict_on_test2/A0061.png" alt="OriginalImage" width="200"/> <img src="best_predict_on_test2/A0061_pred.png" alt="Bleeding Prediction" width="200"/> <img src="best_predict_on_test2/A0061.png_attention.png" alt="CAM_PLOT" width="200"/>


**2)**<img src="best_predict_on_test2/A0173.png" alt="OriginalImage" width="200"/> <img src="best_predict_on_test2/A0173_pred.png" alt="Bleeding Prediction" width="200"/> <img src="best_predict_on_test2/A0173.png_attention.png" alt="CAM_PLOT" width="200"/>


**3)**<img src="best_predict_on_test2/A0205.png" alt="OriginalImage" width="200"/> <img src="best_predict_on_test2/A0205_pred.png" alt="Bleeding Prediction" width="200"/> <img src="best_predict_on_test2/A0205.png_attention.png" alt="CAM_PLOT" width="200"/>


**4)**<img src="best_predict_on_test2/A0298.png" alt="OriginalImage" width="200"/> <img src="best_predict_on_test2/A0298_pred.png" alt="Bleeding Prediction" width="200"/> <img src="best_predict_on_test2/A0298.png_attention.png" alt="CAM_PLOT" width="200"/>


**5)**<img src="best_predict_on_test2/A0283.png" alt="OriginalImage" width="200"/> <img src="best_predict_on_test2/A0283_pred.png" alt="Bleeding Prediction" width="200"/> <img src="best_predict_on_test2/A0283.png_attention.png" alt="CAM_PLOT" width="200"/>


**6)**<img src="best_predict_on_test2/A0408.png" alt="OriginalImage" width="200"/> <img src="best_predict_on_test2/A0408_pred.png" alt="Bleeding Prediction" width="200"/> <img src="best_predict_on_test2/A0408.png_attention.png" alt="CAM_PLOT" width="200"/>

## Dataset
To apply the model to a custom dataset, the data tree should be constructed as follows:
``` 
    ├── data
          ├── images
                ├── image_1.png
                ├── image_2.png
                ├── image_n.png
          ├── masks
                ├── image_1.png
                ├── image_2.png
                ├── image_n.png
```
## CSV generation 
```
python data_split_csv.py --dataset your/data/path --size 0.9 
```

## Training

**1) Prepare the CSV file**
```
python data_split_csv.py --dataset [path_to_dataset] --size 0.9 \# size of train split
```
**2) Use The Generated CSV file for training**
```
python train.py --dataset [path_to_images] --csvfile [csv file containg train and test split] --loss dice --batch 8 --lr 0.001 --epoch 150
```

## Evaluation
```
python evaluation_script.py --dataset [path_to_dataset] --csvfile [csv file containing train and test split] --model [path_to_the_model]
```

## Prediction

Results will be saved to debug directory will be created if doesn't exist
```
python prediction_script.py --dataset [path_to_root_directory] --csvfile [file containing path to images] --model [path_to_the_model]
```

## Team Members
```
Dr Prerana Mukherjee
Paladugu Sai Sivakesh
Vemula Darshan
```
## Acknowledgement
* The codes are modified from [DCSAU-Net](https://github.com/xq141839/DCSAU-Net)
* Aproximately 1000 images were also taken from the following [dataset](https://rdm.inesctec.pt/dataset/nis-2018-003) for further improving the model
## Results
We are proud to share that we have achieved **2nd Rank** in the **AutoWCEBleedGen Vesion V1 Challenge**.

