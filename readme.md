# kaggle-understanding-clouds
Code for 3rd place solution in Kaggle Understanding Clouds from Satellite Images Challenge.

To read the brief description of the solution, please, refer to [the Kaggle post](https://www.kaggle.com/c/understanding_cloud_organization/discussion/117949). If you run into any trouble with the setup/code or have any questions please contact me at davidcao1991@gmail.com

## Archive Contents
```
.
./input                          : images and .csv files for mask labels
./ouput                          : trained models and predictions
./kaggle-cloud-organization      : code
```
## Hardware:
* Ubuntu 16.04 LTS (>=512 GB boot disk)
* intel Xeon Gold 6130
* 1 x NVIDIA Tesla V100 32GB

## Software:
* Python 3.7.3
* CUDA 10.1
* cuddn 7602
* nvidia drivers v418.67

python packages are detailed separately in `requirements.txt`

## Data setup:
Assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed, below are the shell commands used in each step, as run from the top level directory.
```
$ mkdir -p input
$ cd input
$ kaggle competitions download -c understanding_cloud_organization
$ unzip train_images_zip -d ./images
$ unzip test_images_zip -d ./images
```
Please make sure the unzipped train and test images are in the same folder named 'images'

## Data proprecessing
This will convert the raw images to 384x576 and build .csv file for 5-fold info. The 5-fold info .csv is already included in ./kaggle-cloud-organization/files/
```
$ cd kaggle-cloud-organization
$ python preprocessing_images.py
$ python make_folds.py
```

## Model build
There are 4 steps to reproduce the model from scratch. Shell command to run each step is below.
1. Train seg1 model and predict.
```
$ bash ./kaggle-cloud-organization/run_seg1.sh
```
2. Train seg2 models and predict.
```
$ bash ./kaggle-cloud-organization/run_seg2.sh
```
3. Train cls model and predict.
```
$ bash ./kaggle-cloud-organization/run_cls.sh
```
4. Ensemble the final predictions
```
$ cd output
$ mkdir -p b5-Unet-inception-FPN-b7-Unet-b7-FPN-b7-FPNPL
$ mkdir -p ensemble
$ cd ..
$ python ./kaggle-cloud-organization/mask-ensemble-5fold.py
$ python ./kaggle-cloud-organization/2-stage-ensemble-5fold.py
```
The final prediction file can be found as ./output/ensemble/test_5fold_tta3_cls.csv
