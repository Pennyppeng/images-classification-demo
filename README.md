# Images-Classification-Prediction-Demo



# Introduction

This is a demo for image classification and image prediction, with pre-trained model based on ImageNet , using Pytorch. You can use it to learn how to classify image and how to use the trained model to predict a single image.

# Requirement

python

torch

torchvision

cuda

tqdm 

......

# Usage

* Only one file in this repo,  you can download the image_classify_demo_0301.py and run it with any python IDE

* modify all infomation using your own address, like the directory of dataset and other hyper-parameters

* structure of dataset should like this:  

  ```
  FI
  ├── train
  │   ├── classname1
  │   │   ├──xx.jpg
  │   │   ├──...
  │   ├── classname2
  │   │   ├──...
  │   ├── ...
  ├── val
  │   ├── classname1
  │   │   ├──xx.jpg
  │   │   ├──...
  │   ├── classname2
  ```

  

* need to create a new pthfile (like resnet101.pth) to save the better parameters during training, and use this pthfile to predict single image
* if you don't want to predict image, just annotate it with '#'