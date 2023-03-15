# Images-Classification-Prediction-Demo
images classification demo, with pre-trained models on ImageNet,using Pytorch

this is a python demo about images classificaion and using the model to predict one class of single image 


Usage
1.you can download the image_classify_demo_0301.py and run it with any python IDE
2.modify all information like the directory of dataset and other hyperparameter
3.the structrue of dataset should like:
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
4. need to create a new pthfile (like resnet101.pth) to save the better parameters during training, and use this pthfile to predict single image. 
