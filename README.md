# Heuristic Optimization of Deep Learning Models for Segmentation of Intracellular Organelle Networks

This repo is for the paper "Heuristic Optimization of Deep Learning Models for Segmentation of Intracellular Organelle Networks" (under review)

---

## Overview

### Data Preparation

Two self-collect datasets can download from [IEEE Dataport](https://ieee-dataport.org/documents/fluorescence-microscopy-images-cbmi), including endoplasmic reticulum(ER) and mitochondrial(MITO). All images are 256X256 with manual anntotation.

For ER dataset, the training, validation and testing sets consist of 157, 28 and 38 images, respectively. 
For MITO dataset, the training, validation and testing sets consist of 165, 8 and 10 images, respectively. 

Horizontal and vertical flipping as well as 90°/180°/270° rotation were used for training data augmentation.

### Model

![The graphical scheme of our heuristic approach for fluorescence microscopy images](https://github.com/YaoruLuo/bionetwork_segmentation/blob/master/images/Picture1.png)

We only need to output the probability of foreground since it is the binary image segmentation, we use sigmoid activation function makes sure that mask pixels are in \[0, 1\] range.

---

## How to use

### Dependencies

This tutorial depends on the following libraries:

* PyTorch version 1.2.0
* Python version 3.7

### Training

To train the model, you should save the datapath into a **_.txt** file and save in the dictionary **datasets**, then run ***_trainer.py** for different models.

The loss funcitons are save in **models/optimize.py**

Most models are trained for 30 epochs.


### Evaluation
To test the segmentation performance, you should first run **inference.py** to save the prediction of the model, then run **evaluation.py** to get the different metrics scores such as IOU, F1 and others.


### Results
You can train the models to get the rusuls in our paper or directly download our pretraining models.
[PENet]()
[U-Net]()
[Deeplabv3+]()
[UNetPlusPlus]()

Use the trained model to do segmentation on test images.

![archtecture search](https://github.com/YaoruLuo/bionetwork_segmentation/blob/master/images/Picture2.png)
