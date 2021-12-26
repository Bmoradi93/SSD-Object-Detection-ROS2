# SSD (Single Shot Multibox Detector) Object Detection, ROS-Foxy Package #
In order to train your SSD model on your dataset, please refer to [This pipeline](https://github.com/Bmoradi93/SSD-Object-Detection-TFOD-Training-Pipeline).

Single Shot Multibox Detector (SSD) is a CNN-based network architecture that was introduced by Liu et al in 2015 The bounding box regression algorithm has been taken from Google’s Inception Network. The SSD network is combined with bounding box regression to eliminate and replace the region proposal network that was used in Faster R-CNN network architecture. SSD is able to achieve Faster R-CNN accuracy and obtain 59+ fps.

SSD is capable of detecting multiple objects. Unlike R-CNN methods, it propagates the feature map in one forward pass throughout the network. This is the main reason that SSD is able to operate in real-time and handle object overlap in the data points. SSD uses a pre-trained network as a basic net which is trained on the ImageNet dataset. There are multiple convolutional layers and each one of them is individually and directly connected to the fully connected layer. A combination of SSD and bounding box regression allows the network to detect multiple objects with different scales.


### What is this repository for? ###
In this repository, a ROS-Foxy inference package is provided to take care of inference process in real-time.
The followings are its specifications:
* It subscrible to a topic with Image message type
* It publishes an image with detected objects (Bounding Boxes)
* The inference speed is 52 fps (Single Nvidia GPU-powered PC)

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact
