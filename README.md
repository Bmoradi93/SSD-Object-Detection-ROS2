# SSD (Single Shot Multibox Detector) Object Detection, ROS-Foxy Package #
In order to train your SSD model on your dataset, please refer to [my ssd training pipeline](https://github.com/Bmoradi93/SSD-Object-Detection-TFOD-Training-Pipeline).

![Selection_104](https://user-images.githubusercontent.com/47978272/147527939-05ce6730-6c0a-4fde-a3b8-e80f935f2ec4.png)


Single Shot Multibox Detector (SSD) is a CNN-based network architecture that was introduced by Liu et al in 2015 The bounding box regression algorithm has been taken from Google’s Inception Network. The SSD network is combined with bounding box regression to eliminate and replace the region proposal network that was used in Faster R-CNN network architecture. SSD is able to achieve Faster R-CNN accuracy and obtain 59+ fps.

SSD is capable of detecting multiple objects. Unlike R-CNN methods, it propagates the feature map in one forward pass throughout the network. This is the main reason that SSD is able to operate in real-time and handle object overlap in the data points. SSD uses a pre-trained network as a basic net which is trained on the ImageNet dataset. There are multiple convolutional layers and each one of them is individually and directly connected to the fully connected layer. A combination of SSD and bounding box regression allows the network to detect multiple objects with different scales.

![Selection_228](https://user-images.githubusercontent.com/47978272/147528052-0ed9d178-c50e-401b-a590-8b23c3f0534c.png)


### What is this repository for? ###
In this repository, a ROS-Foxy inference package is provided to take care of inference process in real-time.
The followings are its specifications:
* It subscrible to a topic with Image message type
* It publishes an image with detected objects (Bounding Boxes)
* The inference speed is 52 fps (Single Nvidia GPU-powered PC)

### How do I get set up? ###
This ros package comes with a compatible docker image. To run the package, you need to pull and run this docker image.
1- In your $HOME directory, create a folder and name it 'git':

```
cd ~

mkdir git && cd git && git clone https://github.com/Bmoradi93/SSD-Object-Detection-ROS2.git && git lfs pull

```

2- Pull the docker image and run the container:

```
cd $HOME

docker pull behnammoradi026/deep_learning:ROS2

cd cd $HOME/git/SSD-Object-Detection-ROS2/dockerfile

source run_container.sh
```
3- Once your terminal is in the docker container, build the ROS workspace and run the package:

```
cd $HOME/ros_ws && colcon build

source install/setup.bash

ros2 launch ssd_object_detection ssd_object_detection.launch.py
```


### Who do I talk to? ###

* Behnam Moradi (behnammoradi026@gmail.com)
