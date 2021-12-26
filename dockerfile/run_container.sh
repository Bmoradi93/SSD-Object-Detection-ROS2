#!/bin/sh

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run --runtime=nvidia --privileged --rm -itd --gpus all -v /dev:/dev \
            --volume=$XSOCK:$XSOCK:rw \
            --volume=$XAUTH:$XAUTH:rw \
            --volume=$HOME:$HOME \
            --shm-size=1gb \
            --env="XAUTHORITY=${XAUTH}" \
            --env=TERM=xterm-256color \
            --env=QT_X11_NO_MITSHM=1 \
            --env="DISPLAY=$DISPLAY" \
            --env="QT_X11_NO_MITSHM=1" \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
            --volume="/home/behnam/git/SSD-Object-Detection-ROS2:/root/ros_ws/src/ssd_object_detection" \
            --net=host \
            --env=NVIDIA_VISIBLE_DEVICES=all \
            --env=NVIDIA_DRIVER_CAPABILITIES=all \
            --env=QT_X11_NO_MITSHM=1 \
            --runtime=nvidia \
            --name=deep-foxy\
            deep-foxy_1:latest \
            bash

sleep 2
docker exec -it deep-foxy bash