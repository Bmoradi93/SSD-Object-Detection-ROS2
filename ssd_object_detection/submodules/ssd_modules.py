from object_detection.utils import label_map_util
import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import imutils
import cv2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from itertools import chain
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class ssd_detection:
    def __init__(self):
        self.path_to_ssd_model = "/root/autoware_auto_thesis/dockerfiles/AutowareAuto/src/perception/SSD-Object-Detection-ROS2/trained_model/frozen_inference_graph.pb"
        self.path_to_ssd_labels = "/root/autoware_auto_thesis/dockerfiles/AutowareAuto/src/perception/SSD-Object-Detection-ROS2/trained_model/classes.pbtxt"
        self.number_of_classes = 2
        self.input_image_topic = 'image_raw'
        self.otput_ssd_topic = 'ssd_image_output'
        self.output_bounding_box_topic = 'bounding_box_location'
        self.img_resize_value = 500
        self.min_confidence = 0.5
        self.COLORS = np.random.uniform(0, 255, size=(self.number_of_classes, 3))
        self.bounding_box = []
        self.load_model()
        with self.model.as_default():
            self.sess = tf.Session(graph=self.model, config=config)
            self.imageTensor = self.model.get_tensor_by_name("image_tensor:0")
            self.boxesTensor = self.model.get_tensor_by_name("detection_boxes:0")

            self.scoresTensor = self.model.get_tensor_by_name("detection_scores:0")
            self.classesTensor = self.model.get_tensor_by_name("detection_classes:0")
            self.numDetections = self.model.get_tensor_by_name("num_detections:0")
    
    def load_model(self):
        self.model = tf.Graph()
        with self.model.as_default():
            graphDef = tf.GraphDef()
            with tf.gfile.GFile( self. path_to_ssd_model, "rb") as f:
                serializedGraph = f.read()
                graphDef.ParseFromString(serializedGraph)
                tf.import_graph_def(graphDef, name="")

        labelMap = label_map_util.load_labelmap(self.path_to_ssd_labels)
        categories = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes=self.number_of_classes, use_display_name=True)
        self.categoryIdx = label_map_util.create_category_index(categories)

    def process_input_image(self, image):
        (H, W) = image.shape[:2]

        if W > H and W > self.img_resize_value:
            image = imutils.resize(image, width=self.img_resize_value)

        elif H > W and H > self.img_resize_value:
            image = imutils.resize(image, height=self.img_resize_value)

        (H, W) = image.shape[:2]
        output = image.copy()
        # image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0)
        return image, output, H, W

    def predict(self, input_image):
        processed_image, output, H, W = self.process_input_image(input_image)
        (boxes, scores, labels, N) = self.sess.run(
            [self.boxesTensor, self.scoresTensor, self.classesTensor, self.numDetections],
            feed_dict={self.imageTensor: processed_image})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        labels = np.squeeze(labels)
        x = [0, 0, 0, 0]
        approved_boxes = []
        for (box, score, label) in zip(boxes, scores, labels):
            if score < self.min_confidence:
                continue

            (startY, startX, endY, endX) = box
            startX = int(startX * W)
            startY = int(startY * H)
            endX = int(endX * W)
            endY = int(endY * H)

            x = [startY, startX, endY, endX]
            approved_boxes.append([startY, startX, endY, endX])
            label = self.categoryIdx[label]
            idx = int(label["id"]) - 1
            label = "{}: {:.2f}".format(label["name"], score)
            cv2.rectangle(output, (startX, startY), (endX, endY), self.COLORS[idx], 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(output, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.COLORS[idx], 1)
        
        return output, list(chain.from_iterable(approved_boxes))
