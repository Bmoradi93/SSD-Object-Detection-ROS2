import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os
import numpy as np
from .submodules.ssd_modules import ssd_detection

class SSD(Node):
    def __init__(self):
        super().__init__('ssd_vehicle_detection')
        self.prediced_img_publisher = self.create_publisher(Image, 'ssd_output_image', 10)
        self.predicted_box_publisher = self.create_publisher(Int32MultiArray, 'ssd_output_box', 10)
        self.sub_img_raw = self.create_subscription(
            Image,
            'image_raw',
            self.prediction_cb,
            10)
        self.sub_img_raw  # prevent unused variable warning

        print("Initializing the module...")
        # Network Params
        self.ssd_object = ssd_detection()
        self.ssd_object.load_model()
        self.bridge = CvBridge()
        
    def prediction_cb(self, img_raw):
        b_box = []
        ssd_out_img = Image()
        ssd_out_box =Int32MultiArray() 
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_raw, "bgr8")
            input_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            output, b_box = self.ssd_object.predict(input_image)
            ssd_out_img = self.bridge.cv2_to_imgmsg(output, "bgr8")
            ssd_out_img.header.frame_id = "zed2_camera"
            ssd_out_box.data = b_box
            self.prediced_img_publisher.publish(ssd_out_img)
            self.predicted_box_publisher.publish(ssd_out_box)
        except CvBridgeError as e:
            self.get_logger().info('Image_raw is not processed!')
               
def main(args=None):
    rclpy.init(args=args)
    ssd_vehicle_detection = SSD()
    rclpy.spin(ssd_vehicle_detection)
    ssd_vehicle_detection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
