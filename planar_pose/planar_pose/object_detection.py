#!/usr/bin/env python3
# -*- coding: future_fstrings -*-

#Bibliotecas do ROS 2:

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile

# Bibliotecas de Sistema e Utilidades:

from cProfile import Profile
from json.tool import main
from os.path import dirname, abspath, join
from os import listdir, chdir
from sys import argv
from time import time
import json

#Bibliotecas Numéricas e Visuais:

import numpy as np
import cv2

# Módulos Locais:

from planar_pose.local_modules.utils import draw_angled_text, shrink_bbox
from planar_pose.local_modules import config

chdir(dirname(abspath(__file__)))

class ObjectDetection(Node):
    def __init__(self, config = config, to_gray: bool = False):
        super().__init__('object_detection')  # Inicializa o nó ROS 2

        self.config = config
        self.to_gray = config.to_gray
        self.detector_descriptor = config.detector_descriptor
        self.matcher = config.matcher
        
        if self.config.show_image:
            self.viz_frame = {}

        # Holds all the information i.e. keypoints,
        # descriptors, and dimension, for object detection.
        self.query_object_features = {}
        self.get_logger().info("tente de novo")

        if self.config.hold_prev_vals:
            self.object_timer = {}

        # Retrieves all the image feature info needed
        # for each of the object for detection
        self.extract_object_info(self.config.objects, self.config.object_path)

        # Set frame rate
        self.frame_rate = 2
        self.prev = 0

        # Initializes ROS 2 node for object detection
        self.obj_boundary_msg = ''
        self.obj_boundary_info = {}

        # Create the publisher with a QoS profile
        qos_profile = QoSProfile(depth=10)  # Define a política de QoS
        self.obj_boundary_pub = self.create_publisher(
            String,
            '/detected_object',
            qos_profile
        )

        # Set up the image subscriber without message filters
        self.create_subscription(
            String,
            self.config.image_sub['topic'],
            self.callback,
            qos_profile  # Passe a variável qos_profile, não a classe
        )
        

    def extract_object_info(self, objects, obj_path: str):  #Essa função carrega as imagens dos objetos e extrai informações como características e dimensões.
        """
        Extracts all the information and features needed for
        object detection.

        Parameters
        ----------
        objects : list | str
            List of object names.
        obj_path : str
            Path to the folder containing object images.
        """
        # Image extensions that are supported
        supported_formats = ['jpg', 'jpeg', 'png']

        if objects == 'all':    #Carrega todas as imagens se objects for all, ou apenas aquelas especificadas.
            image_files = [join(obj_path, f) for f in listdir(obj_path)
                           if f.endswith(('.jpg', '.png'))]
            
        else:    # Carrega somente os objetos especificados
            image_files = []   
            for f in listdir(obj_path):
                obj, ext = f.split('.')
                for object_ in objects:
                    if obj == object_ and ext in supported_formats:
                        image_files.append(join(obj_path, f'{obj}.{ext}'))
                        objects.remove(object_)

        for im_file in image_files:
            object_name = im_file.split('/')[-1].split('.')[0]

            if self.config.hold_prev_vals:
                self.object_timer[object_name] = time()

            try:
                object_im = cv2.imread(im_file)
                h, w = object_im.shape[0:2]
                obj_boundary = shrink_bbox([[0, 0], [0, h-1],
                                            [w-1, h-1], [w-1, 0]])

                if self.to_gray:
                    object_im = cv2.cvtColor(object_im, cv2.COLOR_BGR2GRAY)
                    object_im = object_im[obj_boundary[0, 1]:obj_boundary[2, 1],  # noqa: E501
                                          obj_boundary[0, 0]:obj_boundary[2, 0]]  # noqa: E501
                else:
                    object_im = object_im[obj_boundary[0, 1]:obj_boundary[2, 1],  # noqa: E501
                                          obj_boundary[0, 0]:obj_boundary[2, 0],  # noqa: E501
                                          :]

                kp, des = self.detector_descriptor.detectAndCompute(object_im, None)  # noqa: E501
                dim = object_im.shape[0:2]
                self.query_object_features[object_name] = [kp, des, dim]
            except cv2.error as e:  # noqa: E722
                self.get_logger().error(f'Erro ao ler imagem em {im_file}: {e}')

    def callback(self, sensor_image: np.ndarray):
        """
        Callback function for the object detection node

        Parameters
        ----------
        sensor_image : numpy.ndarray
            Image retrieved from a sensor (webcam/kinect).
        """
        image = np.frombuffer(
                sensor_image.data, dtype=np.uint8
            ).reshape(
                    sensor_image.height, sensor_image.width, -1
                )
        self.viz_frame = image
        if image is None:
            self.get_logger().info('invalid image received') #rospy para rclpy // rclpy.loginfo -->  self.get_logger().info('imagem inválida recebida')
            return

        time_elapsed = time() - self.prev
        if time_elapsed > 1. / self.frame_rate:
            self.prev = time()
            for object_name, feat in self.query_object_features.items():
                self.detect(object_name, feat, image)

            # Convert the dictionary to string
            self.obj_boundary_msg = json.dumps(self.obj_boundary_info)
            self.obj_boundary_pub.publish(String(data=self.obj_boundary_msg))

            if self.config.show_image:
                cv2.imshow('Detected Objects', self.viz_frame)
                cv2.waitKey(10)

    def annotate_frame(self, viz_frame, dst, object_name):
        viz_frame = cv2.polylines(
                                    viz_frame,
                                    [dst],
                                    True,
                                    255,
                                    1,
                                    cv2.LINE_AA
                                )

        dst = np.squeeze(dst, axis=1)
        tc = (dst[3] + dst[0])/2
        tc = (tc + dst[0])/2

        text_loc = np.array([tc[0], tc[1] - 20], dtype=np.int16)
        base, tangent = dst[3] - dst[0]
        text_angle = np.arctan2(-tangent, base)*180/np.pi
        viz_frame = draw_angled_text(
                            object_name,
                            text_loc,
                            text_angle,
                            viz_frame
                        )
        return viz_frame

    def detect(
                self,
                object_name: str,
                query_img_feat: list,
                sensor_image: np.ndarray
            ):
        """
        Detects if the object is in the frame

        Parameters
        ----------
        object_name : str
            Name of the object.
        query_img_feat : list
            A list containing keypoints, descriptors, and
            dimension information of query object's image
        sensor_image : numpy.ndarray
            Image retrieved from a sensor (webcam/kinect).
        """
        MIN_MATCH_COUNT = self.config.min_match_count
        # If True the frame with detected object will
        # be showed, by default False
        show_image = self.config.show_image
        if self.to_gray:
            # sensor_rgb = sensor_image
            sensor_image = cv2.cvtColor(sensor_image, cv2.COLOR_BGR2GRAY)

        kp1, des1, dim = query_img_feat
        kp2, des2 = self.detector_descriptor.detectAndCompute(sensor_image, None)  # noqa: E501

        matches = self.matcher(des1, des2, **self.config.matcher_kwargs)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32(
                    [kp1[m.queryIdx].pt for m in good]
                ).reshape(-1, 1, 2)

            dst_pts = np.float32(
                    [kp2[m.trainIdx].pt for m in good]
                ).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is None:
                return

            h, w = dim
            pts = np.float32(
                    [[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                ).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M).astype(np.int32)
            # update the location of the object in the image
            # converted to list as ndarray object is not json serializable
            self.obj_boundary_info[object_name] = np.squeeze(dst, axis=1).tolist()  # noqa: E501

            if self.config.hold_prev_vals:
                self.object_timer[object_name] = time()

            if show_image:
                # sensor_rgb = cv2.polylines(sensor_rgb, [dst] ,True,255,1, cv2.LINE_AA)  # noqa: E501
                self.viz_frame = self.annotate_frame(
                                            self.viz_frame,
                                            dst,
                                            object_name
                                        )

        else:
            if self.config.hold_prev_vals:
                if time() - self.object_timer[object_name] > self.config.hold_period:  # noqa: E501
                    self.obj_boundary_info[object_name] = None
                else:
                    if self.config.show_image and object_name in self.obj_boundary_info.keys():  # noqa: E501
                        self.viz_frame = self.annotate_frame(
                                            self.viz_frame,
                                            np.expand_dims(
                                                np.array(self.obj_boundary_info[object_name], dtype=np.int32),  # noqa: E501
                                                axis=1
                                            ),
                                            object_name
                                        )
            else:
                # Set None if the object isn't detected
                self.obj_boundary_info[object_name] = None
                self.get_logger("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))  # noqa: E501 #rospy para rclpy

def main(args=None):
        rclpy.init(args=args)

        config_obj = config  # Instância do config (ajuste se necessário)
        object_detection = ObjectDetection(config_obj)

        try:
            rclpy.spin(object_detection)  # Mantém o nó ativo
        except KeyboardInterrupt:
            print("Error encontrado")
    
        object_detection.destroy_node()  # Destrói o nó explicitamente
        rclpy.shutdown()  # Encerra o sistema ROS 2

if __name__ == '__main__':
    main()