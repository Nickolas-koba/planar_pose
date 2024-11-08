#!/usr/bin/env python3
# -*- coding: future_fstrings -*-

#Bibliotecas do ROS 2:

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from message_filters import Subscriber

#Bibliotecas de Sistema e Utilidades:

from json.tool import main
from sys import argv
from time import time
import json
import struct
from os.path import dirname, abspath
from os import chdir

#Bibliotecas Numéricas e Visuais:

import numpy as np
from numpy.linalg import norm
import cv2

#Módulos Locais:

from planar_pose.local_modules import config
from planar_pose.local_modules.utils import draw_angled_text


chdir(dirname(abspath(__file__)))

# Except warnings as errors
# warnings.filterwarnings("error")

class PlanarPoseEstimation(Node):
    def __init__(self):
        super().__init__('planar_pose_estimation')  # Inicializa o nó ROS 2

        self.config = config
        self.frame_id = self.config.frame_id
        self.viz_pose = self.config.viz_pose

        # Criação dos publishers usando a instância do nó
        self.pose_info_pub = self.create_publisher(String, '/object_pose_info', 10)
        self.pose_array_pub = self.create_publisher(PoseArray, '/object_pose_array', 10)

        self.object_pose_info = {}
        if self.config.hold_prev_vals:
            self.object_pose_timer = {}

        self.pose_array = PoseArray()
        self.get_logger().info("Prueba de nodo")
        
        # Flag para verificar se a informação da câmera foi recebida
        self.camera_info_received = False
        self.P = None  # Inicialização da matriz de projeção

        # Assinatura dos tópicos sem filtros de tempo
        self.object_detection_sub = self.create_subscription(String, '/detected_object', self.object_detection_callback, 10)
        self.pc_sub = self.create_subscription(PointCloud2, self.config.pc_sub['topic'], self.point_cloud_callback, 10)
        self.image_sub = self.create_subscription(Image, self.config.image_sub['topic'], self.image_callback, 10)

        if self.viz_pose:
            # Agora, a matriz de projeção deve ser configurada quando a mensagem de câmera for recebida
            self.create_subscription(CameraInfo, self.config.cam_info_sub['topic'], self.camera_info_callback, 10)

    def camera_info_callback(self, camera_info):
        # Esta função é chamada quando a informação da câmera é recebida
        self.P = np.array(camera_info.P).reshape((3, 4))
        self.camera_info_received = True  # Define que a informação da câmera foi recebida

    def object_detection_callback(self, object_detection_sub: String):
        self.process_callbacks(object_detection_sub.data)
        
    def point_cloud_callback(self, pc_sub: PointCloud2):
        self.pc_data = pc_sub

    def image_callback(self, image_sub: Image):
        self.image_data = image_sub
        if self.viz_pose:
            self.viz_frame = np.frombuffer(image_sub.data, dtype=np.uint8).reshape(image_sub.height, image_sub.width, -1)
        self.process_callbacks()

    def process_callbacks(self, object_detection_data=None):
        if object_detection_data is not None:
            detected_object = json.loads(object_detection_data)
            pose_array_msg = []

            for i, (object_name, bbox) in enumerate(detected_object.items()):
                if self.config.hold_prev_vals:
                    self.object_pose_timer[object_name] = time()
                if bbox is not None:
                    pose_msg = self.estimate_pose(object_name, bbox, self.pc_data)
                    if pose_msg is not None:
                        pose_array_msg.append(pose_msg)
                    else:
                        if self.config.hold_prev_vals and time() - self.object_pose_timer[object_name] > self.config.hold_period:
                            pose_array_msg.append(self.pose_array.poses[i])

            self.pose_array.poses = pose_array_msg
            self.obj_pose_msg = json.dumps(self.object_pose_info)
            self.pose_info_pub.publish(self.obj_pose_msg)
            self.pose_array.header.stamp = self.get_clock().now().to_msg()  # Corrigido para usar get_clock
            self.pose_array_pub.publish(self.pose_array)

        if self.viz_pose and self.viz_frame is not None and self.camera_info_received:
            cv2.imshow('Pose', self.viz_frame)
            cv2.waitKey(1)

    def estimate_pose(
                self,
                object_name: str,
                bbox: list,
                pc_sub: PointCloud2
            ):
        """
        Estimates planar pose of detected objects and
        updates the stored pose.

        Parameters
        ----------
        object_name: str
            Name of the object.
        bbox : list
            Contains the coordinates of the bounding box
            of the detected object.
        pc_sub : PointCloud2
            A pointcloud object containing the 3D locations
            in terms of the frame `self.frame_id`
        """

        bbox = np.array(bbox)

        # Compute the center, the mid point of the right
        # and top segment of the bounding box
        c = (bbox[0] + bbox[2]) // 2
        x = (bbox[2] + bbox[3]) // 2
        y = (bbox[0] + bbox[3]) // 2

        points = np.array([c, x, y]).tolist()
        vectors_3D = np.zeros((3, 3))

        try:
            # Get the corresponding 3D location of c, x, y
            for pt_count, dt in enumerate(
                cv2.read_points(
                        pc_sub,
                        field_names={'x', 'y', 'z'},
                        skip_nans=False, uvs=points
                    )
                ):
                # If any point returns nan, return
                if np.any(np.isnan(dt)):
                    if object_name in self.object_pose_info.keys():
                        del self.object_pose_info[object_name]
                    rclpy.loginfo('No corresponding 3D point found') #trocado rospy para rclpy
                    return
                else:
                    vectors_3D[pt_count] = dt
                    if pt_count == 2:
                        self.vectors_3D = vectors_3D
        except struct.error as err:
            rclpy.loginfo(err) #trocado rospy para rclpy
            return

        try:
            # 3D position of the object
            c_3D = self.vectors_3D[0]

            # Center the vectors to the origin
            x_vec = self.vectors_3D[1] - c_3D
            x_vec /= norm(x_vec)

            y_vec = self.vectors_3D[2] - c_3D
            y_vec /= norm(y_vec)
            # Take the cross product of x and y vector
            # to generate z vector.
            z_vec = np.cross(x_vec, y_vec)
            z_vec /= norm(z_vec)

            # Recompute x vector to make it truly orthognal
            x_vec_orth = np.cross(y_vec, z_vec)
            x_vec_orth /= norm(x_vec_orth)
        except RuntimeWarning as w:
            rclpy.loginfo(w) #trocado rospy para rclpy
            return

        if self.viz_pose:
            self.draw_pose(object_name, np.vstack((self.vectors_3D, z_vec)))

        # Compute Euler angles i.e. roll, pitch, yaw
        roll = np.arctan2(y_vec[2], z_vec[2])
        pitch = np.arctan2(-x_vec_orth[2], np.sqrt(1 - x_vec_orth[2]**2))
        # pitch = np.arcsin(-x_vec_orth[2])
        yaw = np.arctan2(x_vec_orth[1], x_vec_orth[0])

        [qx, qy, qz, qw] = self.euler_to_quaternion(roll, pitch, yaw)

        # Generate Pose message.
        pose_msg = Pose()

        pose_msg.position.x = c_3D[0]
        pose_msg.position.y = c_3D[1]
        pose_msg.position.z = c_3D[2]
        # Make sure the quaternion is valid and normalized
        pose_msg.orientation.x = qx
        pose_msg.orientation.y = qy
        pose_msg.orientation.z = qz
        pose_msg.orientation.w = qw

        self.object_pose_info[object_name] = {
                'position': c_3D.tolist(),
                'orientation': [qx, qy, qz, qw]
            }

        return pose_msg

    def quaternion_from_matrix(self, matrix):
        """Return quaternion from rotation matrix.

        >>> R = rotation_matrix(0.123, (1, 2, 3))
        >>> q = quaternion_from_matrix(R)
        >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
        True

        """
        q = np.empty((4, ), dtype=np.float64)
        M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        t = np.trace(M)
        if t > M[3, 3]:
            q[3] = t
            q[2] = M[1, 0] - M[0, 1]
            q[1] = M[0, 2] - M[2, 0]
            q[0] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / np.sqrt(t * M[3, 3])
        return q

    def draw_pose(self, object_name: str, vectors_3D: np.ndarray):
        """
        Draw poses as directional axes on the image.

        Parameters
        ----------
        object_name : str
            Name of the object
        vectors_3D : numpy.ndarray
            The 3D directional vectors.
        """

        p_image = np.zeros((4, 2), dtype=np.int32)
        coordinates = None
        for i, vec in enumerate(vectors_3D):
            coordinates = self.project3dToPixel(vec)
            if np.isnan(coordinates).any():
                break
            p_image[i] = coordinates

        if coordinates is not None:
            p_image[3] = p_image[0] - (p_image[3] - p_image[0])
            # z = c + (z-c)*(norm(x-c)/norm(z-c))
            p_image[3] = p_image[0] + (p_image[3] - p_image[0])*(norm(
                    p_image[1] - p_image[0])/norm(
                        p_image[3] - p_image[0]
                    )
                )

            colors_ = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            for i in range(1, 4):
                cv2.line(
                        self.viz_frame,
                        tuple(p_image[0]),
                        tuple(p_image[i]),
                        colors_[i-1],
                        thickness=2
                    )
                x1, y1, x2, y2 = self.calc_vertexes(p_image[0], p_image[i])
                cv2.line(
                        self.viz_frame,
                        tuple(p_image[i]),
                        (x1, y1),
                        colors_[i-1],
                        thickness=2
                    )
                cv2.line(
                        self.viz_frame,
                        tuple(p_image[i]),
                        (x2, y2),
                        colors_[i-1],
                        thickness=2
                    )

            # Put object label aligned to the object's in-plane planar rotation
            text_loc = np.array(
                    [p_image[2, 0] - (p_image[1, 0] - p_image[0, 0])/2,
                     p_image[2, 1] - 20],
                    dtype=np.int16
                )
            base, tangent = p_image[1] - p_image[0]
            text_angle = np.arctan2(-tangent, base)*180/np.pi
            self.viz_frame = draw_angled_text(
                    object_name,
                    text_loc,
                    text_angle,
                    self.viz_frame
                )

    def project3dToPixel(self, point):
        """
        Find the 3D point projected to image plane

        Parameters
        ----------
        point : numpy.ndarrya | tuple | list
            The 3D point.
        Returns
        -------
        numpy.ndarray | None
            The pixel location corresponding to the
            3D vector. Returns None if w is 0.
        """
        src = np.array([point[0], point[1], point[2], 1.0]).reshape(4, 1)
        dst = self.P @ src
        x = dst[0, 0]
        y = dst[1, 0]
        w = dst[2, 0]
        if w != 0:
            px = int(x/w)
            py = int(y/w)
            return np.array([px, py], dtype=np.int32)
        else:
            return None

    def calc_vertexes(self, start_cor: np.ndarray, end_cor: np.ndarray):
        """
        Calculate line segments of the vector arrows to be drawn.

        Parameters
        ----------
        start_cor : numpy.ndarray
            Base point of the arrow.
        end_cor : numpy.ndarray
            End point of the arrow.

        Returns
        -------
        list
            Location of the edge of arrow segments.
        """
        start_x, start_y = start_cor
        end_x, end_y = end_cor
        angle = np.arctan2(end_y - start_y, end_x - start_x) + np.pi
        arrow_length = 15
        arrow_degrees_ = 70

        x1 = int(end_x + arrow_length * np.cos(angle - arrow_degrees_))
        y1 = int(end_y + arrow_length * np.sin(angle - arrow_degrees_))
        x2 = int(end_x + arrow_length * np.cos(angle + arrow_degrees_))
        y2 = int(end_y + arrow_length * np.sin(angle + arrow_degrees_))

        return x1, y1, x2, y2

    def euler_to_quaternion(self, roll: float, pitch: float, yaw: float):
        """
        Converts euler angles to quaternion

        Parameters
        ----------
        roll : float
            Roll angle.
        pitch : float
            Pitch angle.
        yaw : float
            Yaw angle.

        Returns
        -------
        list
            Converted Quaternion values.
        """

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)  # noqa: E501
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)  # noqa: E501
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)  # noqa: E501
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)  # noqa: E501

        return [qx, qy, qz, qw]
    
def main(args=None):
    rclpy.init(args=args)

    config_obj = config  # Instância do config (ajuste se necessário)
    print(config)
    planar_pose_estimation = PlanarPoseEstimation()

    try:
        rclpy.spin(planar_pose_estimation)  # Mantém o nó ativo
    except KeyboardInterrupt:
        print("Error encontrado")

    planar_pose_estimation.destroy_node()  # Destrói o nó explicitamente
    rclpy.shutdown()  # Encerra o sistema ROS 2



if __name__ == "__main__":
    main()