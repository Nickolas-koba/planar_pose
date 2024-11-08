#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class NodeTest(Node):
    def __init__(self):
        super().__init__('solo_un_nodo_de_prueba')
        self.get_logger().info("Menbsaje de prueba")

def main(args = None):
    rclpy.init(args=args)

    planar_pose_estimation = NodeTest()

    try:
        rclpy.spin(planar_pose_estimation)  # Mantém o nó ativo
    except KeyboardInterrupt:
        print("Error encontrado")

    planar_pose_estimation.destroy_node()  # Destrói o nó explicitamente
    rclpy.shutdown()  # Encerra o sistema ROS 2

if __name__=="__main__":
    main()