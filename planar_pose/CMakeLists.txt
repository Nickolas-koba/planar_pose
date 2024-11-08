cmake_minimum_required(VERSION 3.5) ##de 3.0.2 para 3.5
project(planar_pose)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif() #adicionado

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif() #adicionado

# Encontrar pacotes necessários do ROS 2
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)  # Substituído rospy por rclpy
find_package(rclcpp REQUIRED)  
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY doc_ objects
  DESTINATION share/${PROJECT_NAME}
)

# Ejecutores de nodos python
install(PROGRAMS
  ${PROJECT_NAME}/planar_pose_estimation.py
  ${PROJECT_NAME}/test_node.py
  ${PROJECT_NAME}/object_detection.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()