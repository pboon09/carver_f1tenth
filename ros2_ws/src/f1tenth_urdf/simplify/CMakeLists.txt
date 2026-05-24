cmake_minimum_required(VERSION 3.8)
project(f1tenth_urdf)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch meshes urdf rviz
  DESTINATION share/${PROJECT_NAME})

ament_package()