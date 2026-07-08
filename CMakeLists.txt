# Sets the minimum required CMake version for ROS 2 Humble
cmake_minimum_required(VERSION 3.8)

# Initializes the build project for the haptic teleoperation package
project(haption_teleoperation)

# Enforces strict compiler warnings for safety-critical C++ code
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ==============================================================================
# 1. DEPENDENCY RESOLUTION
# ==============================================================================
# Locates core ament and ROS 2 middleware packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Locates the TIRPC module required by the Virtuose API network stack
find_package(PkgConfig REQUIRED)
pkg_check_modules(TIRPC REQUIRED libtirpc)

# Injects the TIRPC headers into the global include path
include_directories(${TIRPC_INCLUDE_DIRS})

# Injects local package headers into the global include path
include_directories(include/haption_teleoperation)

# ==============================================================================
# 2. C++ BINARY COMPILATION
# ==============================================================================
# Compiles the main Virtuose server node from C++ source
add_executable(virtuose_server_node src/virtuose_server_node.cpp)
# Assigns public include directories to the server node target
target_include_directories(virtuose_server_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Compiles the Virtuose spatial calibration tool from C++ source
add_executable(virtuose_calibration src/calibration_main.cpp)
# Assigns public include directories to the calibration target
target_include_directories(virtuose_calibration PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# ==============================================================================
# 3. DYNAMIC LINKING
# ==============================================================================
# Links ROS 2 middleware dependencies to the server node
ament_target_dependencies(virtuose_server_node
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
)

# Links the proprietary Virtuose spatial API and TIRPC to the server node
target_link_libraries(virtuose_server_node 
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/libVirtuoseAPI.so 
  ${TIRPC_LIBRARIES}
)

# Links the proprietary Virtuose spatial API and TIRPC to the calibration tool
target_link_libraries(virtuose_calibration 
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/libVirtuoseAPI.so 
  ${TIRPC_LIBRARIES}
)

# ==============================================================================
# 4. INSTALLATION EXPORTS
# ==============================================================================
# Deploys the C++ executable binaries into the ROS 2 package library space
install(TARGETS 
  virtuose_server_node
  virtuose_calibration
  DESTINATION lib/${PROJECT_NAME}
)

# Deploys the package C++ header files into the ROS 2 include space
install(DIRECTORY include/
  DESTINATION include
)

# Deploys the proprietary Haption shared library into the core ROS 2 library space
install(FILES lib/libVirtuoseAPI.so
  DESTINATION lib
)

# Deploys the Python executable scripts into the ROS 2 package library space
install(PROGRAMS
  scripts/haption_plotter.py
  scripts/haptic_force_manager_battery.py
  scripts/haptic_force_manager_tutorial.py
  scripts/haptic_force_manager_full_tutorial.py
  scripts/haptic_force_manager_noguidance_tutorial.py
  scripts/haptic_force_manager_blending_tutorial.py
  scripts/haptic_force_manager_joystick_sync_tutorial.py
  scripts/teleop_demo_integrator.py
  scripts/teleop_triago.py
  scripts/teleop_triago_clutch.py
  scripts/teleop_triago_joystick.py
  scripts/workspace_debug_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

# Registers the formalized package structure with the ROS 2 indexer
ament_package()