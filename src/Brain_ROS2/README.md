# Brain Operations using ROS2
## 1. Structure
```text
├── src
│   ├── bfmc_brain
│   │   ├── bfmc_brain # lane_keeping node
│   │   │   ├── control_node.py
│   │   │   └── lane_keeping.py
│   ├── bfmc_hardware # camera node and serial node
│   │   ├── bfmc_hardware
│   │   │   ├── camera_node.py
│   │   │   └── serial_node.py
│   ├── bfmc_interfaces # control the main messages (only the LaneInfo is being used)
│   │   ├── msg 
│   │   │   ├── Control.msg
│   │   │   ├── LaneInfo.msg
│   │   │   └── SerialStatus.msg
│   ├── bfmc_launch # containing the launch file to launch all nodes at once
│   │   ├── launch
│   │   │   └── car_launch.py
│   └── bfmc_perception # containing the lane_detector node
│       ├── bfmc_perception
│       │   ├── lane_detector_node.py
│       │   └── lane_detect.py
└── test # the test.py is currently being used as an alternative while I don't have the pi camera here
    ├── image.jpg
    ├── road7.jpeg
    └── test.py
```
## 2. Commands
a. Building workspace
```text
cd Brain_ROS2/
colcon build --allow-overriding bfmc_brain bfmc_hardware bfmc_perception bfmc_interfaces bfmc_launch
source install/setup.bash
```
b. Running nodes
```text
ros2 run [package_name] [executable_name] # for package name and executable name pls search in setup.py:console_script
```
c. Launching the launch file (will launch all nodes at once):
```text
ros2 launch bfmc_launch car_launch.py
```
d. Viewing the result
- For viewing the ouput of the steer and speed, pls use:
```text
ros2 topic echo /cmd_vel
```
- For more detailed view of topics and images, pls use:
```text
rqt # then in the plugins, choose the desired plugins like topic monitor, image viewer, etc.
```
## 3. Usage
a. Without nucleo and camera
- Without nucleo, the TestSerial class is used to print demo serial line to view on terminal
- Without camera, please don't run the camera node and run the test.py instead, which will feed an sample image

b. With nucleo and camera
- [UNDER CONSTRUCTION]
## 4. Notes
- In the bfmc_brain pkg, the control_node.py: the self.controller.prev_steer is set to 0 to avoid the test image being read too many times, letting the previous steer > 0
  - In case there is different from main.py: steer = 0.6 * new_steer + 0.4 * prev_steer
    - ROS2: steer = 0.6 * 24.51 + 0.4 * 24.51 = 24.51
    - Main.py: steer = 0.6 * 24.51 + 0.4 * 0 = 14.71
