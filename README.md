# ROWMow Sim: INDOT-Autonomous-Mower-Project

Welcome to the **ROWMow Sim**! This project is focused on developing an autonomous mowing system to assist in efficient and safe maintenance of roadside vegetation. The work was funded by Indiana Department of Transportation through the Joint Transportation Research Program and targeted to Indiana roadways for vegetation management.

![Project Banner](Images/UE_vehicles.png) <!-- Optional: Add an image banner or logo here -->

---

## Table of Contents
- [Overview](#overview)
- [Project Goals](#project-goals)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Collection](#data-collection)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
The INDOT Autonomous Mower Project is designed to create an autonomous robotic platform capable of safely and efficiently mowing designated roadside areas. This project uses a combination of **ROS2**, **Unreal Engine**, and **TerraForm Pro** to develop digital-twin roadsides for mowing navigation along a coverage region and obstacle avoidance.

This README provides an overview of the project’s setup, configuration, and usage instructions for contributors and users.

**Additional documentation and details can be found in our following papers:**
- Digital-twin environment generation
  - Mardikes, M., Evans, J., Brown, E., Sprague, N., Wiegman, T., & Shaver, G. (2024). *Constructing Digital-Twin Roadways for Testing and Evaluation of Autonomous Roadside Mowing Vehicles.* Unpublished manuscript.

- ROWMow Sim vehicle configurations and sim2real testing (Nav2 navigation)
  - Mardikes, M., Evans, J., Brown, E., Sprague, N., Wiegman, T., Supe, S., & Shaver, G. (2024). *ROWMow Sim: A Digital-Twin Robotic Simulator for Testing Autonomous Roadside Mowers.* Unpublished manuscript.

- Reinforcement learning based navigation and obstacle avoidance

## Project Goals
- **Real-World Testsites**: Generate real-world locations for simulation testing to target improvements to sim2real transfers.
- **Autonomous Navigation**: Enable autonomous mower research to navigate roadsides using GPS, IMU, and a coupled obstacle detection sensor.
- **Obstacle Detection**: Implement obstacle detection methods with an advanced sensor suite.
- **Data Logging**: Record sensor data for analysis and continuous improvement.
- **Safety Analysis**: Develop safety report from recorded sensor data for analysis by developers.
- **Vehicle Configuration**: Optimize the system based on a developer's vehicle, software, and sensors for maximizing mowing efficiency and safety.

## Features
- **Real-time Navigation**: Uses custom user developed **Nav2** or **RL** methods for autonomous path planning and obstacle avoidance.
- **Sensor Fusion**: Combines data from sensors for positioning and navigation with adjustable sampling rates and noise.
- **Data Recording**: Logs relevant data for later analysis and debugging.
- **Simulation Support**: Offers a high-fidelity Unreal Engine 5.1 simulation for development and testing of vegetation management platforms in a virtual environment.

---

## Installation

### Prerequisites
- Minimum 75 GB of available storage on SSD
- [ROS2](https://docs.ros.org/en/galactic/index.html) (Recommended Distro from rclUE plugin: Humble)
- Python 3.8+
- CMake
- Dependencies:
  ```bash
  sudo apt install ros-${ROS_DISTRO}-nav2 ros-${ROS_DISTRO}-gazebo-ros
- Packages:
  - [rclUE](https://github.com/rapyuta-robotics/rclUE/tree/UE5_devel_humble)
  - [RapyutaSimulationPlugins](https://github.com/rapyuta-robotics/RapyutaSimulationPlugins/tree/devel)
  - [Fields2Cover](https://github.com/Fields2Cover/Fields2Cover)
  - oscar_ros **(NEED LINK AND REPO SETUP FOR HERE)**
  
## Usage
### First Time Using Simulation:
To build the Unreal Engine project:
  - Open the Unreal Engine project's workspace in vscode
  - Identify and click-on the "Run and Debug" Tab (Left-Hand-Side)
  - Navigate to the "RUN AND DEBUG" drop down menu at the top of the application and select "Generate Project Files (UE5)" from the item list
  - Press the green arrow button to run the project generation
### Launching the Simulation (Example Demonstrations with OSCAR Platform):
To test in a simulated environment:
- **Launch the Unreal Engine Project in vscode**
  - Use: Launch Test_Segment_1 Editor (DebugGame) (UE5)
- **Select the map desired from the content browser**
- **Implement the mowing vehicle into the environment if platform does not already exist in environment**
  - Set the autopossess player to player 0 from disabled if it is disabled
- **Play the simulated environment**
- **Launch the external ros2 connection for the GPS sensor in a terminal:**
```
source ~/oscar_ros/install/setup.bash
ros2 launch oscar_ros launch_sim_gps.launch.py
```
- **Launch the external Nav2 controller in a terminal:**
```
source ~/oscar_ros_ws/install/setup.bash
cd ~/oscar_ros_ws/src/oscar_ros/config/sim
ros2 launch nav2_bringup bringup_launch.py params_file:=./<nav2_param_yaml_file>.yaml map:=../../maps/<map_yaml_file>.yaml use_sim_time:= true
```
  - An example launch commmand used for obstacle avoidance from files in oscar_ros package:
```
ros2 launch nav2_bringup bringup_launch.py params_file:=./nav2_params_pcloud.yaml map:=../../maps/empty_map.yaml use_sim_time:= true
```
- **Launch RViz2 for visualized view of robot's inputs and outputs:**
```
source ~/oscar_ros/install/setup.bash
rviz2
```
  - (Optional) Send goal poses in RViz2
- **(Optional) Run external navigation to send goal poses in a terminal:**
```
source ~/oscar_ros_ws/install/setup.bash
cd /<path_to_waypoint_publisher_file>
python3 <waypoint_publisher_file>.py
```
- **Recording Data**
  - To record a rosbag of sensor data:
```
source ~/oscar_ros_ws/install/setup.bash
ros2 bag record --all
```
  - To record simulation data at end of test:
    - Ensure you are the active as the player in Unreal Engine, F8 to change between, and press "R" on the keyboard.
      - A printout statement confirming that data has been generated should appear on the screen
   
### Real-world deployment (Example Demonstrations with OSCAR Platform):
Testing the virtual model developments in the real-world:
- **First Time Use:**
  - **Have a physical model of the vehicle ready**
    - Includes any necessary hardware components and sensors to perform the necessary inputs and outputs for navigation
    - **Note:** Depending on your onboard computing hardware, you may find some of the desired sample rates are too high
    - **Note:** Verify that all orientations for commands match platform if this was not verified for the virtual model development
      - Always good to double check as well! 
  - **Install the navigation package for the robot onto the onboard computer** (i.e., oscar_ros)

- **Launch the IMU in a new terminal (ZED IMU used):**
```
source ~/oscar_ros_ws/install/setup.bash
ros2 launch oscar_ros zed2i.launch.py
```
- **Launch the robot configuration with GPS position estiamtion in a new terminal:**
```
source ~/oscar_ros_ws/install/setup.bash
ros2 launch oscar_ros launch_robot.launch.py
```
- **Launch the robot configuration with GPS position estiamtion in a new terminal:**
```
source ~/oscar_ros_ws/install/setup.bash
ros2 launch oscar_ros robot.launch.py
```
- **Launch the external Nav2 controller in a terminal:**
```
source ~/oscar_ros_ws/install/setup.bash
cd ~/oscar_ros_ws/src/oscar_ros/config/sim
ros2 launch nav2_bringup bringup_launch.py params_file:=./<nav2_param_yaml_file>.yaml map:=../../maps/<map_yaml_file>.yaml use_sim_time:= true
```
  - An example launch commmand used for obstacle avoidance from files in oscar_ros package:
```
ros2 launch nav2_bringup bringup_launch.py params_file:=./nav2_params_pcloud.yaml map:=../../maps/empty_map.yaml use_sim_time:= true
```
- **Launch the coupled sensors for obstacle avoidance in a terminal:**
  - Launch ouster LiDAR demo command:
```
source ~/oscar_ros_ws/install/setup.bash
ros2 launch oscar_ros ouster.launch.py
```
  - Launch OAK-D stereo demo command:
```
source ~/oscar_ros_ws/install/setup.bash
ros2 launch oscar_ros oakd.launch.py
```
- **Launch RViz2 for visualized view of robot's inputs and outputs:**
```
source ~/oscar_ros/install/setup.bash
rviz2
```
  - (Optional) Send goal poses in RViz2
- **(Optional) Run external navigation to send goal poses in a terminal:**
```
source ~/oscar_ros_ws/install/setup.bash
cd /<path_to_waypoint_publisher_file>
python3 <waypoint_publisher_file>.py
```

- **Record Data**
  - To record a rosbag of sensor data:
```
source ~/oscar_ros_ws/install/setup.bash
ros2 bag record --all
```

## Project Structure
**Unreal Project:** Contains virtual environments, vehicle models, and sensors for developing navigation methods in specific environments and scenarios.

**oscar_ros:** Contains navigation, perception, and control ROS2 launch scripts and interface scripts for the autonomous mower.

**Scripts:** Contains additional supplement scripts for filtering data entries to navigation, such as limiting data inputs to a specified region in the field-of-view of a sensor.

<!-- **src:** Contains the main ROS2 packages for navigation, perception, and control.
launch: Launch files for simulation and real-world deployment.
config: Configuration files for sensors, navigation parameters, and tuning.
data: Folder for collected datasets, logs, and ROS2 bag files.
scripts: Utility scripts for data processing, analysis, and configuration setup.-->

## Data Collection
This project includes modules for recording sensor data:

**ROS2 Bag:** Record all relevant topics, excluding unnecessary topics.
**CSV Logging:** Logs GPS, terrain, road entries, and collisions in CSV format for post-analysis.

## Configuration
Adjust the parameters in the config/ directory to fine-tune navigation, sensor thresholds, and control parameters.

Examples:

**GPS Configuration:** Set parameters such as update frequency and accuracy.
**LiDAR Filtering:** Configure point cloud filtering based on range and intensity.

## Results
Here, you can detail the experimental results of your autonomous mower:

- Mowing coverage performance
- Obstacles avoided and accuracy of path-following
- Safety result summary
- Simulation to Real-World

## Contributing
We welcome contributions! Please read our CONTRIBUTING.md for guidelines on submitting pull requests and issues.

## Reporting Issues
If you encounter any issues, feel free to open a GitHub Issue.

## Code of Conduct
Please adhere to our Code of Conduct when interacting with others on this project.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

