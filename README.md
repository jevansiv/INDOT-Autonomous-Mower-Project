# INDOT-Autonomous-Mower-Project

Welcome to the **INDOT Autonomous Mower Project**! This project is focused on developing an autonomous mowing system to assist in efficient and safe maintenance of roadside vegetation.

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
- [ROS2](https://docs.ros.org/en/galactic/index.html) (Recommended Distro: Humble)
- Python 3.8+
- CMake
- Dependencies:
  ```bash
  sudo apt install ros-${ROS_DISTRO}-nav2 ros-${ROS_DISTRO}-gazebo-ros
- Packages:
  - [rclUE](https://github.com/rapyuta-robotics/rclUE/tree/UE5_devel_humble)
  - [RapyutaSimulationPlugins](https://github.com/rapyuta-robotics/RapyutaSimulationPlugins/tree/devel)
  - [Fields2Cover](https://github.com/Fields2Cover/Fields2Cover)
  - 
  
## Usage
Launching the Simulation
To test in a simulated environment:

```
ros2 launch indot_mower simulation.launch.py
```

Running the Autonomous Mower
For real-world deployment:

```
ros2 launch indot_mower mower.launch.py
```

Recording Data
To record relevant data (excluding unnecessary topics):

```
ros2 bag record --all
```

## Project Structure
**src:** Contains the main ROS2 packages for navigation, perception, and control.
launch: Launch files for simulation and real-world deployment.
config: Configuration files for sensors, navigation parameters, and tuning.
data: Folder for collected datasets, logs, and ROS2 bag files.
scripts: Utility scripts for data processing, analysis, and configuration setup.

## Data Collection
This project includes modules for recording sensor data:

**ROS2 Bag:** Record all relevant topics, excluding unnecessary topics.
**CSV Logging:** Logs GPS, IMU, and LiDAR data in CSV format for post-analysis.

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

