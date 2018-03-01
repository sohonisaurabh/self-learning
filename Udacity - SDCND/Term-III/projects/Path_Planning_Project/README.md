# CarND-Path-Planning-Project

This repository contains C++ code for implementation of Path Planner. This path of waypoints is used to fed to the controller module of a car running on a highway. This task was implemented to partially fulfill Term-III goals of Udacity's self driving car nanodegree program.


## Background

A critical module in the working of a self driving car system is the path planning module. It is essentially the brain of the whole eco system. Path planning module derives the path to be followed by car ahead of time. This path has information on mostly position and velocity of the car in the future. This information acts as an input to the controller module. The controller modules then ensures there is minimum deviation in the planned path and the executed path.

Following parameters serve as an input to the path planning module:
  1. Map of the environment with start and goal location. This is the global map having information on the best possible route from the start to the destination.
  2. Local map of th environment. This map is a subset and a more detailed version of the global map and has information about the landmarks in the area surrounded by car. This map changes when the car moves.
  3. Position of the other vehicles, pedestrians, animals, traffic lights, etc. in the local map. This information is deduced by creating point clouds from the data received from sensor fusion module.
  4. Current position of the car in the local map. This is derived by the localisation module.
  
Information from all the inputs is then used to perfom following tasks:
  1. Prediction - This involves predicting the behavior of car and other elements in the surrounding
  2. Behavior planning - This involves plannning the possible states of the car. For e.g.: Acceleration, Deceleration, lane change, left and right turns, etc.
  3. Trajectory planning - This involves determining the trajectory of the car for a few meters ahead of it based on the speed limit, traffic and capabilities of the car.
  

## Working of Path Planning Module

Path planner assumes that the controller module of car is loss less and that it follows the trajectory perfectly. Hence, a working implementation of path planner is responsible for:

  1. Creating smooth transition path from current location of the few meters ahead towards the goal
  2. Providing discrete waypoints having information on the desired velocity of the car at that location
  3. Updation of the path in real time based on changes in the environment


## Project Goal

The goal of this project was to design a path planner that is able to create smooth, safe paths for the car to follow along a 3 lane highway with traffic. A successful path planner should be able to keep inside its lane, avoid hitting other cars, and pass slower moving traffic all by using localization, sensor fusion, and map data.


## Project Implementation

Simulation of a circular track was achieved in the [Udacity's self driving car simulator](https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2). While Path planner was implmented in C++, the simulator communicated to C++ code with the help of [uWebSockets library](https://github.com/uNetworking/uWebSockets). Following parameters were received from the simulator for each communication ping:

### Main car's localization Data (No Noise)

("x") The car's x position in map coordinates

("y") The car's y position in map coordinates

("s") The car's s position in frenet coordinates

("d") The car's d position in frenet coordinates

("yaw") The car's yaw angle in the map

("speed") The car's speed in MPH

### Previous path data given to the Planner

//Note: Return the previous list but with processed points removed, can be a nice tool to show how far along the path has processed since last time.

("previous_path_x") The previous list of x points previously given to the simulator

("previous_path_y") The previous list of y points previously given to the simulator

Previous path's end s and d values
("end_path_s") The previous list's last point's frenet s value

("end_path_d") The previous list's last point's frenet d value

Sensor Fusion Data, a list of all other car's attributes on the same side of the road. (No Noise)
("sensor_fusion") A 2d vector of cars and then that car's (car's unique ID, car's x position in map coordinates, car's y position in map coordinates, car's x velocity in m/s, car's y velocity in m/s, car's s position in frenet coordinates, car's d position in frenet coordinates).

The final implementation consisted of following major steps:

  ### 1. Creation of smooth trajectory ahead of car
  
  In this step, C++ [spline tool](http://kluge.in-chemnitz.de/opensource/spline/) was used to interpolate a smooth curve out of 5 anchor points defined on the path ahead of the car. The 5 anchor points were chosen as follows:
  
  a. A point behind current car's location
  b. Current location of car
  c. Point ahead of car by 30m
  d. Point ahead of car by 60m
  e. Point ahead of car by 90m

  The speed limit for car was 50 MPH. Hence, the path planner followed a safe speed limit of 48 MPH and determined 30 waypoints ahead of car on the smooth curve by spline.
  
  C++ code for this task is implemented from line 375 to line 503 in main.cpp.
  
  
  ### 2. Prediction of behavior of other cars on the highway
  
  In this step, sensor fusion data passed by simulator was used to find cars ahead, in the right lane and in the left lane of the car.   Following were the flags raised to warn the path planner of the behavior of others cars on the same side of the road:
  
  a. is_car_ahead - This flag was raised when the self driving car was approaching a car ahead of it in the same lane and the distance between them was less than 30m
  b. is_car_right - This flag was raised when cars in the lane to the right of self driving car were either in the range of 30m ahead or 15m behind
  c. is_car_left - This flag was raised when cars in the lane to the left of self driving car were either in the range of 30m ahead or 15m behind
  
  This information was prepared to be consumed by the behavior planner. Behavior of cars on the other side of the road were ignored for the scope of this project.
  
  C++ code for this task is implemented from line 283 to line 339 in main.cpp.

  ### 3. Determination of behavior of self driving car
  
  In this step, the car followed a less complex version of finite state machine having following states:
  
  a. Accelerate - Continue in current lane and accelerate reaching speed limit
  b. Decelerate - Slow down in current lane in order to avoid collision with car ahead
  c. Lane change Left - Change lane to left with current speed if not in leftmost lane
  d. Lane change RIght - Change lane to right with current speed if not in rightmost lane
  
  This information was prepared to be consumed by the trajectory planner to enhance the basic trajectory already devised in step 1.
  

## Project Output

Path planner was used to drive car with a maximum speed of 48 MPH along the highway. Whenver the car got an opportunity to overtake other cars in front of it, it transitioned smoothly from its current lane to either the right or the left lane based on constraints. Snapshots of different motions of car are shown below:

![Car straight motion](https://raw.githubusercontent.com/sohonisaurabh/CarND-Path-Planning-Project/master/image-resources/car-straight-motion.PNG)

![Car overtake left](https://raw.githubusercontent.com/sohonisaurabh/CarND-Path-Planning-Project/master/image-resources/car-overtake-left.PNG)

![Car overtake right](https://raw.githubusercontent.com/sohonisaurabh/CarND-Path-Planning-Project/master/image-resources/car-overtake-right.PNG)

![Car prepare pvertake left](https://raw.githubusercontent.com/sohonisaurabh/CarND-Path-Planning-Project/master/image-resources/car-prepare-overtake-left.PNG)

![Car prepare overtake right](https://raw.githubusercontent.com/sohonisaurabh/CarND-Path-Planning-Project/master/image-resources/car-prepare-overtake-right.PNG)

The was able to drive for more than 4.32 miles to meet the rubric specification of this project.

Detailed insight into features of the simulator and implementation is demonstrated in this [Path Planning demo video](https://youtu.be/tg2NlXvlQYo).

  
## Steps for building the project

### Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
 * Linux and Mac OS, you can also skip to installation of uWebSockets as it installs it as a dependency.
 
* make >= 4.1(mac, Linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
  * Linux and Mac OS, you can also skip to installation of uWebSockets as it installs it as a dependency.
  
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
  * Linux and Mac OS, you can also skip to installation of uWebSockets as it installs it as a dependency.
  
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`. This will install cmake, make gcc/g++ too.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x.
    
* Fortran Compiler
  * Mac: `brew install gcc` (might not be required)
  * Linux: `sudo apt-get install gfortran`. Additionally you have also have to install gcc and g++, `sudo apt-get install gcc g++`. Look in [this Dockerfile](https://github.com/udacity/CarND-MPC-Quizzes/blob/master/Dockerfile) for more info.
  
* [Ipopt](https://projects.coin-or.org/Ipopt)
  * If challenges to installation are encountered (install script fails).  Please review this thread for tips on installing Ipopt.
  * Mac: `brew install ipopt`
  * Linux
    * You will need a version of Ipopt 3.12.1 or higher. The version available through `apt-get` is 3.11.x. If you can get that version to work great but if not there's a script `install_ipopt.sh` that will install Ipopt. You just need to download the source from the Ipopt [releases page](https://www.coin-or.org/download/source/Ipopt/).
    * Then call `install_ipopt.sh` with the source directory as the first argument, ex: `sudo bash install_ipopt.sh Ipopt-3.12.1`. 
  * Windows: If you can use the Linux subsystem and follow the Linux instructions or use Docker environment.
  
* [CppAD](https://www.coin-or.org/CppAD/)
  * Mac: `brew install cppad`
  * Linux `sudo apt-get install cppad` or equivalent.
  * Windows: If you can use the Linux subsystem and follow the Linux instructions or use Docker environment.

* Simulator. You can download these from the [Udacity simulator releases tab](https://github.com/udacity/self-driving-car-sim/releases).

### Running the project in Ubuntu

  1. Check the dependencies section for installation of gcc, g++, cmake, make, uWebsocketIO API, CppAd and Ipopt library.
  
  2. Manually build the project and run using:
    a. mkdir build && cd build
    b. cmake ..
    c. make
    d. ./path-planning
    
  3. Run the Udacity simulator and check the results
