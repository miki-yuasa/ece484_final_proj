import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse
from gazebo_msgs.msg import ModelState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from std_msgs.msg import Float32MultiArray
import math
from util import euler_to_quaternion, quaternion_to_euler
import time

class vehicleController():

    def __init__(self):
        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size = 1)
        self.prev_vel = 0
        self.L = 1.75 # Wheelbase, can be get from gem_control.py
        self.log_acceleration = False

    def getModelState(self):
        # Get the current state of the vehicle
        # Input: None
        # Output: ModelState, the state of the vehicle, contain the
        #   position, orientation, linear velocity, angular velocity
        #   of the vehicle
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp = serviceResponse(model_name='gem')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
            resp = GetModelStateResponse()
            resp.success = False
        return resp


    # Tasks 1: Read the documentation https://docs.ros.org/en/fuerte/api/gazebo/html/msg/ModelState.html
    #       and extract yaw, velocity, vehicle_position_x, vehicle_position_y
    # Hint: you may use the the helper function(quaternion_to_euler()) we provide to convert from quaternion to euler
    def extract_vehicle_info(self, currentPose):

        ####################### TODO: Your TASK 1 code starts Here #######################
        
        pos_x, pos_y, vel, yaw = 0, 0, 0, 0
        
        # Extract vehicle position
        pos_x = currentPose.pose.position.x                 # syntax taken from set_position() in set_pos.py
        pos_y = currentPose.pose.position.y
        
        # Extract linear velocity (assuming velocity in x is the forward velocity of the ego vehicle)
        vel = currentPose.twist.linear.x
        
        # Extract yaw from quaternion
        # From set_position(), yaw has been previously converted from euler to quaternion
        orientation_q = currentPose.pose.orientation        # more human-readable syntax
        yaw = quaternion_to_euler(orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
        
        ####################### TODO: Your Task 1 code ends Here #######################

        return pos_x, pos_y, vel, yaw # note that yaw is in radian

    # Task 2: Longtitudal Controller
    # Based on all unreached waypoints, and your current vehicle state, decide your velocity
    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):
        
        ####################### TODO: Your TASK 2 code starts Here #######################
        
        target_velocity = 10                                # baseline pre-calculation target_velocity given by initial code
        
        # Suggested baseline speeds
        straight_speed = 12                                 # m/s for straight sections of map
        turn_speed = 8                                      # m/s for curved/turning sections of map
        
        # Check if there are enough waypoints to determine curvature
        if len(future_unreached_waypoints) < 2:
            return straight_speed                           # Not enough waypoints, assume straight
        
        # Get the next two waypoints (assuming there are two or more waypoints now)
        wp1 = future_unreached_waypoints[0]
        wp2 = future_unreached_waypoints[1]

        # Compute vector from current position to next waypoint
        vector_to_wp1 = [wp1[0] - curr_x, wp1[1] - curr_y]
        
        # Computer vector from next waypoint to waypoint after (We are using this approach)
        # Can also use approach of curr to next and curr to next + 1
        vector_wp1_to_wp2 = [wp2[0] - wp1[0], wp2[1] - wp1[1]]
        
        # Compute and compare the angles between the vectors to determine straight vs turn
        # We can use angle = cos^-1[(a . b) / (|a||b|)]
        # We can calculate the magnitude of a vector using |V| = sqrt(X^2 + Y^2)
        # np.linalg.norm is used to calculate the magnitude
        angle = np.arccos(np.dot(vector_to_wp1, vector_wp1_to_wp2) / 
                      (np.linalg.norm(vector_to_wp1) * np.linalg.norm(vector_wp1_to_wp2)))
        
        # Determine if the path is currently straight with some leeway given to the angle to prevent false positives
        if angle < np.degrees(10):                          # +- 10 degrees of error tolerance
            target_velocity = straight_speed                # assumption that vehicle is moving straight
        else:
            target_velocity = turn_speed 

        ####################### TODO: Your TASK 2 code ends Here #######################
        return target_velocity


    # Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints):

        ####################### TODO: Your TASK 3 code starts Here #######################
        target_steering = 0                                 # initial variable declaration
        
        # Set a fixed look-ahead distance
        lookahead_distance = 5                              # We want a larger L for smoother curvature during steering
        
        # Compute the angle between the vehicle's orientation and the look-ahead line
        # Convert target waypoint to relative coordinates
        target_x_rel = target_point[0] - curr_x
        target_y_rel = target_point[1] - curr_y
        angle_to_target = np.arctan2(target_y_rel, target_x_rel)
        
        # Calculate the angle alpha, find difference between look-ahead line angle and current yaw angle
        alpha = angle_to_target - curr_yaw                  # Alpha represents the angle of turn needed

        # Ensure alpha is within the range [-pi, pi] (fail safe check)
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        # Calculate the steering angle delta using the Pure Pursuit formula
        L = self.L                                          # Wheelbase of the vehicle
        target_steering = np.arctan2(2 * L * np.sin(alpha), lookahead_distance)
        
        ####################### TODO: Your TASK 3 code starts Here #######################
        return target_steering


    def execute(self, currentPose, target_point, future_unreached_waypoints):
        # Compute the control input to the vehicle according to the
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: ModelState, the current state of the vehicle
        #   target_point: [target_x, target_y]
        #   future_unreached_waypoints: a list of future waypoints[[target_x, target_y]]
        # Output: None

        curr_x, curr_y, curr_vel, curr_yaw = self.extract_vehicle_info(currentPose)

        # Acceleration Profile
        if self.log_acceleration:
            acceleration = (curr_vel- self.prev_vel) * 100 # Since we are running in 100Hz



        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints)
        target_steering = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints)


        #Pack computed velocity and steering angle into Ackermann command
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = target_velocity
        newAckermannCmd.steering_angle = target_steering

        # Publish the computed control input to vehicle model
        self.controlPub.publish(newAckermannCmd)

    def stop(self):
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = 0
        self.controlPub.publish(newAckermannCmd)