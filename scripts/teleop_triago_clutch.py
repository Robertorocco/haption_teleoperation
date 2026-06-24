#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Bool

import numpy as np
from scipy.spatial.transform import Rotation as R

class TeleopClutch(Node):
    def __init__(self):
        super().__init__('teleop_clutch')

        # --- State Variables ---
        self.initialized = False
        self.ref_pos = np.zeros(3)
        self.ref_rot = R.identity()
        
        self.v_cmd = np.zeros(3)
        self.w_cmd = np.zeros(3)

        self.clutch_engaged = False

        # --- Parameters ---
        self.freq = 150.0  # Hz
        self.dt = 1.0 / self.freq
        
        self.K_trans = 1.0  # Translational scale factor
        self.K_rot = 1.0    # Rotational scale factor (Best kept at 1.0 for intuition)
        
        # 6.0 = Full 6D control | 5.0 = Free rotation around approach axis
        self.task_dim = 6.0 

        # --- ROS 2 Interfaces ---
        # 1. Listen to Haption Twist and Button (Clutch)
        self.create_subscription(Twist, 'virtuose/velocity', self.twist_callback, 10)
        self.create_subscription(Bool, 'virtuose/button', self.button_callback, 10)

        # 2. Listen to TRIAGo Real Pose (Used ONCE to anchor the integration)
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self.ee_callback, 10)
        
        # 3. Publish Command to Controller
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/arm_right/cartesian_reference', 10)

        # Control loop timer
        self.timer = self.create_timer(self.dt, self.integration_loop)
        
        self.get_logger().info("Teleop Clutch started. Waiting for initial TRIAGo pose...")

    def ee_callback(self, msg):
        """Grabs the robot's current pose to initialize the integration anchor."""
        if not self.initialized:
            try:
                # Extract initial Position
                self.ref_pos = np.array(msg.data[0:3])
                
                # Extract initial Orientation (RPY)
                rpy_real = np.array(msg.data[12:15])
                self.ref_rot = R.from_euler('xyz', rpy_real, degrees=False)
                
                self.initialized = True
                self.get_logger().info(f"Integration Anchor Initialized at: {self.ref_pos}")
            except IndexError:
                self.get_logger().warn("Malformed /qp_debug/ee_real message received.")

    def twist_callback(self, msg):
        """Maps Haption Twist to TRIAGo Frame (180-deg rotation on Z)."""
        # Haption Base (X: back, Y: right) -> TRIAGo Base (X: forward, Y: left)
        # We invert X and Y to mathematically achieve the 180-deg Z-axis flip.
        
        self.v_cmd[0] = -msg.linear.x * self.K_trans
        self.v_cmd[1] = -msg.linear.y * self.K_trans
        self.v_cmd[2] =  msg.linear.z * self.K_trans

        self.w_cmd[0] = -msg.angular.x * self.K_rot
        self.w_cmd[1] = -msg.angular.y * self.K_rot
        self.w_cmd[2] =  msg.angular.z * self.K_rot

    def button_callback(self, msg):
        """Updates the clutch state and logs transitions."""
        # Only trigger logic on a state change to avoid spamming the console
        if msg.data != self.clutch_engaged:
            self.clutch_engaged = msg.data
            if self.clutch_engaged:
                self.get_logger().info(" CLUTCH ENGAGED: Robot frozen. Reposition your hand freely.")
            else:
                self.get_logger().info(" CLUTCH RELEASED: Teleoperation tracking resumed.")

    def integration_loop(self):
        """Integrates the twist and publishes the 13-element array."""
        if not self.initialized:
            return # Do nothing until we know where the robot is

        # If clutch is NOT pressed, integrate velocities and pass them to the robot
        if not self.clutch_engaged:
            # 1. Integrate Translation (P_new = P_old + V * dt)
            self.ref_pos += self.v_cmd * self.dt

            # 2. Integrate Rotation (R_new = Delta_R * R_old)
            delta_rot = R.from_rotvec(self.w_cmd * self.dt)
            self.ref_rot = delta_rot * self.ref_rot 
            
            pub_v = self.v_cmd
            pub_w = self.w_cmd
            
        # If clutch IS pressed, freeze the pose and send zero velocity
        else:
            pub_v = np.zeros(3)
            pub_w = np.zeros(3)

        rpy_ref = self.ref_rot.as_euler('xyz', degrees=False)

        # 3. Construct the NEW PROTOCOL Message
        cmd_msg = Float64MultiArray()
        cmd_msg.data = [
            float(self.ref_pos[0]), float(self.ref_pos[1]), float(self.ref_pos[2]), # 0:3 Position
            float(rpy_ref[0]),      float(rpy_ref[1]),      float(rpy_ref[2]),      # 3:6 RPY
            float(pub_v[0]),        float(pub_v[1]),        float(pub_v[2]),        # 6:9 Linear Vel
            float(pub_w[0]),        float(pub_w[1]),        float(pub_w[2]),        # 9:12 Angular Vel
            float(self.task_dim)                                                    # 12: Task Dim Flag
        ]

        # 4. Publish
        self.cmd_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TeleopClutch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()