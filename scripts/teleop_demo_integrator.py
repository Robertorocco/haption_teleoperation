#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
from scipy.spatial.transform import Rotation as R

class TeleopDemoIntegrator(Node):
    def __init__(self):
        super().__init__('teleop_demo_integrator')

        # --- State Variables ---
        # Start the gripper at the origin of the world
        self.gripper_pos = np.array([0.0, 0.0, 0.0])
        self.gripper_rot = R.identity()
        self.current_twist = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]

        # --- Timing ---
        self.freq = 150.0  # Hz
        self.dt = 1.0 / self.freq

        # --- ROS 2 Interfaces ---
        self.create_subscription(Twist, 'virtuose/velocity', self.twist_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'demo/gripper_markers', 10)

        # Control loop timer
        self.timer = self.create_timer(self.dt, self.integration_loop)
        
        self.get_logger().info("Twist Integrator Demo started! Waiting for Virtuose velocities...")

    def twist_callback(self, msg):
        """Stores the incoming Haption velocities."""
        self.current_twist[0] = msg.linear.x
        self.current_twist[1] = msg.linear.y
        self.current_twist[2] = msg.linear.z
        self.current_twist[3] = msg.angular.x
        self.current_twist[4] = msg.angular.y
        self.current_twist[5] = msg.angular.z

    def integration_loop(self):
        """Integrates the twist into a pose and publishes the 3D marker."""
        # 1. Integrate Translation (P_new = P_old + V * dt)
        v_linear = self.current_twist[0:3]
        self.gripper_pos += v_linear * self.dt

        # 2. Integrate Rotation (R_new = Delta_R * R_old)
        v_angular = self.current_twist[3:6]
        # Create a rotation delta from the scaled angular velocity vector
        delta_rot = R.from_rotvec(v_angular * self.dt)
        # Apply the rotation delta in the global frame (left-multiplication)
        self.gripper_rot = delta_rot * self.gripper_rot

        # 3. Build Homogeneous Transformation Matrix
        T_pose = np.eye(4)
        T_pose[:3, :3] = self.gripper_rot.as_matrix()
        T_pose[:3, 3]  = self.gripper_pos

        # 4. Generate the Gripper Markers
        now = self.get_clock().now().to_msg()
        markers = self.create_gripper_markers(T_pose, opacity=1.0, step_index=0, now=now)
        
        # 5. Publish the MarkerArray to RViz
        marker_array = MarkerArray()
        marker_array.markers = markers
        self.marker_pub.publish(marker_array)

    def create_gripper_markers(self, T_pose, opacity, step_index, now):
        """Builds a 3-part RED gripper with X as the approach axis."""
        markers = []
        p_center = T_pose[:3, 3]
        R_mat = T_pose[:3, :3]
        quat = R.from_matrix(R_mat).as_quat()
        
        # We use a generic "world" frame for the floating demo
        frame_id = "map"

        # 1. The Base (Palm)
        base = Marker()
        base.header.frame_id = frame_id
        base.header.stamp = now
        base.ns = "demo_gripper"
        base.id = step_index * 3 
        base.type = Marker.CUBE
        base.action = Marker.ADD
        base.pose.position.x, base.pose.position.y, base.pose.position.z = p_center[0], p_center[1], p_center[2]
        base.pose.orientation.x, base.pose.orientation.y, base.pose.orientation.z, base.pose.orientation.w = quat[0], quat[1], quat[2], quat[3]
        
        base.scale.x, base.scale.y, base.scale.z = 0.02, 0.08, 0.03  
        # Made the color RED as requested (R=1.0, G=0.0, B=0.0)
        base.color.r, base.color.g, base.color.b, base.color.a = 1.0, 0.0, 0.0, opacity
        markers.append(base)

        # 2. Left Finger
        offset_l = np.array([0.03, 0.035, 0.0]) 
        p_left = p_center + (R_mat @ offset_l) 
        
        left = Marker()
        left.header = base.header
        left.ns = base.ns
        left.id = step_index * 3 + 1
        left.type = Marker.CUBE
        left.action = Marker.ADD
        left.pose.position.x, left.pose.position.y, left.pose.position.z = p_left[0], p_left[1], p_left[2]
        left.pose.orientation = base.pose.orientation  
        left.scale.x, left.scale.y, left.scale.z = 0.06, 0.01, 0.02  
        left.color = base.color
        markers.append(left)

        # 3. Right Finger
        offset_r = np.array([0.03, -0.035, 0.0]) 
        p_right = p_center + (R_mat @ offset_r)
        
        right = Marker()
        right.header = base.header
        right.ns = base.ns
        right.id = step_index * 3 + 2
        right.type = Marker.CUBE
        right.action = Marker.ADD
        right.pose.position.x, right.pose.position.y, right.pose.position.z = p_right[0], p_right[1], p_right[2]
        right.pose.orientation = base.pose.orientation
        right.scale.x, right.scale.y, right.scale.z = 0.06, 0.01, 0.02
        right.color = base.color
        markers.append(right)

        return markers

def main(args=None):
    rclpy.init(args=args)
    node = TeleopDemoIntegrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()