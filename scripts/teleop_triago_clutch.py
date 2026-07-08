#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Bool, String

import numpy as np
from scipy.spatial.transform import Rotation as R

# Single source of truth for the shared-autonomy blending architecture (also
# read by triago_control/scripts/qp_arm_teleop/main_shared_autonomy.py). See
# cfg.BLENDING's docstring in triago_control/qp_controller/config.py for the
# full topic-routing rationale.
import triago_control.qp_controller.config as cfg


class TeleopClutch(Node):
    def __init__(self):
        super().__init__('teleop_clutch')

        # Fail loudly if launched under the wrong study condition. The clutch
        # teleop is the position-control input for ALL three CLUTCH conditions
        # (sync / guided-feedback / full), so it only constrains the control mode.
        cfg.validate_condition('teleop_triago_clutch', control_mode=cfg.CLUTCH)

        # --- State Variables ---
        self.initialized = False
        self.ref_pos = np.zeros(3)
        self.ref_rot = R.identity()
        
        self.v_cmd = np.zeros(3)
        self.w_cmd = np.zeros(3)

        self.clutch_engaged = False

        # --- Grasp-execution handover ---
        # When shared_autonomy drives the arm autonomously (grasp approach/close/
        # lift), it publishes /shared_autonomy/grasp_active=True. We freeze (stop
        # publishing) and, on the falling edge, force a re-anchor so teleop resumes
        # from the actual post-grasp robot pose with no jump.
        self.grasp_active = False

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
        self.create_subscription(Bool, 'virtuose/button_right', self.button_callback, 10)

        # 2. Listen to TRIAGo Real Pose (Used ONCE to anchor the integration)
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self.ee_callback, 10)

        # 2b. Listen to the grasp-execution handover flag from shared_autonomy
        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_callback, 10)
        
        # 3. Publish Command to Controller (switchable between arms)
        #
        # Topic routing depends on cfg.BLENDING (shared with main_shared_autonomy.py):
        #   BLENDING=False (legacy): publish the pure user pose directly on
        #     /arm_*/cartesian_reference -- this node drives the QP controller
        #     directly, same as always.
        #   BLENDING=True: publish on /arm_*/user_cartesian_reference instead.
        #     main_shared_autonomy.py listens there for pure user intent, blends
        #     it with the belief-weighted assistive policy, and becomes the SOLE
        #     publisher of the real /arm_*/cartesian_reference. This avoids two
        #     nodes ever racing to publish the same topic.
        self.active_arm = 'right'
        _topic_right = ('/arm_right/user_cartesian_reference' if cfg.ASSIST_BLENDING
                        else '/arm_right/cartesian_reference')
        _topic_left = ('/arm_left/user_cartesian_reference' if cfg.ASSIST_BLENDING
                       else '/arm_left/cartesian_reference')
        self.cmd_pub_right = self.create_publisher(Float64MultiArray, _topic_right, 10)
        self.cmd_pub_left = self.create_publisher(Float64MultiArray, _topic_left, 10)
        self.cmd_pub = self.cmd_pub_right  # current active publisher

        self.get_logger().info(
            f"[TELEOP] CONTROL_MODE={cfg.CONTROL_MODE}, ASSIST_BLENDING={cfg.ASSIST_BLENDING} "
            f"-> publishing user reference on '{_topic_right}' (right) / '{_topic_left}' (left).")

        # 4. Subscribe to arm-switch notifications from shared_autonomy
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)

        # Control loop timer
        self.timer = self.create_timer(self.dt, self.integration_loop)
        
        self.get_logger().info("Teleop Clutch started. Waiting for initial TRIAGo pose...")

    def active_arm_cb(self, msg):
        """Switches which arm the teleop publishes to."""
        if msg.data in ('right', 'left') and msg.data != self.active_arm:
            self.active_arm = msg.data
            self.cmd_pub = self.cmd_pub_right if msg.data == 'right' else self.cmd_pub_left
            # Force a re-anchor so integration starts from the new arm's pose.
            self.initialized = False
            self.get_logger().info(f"[TELEOP] Arm switched to {msg.data.upper()}. Re-anchoring.")

    def ee_callback(self, msg):
        """Grabs the robot's current pose to initialize the integration anchor."""
        if not self.initialized:
            try:
                # EE real layout: [pos_R(3), vel_R(3), pos_L(3), vel_L(3), rpy_R(3), rpy_L(3)]
                if self.active_arm == 'right':
                    self.ref_pos = np.array(msg.data[0:3])
                    rpy_real = np.array(msg.data[12:15])
                else:
                    self.ref_pos = np.array(msg.data[6:9])
                    rpy_real = np.array(msg.data[15:18])
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

    def grasp_active_callback(self, msg):
        """Handover flag: suspend teleop while shared_autonomy drives the grasp."""
        if msg.data and not self.grasp_active:
            self.grasp_active = True
            self.get_logger().info(" GRASP EXEC: teleop suspended (arm driven autonomously).")
        elif not msg.data and self.grasp_active:
            self.grasp_active = False
            # Force a re-anchor at the actual post-grasp pose so the resumed
            # integration starts from where the robot really is (no jump back).
            self.initialized = False
            self.get_logger().info(" GRASP DONE: re-anchoring, teleop resuming.")

    def integration_loop(self):
        """Integrates the twist and publishes the 13-element array."""
        # While shared_autonomy drives the grasp, publish nothing (yield authority).
        if self.grasp_active:
            return
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