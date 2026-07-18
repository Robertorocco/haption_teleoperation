#!/usr/bin/env python3
"""CLUTCH-mode teleop: integrates the Haption handle twist into a pose reference, with indexing."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Bool, String

import numpy as np
from scipy.spatial.transform import Rotation as R

# Cross-package condition selector: single source of truth for the 2x2x2 study cell.
import triago_control.qp_controller.config as cfg


class TeleopClutch(Node):
    # Position-control input node for all four CLUTCH study cells.
    def __init__(self):
        super().__init__('teleop_clutch')

        # Hard-error at startup if config.py selects a non-CLUTCH cell.
        cfg.validate_condition('teleop_triago_clutch', control_mode=cfg.CLUTCH)

        self.initialized = False
        self.ref_pos = np.zeros(3)
        self.ref_rot = R.identity()

        self.v_cmd = np.zeros(3)
        self.w_cmd = np.zeros(3)

        self.clutch_engaged = False

        # While shared autonomy drives an autonomous grasp, teleop is suspended and re-anchors on resume.
        self.grasp_active = False

        self.freq = 150.0  # Hz
        self.dt = 1.0 / self.freq

        self.K_trans = 1.0  # translational scale factor
        self.K_rot = 1.0    # rotational scale factor

        # 6.0 = full 6D tracking | 5.0 = free rotation about the approach axis.
        self.task_dim = 6.0

        self.create_subscription(Twist, 'virtuose/velocity', self.twist_callback, 10)
        self.create_subscription(Bool, 'virtuose/button_right', self.button_callback, 10)

        # Real EE pose, used once per (re)anchor to start integration from where the robot is.
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self.ee_callback, 10)

        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_callback, 10)

        # Blending on -> publish pure user intent; blending off -> drive the QP reference directly.
        self.active_arm = 'right'
        _topic_right = ('/arm_right/user_cartesian_reference' if cfg.ASSIST_BLENDING
                        else '/arm_right/cartesian_reference')
        _topic_left = ('/arm_left/user_cartesian_reference' if cfg.ASSIST_BLENDING
                       else '/arm_left/cartesian_reference')
        self.cmd_pub_right = self.create_publisher(Float64MultiArray, _topic_right, 10)
        self.cmd_pub_left = self.create_publisher(Float64MultiArray, _topic_left, 10)
        self.cmd_pub = self.cmd_pub_right

        self.get_logger().info(
            f"[TELEOP] CONTROL_MODE={cfg.CONTROL_MODE}, ASSIST_BLENDING={cfg.ASSIST_BLENDING} "
            f"-> publishing user reference on '{_topic_right}' (right) / '{_topic_left}' (left).")

        # Arm switching is decided solely by shared autonomy; this node follows.
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)

        self.timer = self.create_timer(self.dt, self.integration_loop)

        self.get_logger().info("Teleop Clutch started. Waiting for initial TRIAGo pose...")

    def active_arm_cb(self, msg):
        """Switches the publishing arm and forces a re-anchor at the new arm's pose."""
        if msg.data in ('right', 'left') and msg.data != self.active_arm:
            self.active_arm = msg.data
            self.cmd_pub = self.cmd_pub_right if msg.data == 'right' else self.cmd_pub_left
            self.initialized = False
            self.get_logger().info(f"[TELEOP] Arm switched to {msg.data.upper()}. Re-anchoring.")

    def ee_callback(self, msg):
        """Anchors the integration at the robot's current EE pose (once per re-anchor)."""
        if not self.initialized:
            try:
                # ee_real layout: [pos_R(3), vel_R(3), pos_L(3), vel_L(3), rpy_R(3), rpy_L(3)].
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
        """Maps the Haption twist into the TRIAGo base frame (180-deg rotation about Z: negate X, Y)."""
        self.v_cmd[0] = -msg.linear.x * self.K_trans
        self.v_cmd[1] = -msg.linear.y * self.K_trans
        self.v_cmd[2] =  msg.linear.z * self.K_trans

        self.w_cmd[0] = -msg.angular.x * self.K_rot
        self.w_cmd[1] = -msg.angular.y * self.K_rot
        self.w_cmd[2] =  msg.angular.z * self.K_rot

    def button_callback(self, msg):
        """Updates the clutch state, logging only on transitions."""
        if msg.data != self.clutch_engaged:
            self.clutch_engaged = msg.data
            if self.clutch_engaged:
                self.get_logger().info(" CLUTCH ENGAGED: Robot frozen. Reposition your hand freely.")
            else:
                self.get_logger().info(" CLUTCH RELEASED: Teleoperation tracking resumed.")

    def grasp_active_callback(self, msg):
        """Suspends teleop during autonomous grasp; re-anchors at the post-grasp pose on resume."""
        if msg.data and not self.grasp_active:
            self.grasp_active = True
            self.get_logger().info(" GRASP EXEC: teleop suspended (arm driven autonomously).")
        elif not msg.data and self.grasp_active:
            self.grasp_active = False
            self.initialized = False
            self.get_logger().info(" GRASP DONE: re-anchoring, teleop resuming.")

    def integration_loop(self):
        """150 Hz: integrates the twist into the pose reference and publishes the 13-float protocol."""
        # Yield authority while shared autonomy drives the grasp.
        if self.grasp_active:
            return
        if not self.initialized:
            return

        if not self.clutch_engaged:
            # P_new = P_old + v*dt; R_new = exp(w*dt) * R_old.
            self.ref_pos += self.v_cmd * self.dt

            delta_rot = R.from_rotvec(self.w_cmd * self.dt)
            self.ref_rot = delta_rot * self.ref_rot

            pub_v = self.v_cmd
            pub_w = self.w_cmd

        # Clutch held: pose frozen, zero feed-forward twist (indexing).
        else:
            pub_v = np.zeros(3)
            pub_w = np.zeros(3)

        rpy_ref = self.ref_rot.as_euler('xyz', degrees=False)

        # Protocol: [pos(3), rpy(3), vel_lin(3), vel_ang(3), task_dim(1)].
        cmd_msg = Float64MultiArray()
        cmd_msg.data = [
            float(self.ref_pos[0]), float(self.ref_pos[1]), float(self.ref_pos[2]),
            float(rpy_ref[0]),      float(rpy_ref[1]),      float(rpy_ref[2]),
            float(pub_v[0]),        float(pub_v[1]),        float(pub_v[2]),
            float(pub_w[0]),        float(pub_w[1]),        float(pub_w[2]),
            float(self.task_dim)
        ]

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
