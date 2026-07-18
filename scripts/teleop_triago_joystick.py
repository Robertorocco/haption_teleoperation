#!/usr/bin/env python3
"""JOYSTICK-mode teleop: spring-centered handle whose displacement from home is the commanded twist."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Float64MultiArray, Bool, String

import numpy as np
from scipy.spatial.transform import Rotation as R

# Cross-package condition selector: single source of truth for the 2x2x2 study cell.
import triago_control.qp_controller.config as cfg

# TRIAGo <-> Haption base-frame map: 180-deg rotation about Z, its own inverse, valid for linear and axial vectors.
_FRAME_FLIP = np.array([-1.0, -1.0, 1.0])


class TeleopJoystick(Node):
    # Velocity-control input node for all four JOYSTICK study cells.
    def __init__(self):
        super().__init__('teleop_joystick')

        # Hard-error at startup if config.py selects a non-JOYSTICK cell.
        cfg.validate_condition('teleop_triago_joystick', control_mode=cfg.JOYSTICK)

        # Home pose (Haption base frame): fixed position, orientation rebased live onto the gripper.
        self.home_pos = np.array(cfg.JOYSTICK_NEUTRAL_POSITION_M, dtype=float)
        self.neutral_rot = R.from_quat(cfg.JOYSTICK_NEUTRAL_ORIENTATION_XYZW)  # xyzw
        self.home_rot = self.neutral_rot

        self.handle_pos = None
        self.handle_rot = None
        self.handle_vel = np.zeros(6)

        self.ee_pos = None
        self.ee_rot = None
        # Per-arm gripper orientation mapped to the neutral handle: captured once at first activation,
        # saved/restored across arm switches, never re-anchored (keeps home synced through grasps).
        self.grip_ref_rot = {'right': None, 'left': None}

        # While shared autonomy drives a grasp, twist publishing is suspended (home keeps updating).
        self.grasp_active = False

        self.freq = 150.0
        self.dt = 1.0 / self.freq
        self.task_dim = 6.0

        # Direct-drive latch (blending off): an absolute integrated pose target; publishing the live EE
        # instead would make the reference a follower and integrate QP micro-drift into visible creep.
        self.ref_pos = None
        self.ref_rot = None
        self._ref_valid = False
        self.MAX_REF_LEAD_LIN = 0.10   # m    cap on how far the latch may lead the real EE
        self.MAX_REF_LEAD_ANG = 0.35   # rad  same cap for orientation

        self.active_arm = 'right'
        # virtuose/pose is geometry_msgs/Pose (not PoseStamped) -- the wrong type silently receives nothing.
        self.create_subscription(Pose, 'virtuose/pose', self.handle_pose_cb, 10)
        self.create_subscription(Twist, 'virtuose/velocity', self.vel_cb, 10)
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self.ee_cb, 10)
        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_cb, 10)
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)

        # Blending on -> publish pure user twist for the blender; blending off -> drive the QP directly.
        _topic_right = ('/arm_right/user_cartesian_reference' if cfg.BLENDING
                        else '/arm_right/cartesian_reference')
        _topic_left = ('/arm_left/user_cartesian_reference' if cfg.BLENDING
                       else '/arm_left/cartesian_reference')
        self.cmd_pub_right = self.create_publisher(Float64MultiArray, _topic_right, 10)
        self.cmd_pub_left = self.create_publisher(Float64MultiArray, _topic_left, 10)
        self.cmd_pub = self.cmd_pub_right

        # This node owns the live home pose; the force manager's spring targets the same broadcast pose.
        self.home_pub = self.create_publisher(Float64MultiArray, cfg.JOYSTICK_HOME_POSE_TOPIC, 10)

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"[JOYSTICK] CONTROL_MODE={cfg.CONTROL_MODE}, ASSIST_BLENDING={cfg.ASSIST_BLENDING} "
            f"-> publishing user twist on '{_topic_right}' (right) / '{_topic_left}' (left).")
        self.get_logger().info(
            f"[JOYSTICK] Joystick Mode teleop started. Home position (Haption) "
            f"= {self.home_pos.tolist()}; deadband = "
            f"{cfg.JOYSTICK_DEADBAND_LIN*100:.1f} cm / "
            f"{np.degrees(cfg.JOYSTICK_DEADBAND_ANG):.1f} deg. Waiting for handle + EE...")

    # ------------------------------------------------------------------ callbacks
    def active_arm_cb(self, msg):
        """Switches the publishing arm; per-arm home reference is restored, never reset to neutral."""
        if msg.data in ('right', 'left') and msg.data != self.active_arm:
            self.active_arm = msg.data
            self.cmd_pub = self.cmd_pub_right if msg.data == 'right' else self.cmd_pub_left
            # Drop the stale old-arm EE so no home/twist is built from it for a tick.
            self.ee_pos = None
            self.ee_rot = None
            self._ref_valid = False
            restored = self.grip_ref_rot[self.active_arm] is not None
            self.get_logger().info(
                f"[JOYSTICK] Arm -> {msg.data.upper()} "
                f"({'home restored' if restored else 'first activation, home = neutral'}).")

    def grasp_active_cb(self, msg):
        """Suspends twist publishing during autonomous grasp; the home reference is never re-anchored."""
        if msg.data and not self.grasp_active:
            self.grasp_active = True
            self.get_logger().info("[JOYSTICK] Grasp exec: teleop suspended.")
        elif not msg.data and self.grasp_active:
            self.grasp_active = False
            self._ref_valid = False
            self.get_logger().info(
                "[JOYSTICK] Grasp done: teleop resuming (home stayed synced to the gripper).")

    def handle_pose_cb(self, msg):
        """Stores the latest handle pose (Haption base frame)."""
        p = msg.position
        q = msg.orientation
        self.handle_pos = np.array([p.x, p.y, p.z])
        self.handle_rot = R.from_quat([q.x, q.y, q.z, q.w])

    def vel_cb(self, msg):
        """Stores the latest handle 6-DOF spatial velocity (Haption base frame)."""
        self.handle_vel = np.array([
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z])

    def ee_cb(self, msg):
        """Stores the active arm's EE pose; captures the per-arm gripper reference on first sample."""
        if len(msg.data) < 18:
            return
        # ee_real layout: [pos_R(3), vel_R(3), pos_L(3), vel_L(3), rpy_R(3), rpy_L(3)].
        if self.active_arm == 'right':
            self.ee_pos = np.array(msg.data[0:3])
            rpy = np.array(msg.data[12:15])
        else:
            self.ee_pos = np.array(msg.data[6:9])
            rpy = np.array(msg.data[15:18])
        self.ee_rot = R.from_euler('xyz', rpy)
        if self.grip_ref_rot[self.active_arm] is None:
            self.grip_ref_rot[self.active_arm] = self.ee_rot
            self.get_logger().info(
                f"[JOYSTICK] {self.active_arm.upper()} gripper reference captured (home = neutral).")

    # ------------------------------------------------------------------ math
    def _update_home_orientation(self):
        """Rebases home onto the gripper: delta angle compressed by ROT_HOME_SCALE, axis frame-flipped."""
        ref = self.grip_ref_rot[self.active_arm]
        if self.ee_rot is None or ref is None:
            return
        delta_triago = (self.ee_rot * ref.inv()).as_rotvec()
        # Compression fits the gripper's excursion into the Haption's narrower rotational workspace.
        scaled_triago = delta_triago / cfg.JOYSTICK_ROT_HOME_SCALE
        delta_haption = _FRAME_FLIP * scaled_triago
        self.home_rot = R.from_rotvec(delta_haption) * self.neutral_rot

    @staticmethod
    def _deadband_radial(vec, deadband):
        """Removes a radial deadband, continuous at the boundary (magnitude shrinks by the deadband)."""
        n = float(np.linalg.norm(vec))
        if n <= deadband:
            return np.zeros(3)
        return vec * ((n - deadband) / n)

    @staticmethod
    def _clamp_norm(vec, max_norm):
        """Clamps a vector's magnitude, preserving direction."""
        n = float(np.linalg.norm(vec))
        if n > max_norm and n > 1e-9:
            return vec * (max_norm / n)
        return vec

    def _compute_user_twist(self):
        """Maps handle displacement past the deadband to a damped, clamped twist in the TRIAGo frame."""
        d_lin = self.handle_pos - self.home_pos
        eff_lin = self._deadband_radial(d_lin, cfg.JOYSTICK_DEADBAND_LIN)
        if float(np.linalg.norm(eff_lin)) < 1e-9:
            # Damping applies only outside the deadband, so a resting handle commands exactly zero.
            v_haption = np.zeros(3)
        else:
            v_haption = (cfg.JOYSTICK_K_TRANS * eff_lin
                         - cfg.JOYSTICK_DAMP_LIN * self.handle_vel[0:3])
        v_triago = _FRAME_FLIP * v_haption
        v_triago = self._clamp_norm(v_triago, cfg.JOYSTICK_V_MAX_LIN)

        delta_rot = (self.handle_rot * self.home_rot.inv()).as_rotvec()
        eff_ang = self._deadband_radial(delta_rot, cfg.JOYSTICK_DEADBAND_ANG)
        if float(np.linalg.norm(eff_ang)) < 1e-9:
            w_haption = np.zeros(3)
        else:
            w_haption = (cfg.JOYSTICK_K_ROT * eff_ang
                         - cfg.JOYSTICK_DAMP_ANG * self.handle_vel[3:6])
        w_triago = _FRAME_FLIP * w_haption
        w_triago = self._clamp_norm(w_triago, cfg.JOYSTICK_V_MAX_ANG)

        return v_triago, w_triago

    def _advance_ref_latch(self, v_cmd, w_cmd):
        """Advances the absolute latched reference by the twist, with a bounded lead over the real EE."""
        # Re-anchor at the live EE on startup, arm switch, or grasp resume.
        if not self._ref_valid or self.ref_pos is None or self.ref_rot is None:
            self.ref_pos = self.ee_pos.copy()
            self.ref_rot = self.ee_rot
            self._ref_valid = True

        # A still handle (twist=0) holds the latch fixed: an absolute target that opposes drift.
        self.ref_pos = self.ref_pos + v_cmd * self.dt
        if float(np.linalg.norm(w_cmd)) > 1e-9:
            self.ref_rot = R.from_rotvec(w_cmd * self.dt) * self.ref_rot

        # Lead caps stop the reference running away when the arm lags or hits a constraint.
        lead = self.ref_pos - self.ee_pos
        lead_norm = float(np.linalg.norm(lead))
        if lead_norm > self.MAX_REF_LEAD_LIN:
            self.ref_pos = self.ee_pos + lead * (self.MAX_REF_LEAD_LIN / lead_norm)

        ang_lead = (self.ref_rot * self.ee_rot.inv()).as_rotvec()
        ang_norm = float(np.linalg.norm(ang_lead))
        if ang_norm > self.MAX_REF_LEAD_ANG:
            clamped = ang_lead * (self.MAX_REF_LEAD_ANG / ang_norm)
            self.ref_rot = R.from_rotvec(clamped) * self.ee_rot

        return self.ref_pos, self.ref_rot

    # ------------------------------------------------------------------ loop
    def control_loop(self):
        """150 Hz: refresh the home pose, compute the user twist, publish the 13-float protocol."""
        # Home is refreshed even while suspended, so the spring tracks the gripper with no jumps.
        if self.ee_rot is not None:
            self._update_home_orientation()
            self._publish_home_pose()

        if self.grasp_active:
            return
        if self.handle_pos is None or self.handle_rot is None or self.ee_pos is None:
            return

        v_cmd, w_cmd = self._compute_user_twist()

        if cfg.BLENDING:
            # Pose slots carry the live EE so downstream user anchor == EE; the blender integrates.
            p_ref = self.ee_pos
            rot_ref = self.ee_rot
        else:
            # Direct drive: publish the persistent latched pose, not the live EE.
            p_ref, rot_ref = self._advance_ref_latch(v_cmd, w_cmd)

        rpy_ref = rot_ref.as_euler('xyz')
        # Protocol: [pos(3), rpy(3), vel_lin(3), vel_ang(3), task_dim(1)].
        cmd = Float64MultiArray()
        cmd.data = [
            float(p_ref[0]),  float(p_ref[1]),  float(p_ref[2]),
            float(rpy_ref[0]), float(rpy_ref[1]), float(rpy_ref[2]),
            float(v_cmd[0]),  float(v_cmd[1]),  float(v_cmd[2]),
            float(w_cmd[0]),  float(w_cmd[1]),  float(w_cmd[2]),
            float(self.task_dim),
        ]
        self.cmd_pub.publish(cmd)

    def _publish_home_pose(self):
        """Broadcasts the live home pose (Haption base frame) for the force manager's spring."""
        q = self.home_rot.as_quat()  # xyzw
        msg = Float64MultiArray()
        msg.data = [
            float(self.home_pos[0]), float(self.home_pos[1]), float(self.home_pos[2]),
            float(q[0]), float(q[1]), float(q[2]), float(q[3]),
        ]
        self.home_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopJoystick()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
