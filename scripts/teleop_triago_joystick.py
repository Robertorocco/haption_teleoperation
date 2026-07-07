#!/usr/bin/env python3
"""teleop_triago_joystick -- Joystick Mode teleoperation for BLENDING.

This is the blending-mode replacement for teleop_triago_clutch.py. The previous
BLENDING design integrated the raw Haption twist into a pose AND rendered an
F_sync tether force back onto the handle; that force displaced the handle, the
displacement was read back as user twist, and the loop went unstable (plus the
clutch interacted badly with it). Joystick Mode removes that coupling entirely:
the handle is spring-centered to a FIXED home pose (the spring lives in
haptic_force_manager_blending_tutorial.py), and this node reads ONLY the handle's
DISPLACEMENT from that home and maps it to a pure Cartesian twist. There is no
pose integration and no clutch here -- releasing the handle recenters it, which
is a zero command.

Handle mechanics (per axis):
    displacement d = handle_pose - home_pose        (Haption base frame)
    v = K * (d with a radial deadband removed)       (magnitude ~ distance)
    v mapped Haption -> TRIAGo base via the 180-deg-Z flip (negate X, negate Y).
A displacement inside JOYSTICK_DEADBAND_{LIN,ANG} yields exactly zero twist.

Dynamic home orientation (kinematic-mismatch handling):
    The gripper may be posed very differently from the handle (e.g. a top-down
    grasp). So the home ORIENTATION tracks the gripper: per arm, the gripper's
    orientation is captured ONCE at that arm's first activation (its home then ==
    neutral), and thereafter the home rotates with the gripper's delta from that
    reference, with the rotation ANGLE scaled DOWN by JOYSTICK_ROT_HOME_SCALE
    (gripper 90 deg -> handle ~69 deg) because the Haption's rotational workspace
    is more restrictive. This scaling applies ONLY here to the home pose, never to
    the commanded twist. The home POSITION stays fixed. The reference is SAVED /
    RESTORED across arm switches but is NEVER re-anchored mid-session (in
    particular not after an autonomous grasp) -- the home is updated every tick,
    including while suspended during grasp execution, so it stays continuously
    synchronized with the gripper with no jumps at any state transition.

Ownership: this node is the single source of truth for the live home pose and
publishes it on cfg.JOYSTICK_HOME_POSE_TOPIC so the force manager renders its
spring toward the exact same target (no drift between the two nodes).

Outputs (BLENDING=True): the pure user twist on /arm_*/user_cartesian_reference
(13-float protocol [pos(3), rpy(3), vel_lin(3), vel_ang(3), task_dim]); the pose
slots carry the live EE pose so downstream current_T_user == current_T_EE. In
BLENDING=False this node is not the intended teleop (teleop_triago_clutch.py is);
it still routes to /arm_*/cartesian_reference and warns.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Float64MultiArray, Bool, String

import numpy as np
from scipy.spatial.transform import Rotation as R

# Single source of truth for the shared-autonomy blending architecture (also read
# by triago_control/scripts/qp_arm_teleop/main_shared_autonomy.py and the haptic
# force manager). See cfg.BLENDING's docstring in config.py.
import triago_control.qp_controller.config as cfg

# TRIAGo <-> Haption base-frame map: a pure 180-deg rotation about Z. Applied to a
# 3-vector this negates X and Y and keeps Z; it is its own inverse and a proper
# rotation, so it maps both linear vectors and rotation vectors (axial) identically.
_FRAME_FLIP = np.array([-1.0, -1.0, 1.0])


class TeleopJoystick(Node):
    def __init__(self):
        super().__init__('teleop_joystick')

        # --- Home pose (Haption base frame) ---
        self.home_pos = np.array(cfg.JOYSTICK_NEUTRAL_POSITION_M, dtype=float)
        self.neutral_rot = R.from_quat(cfg.JOYSTICK_NEUTRAL_ORIENTATION_XYZW)  # xyzw
        self.home_rot = self.neutral_rot  # updated live to track the gripper

        # --- Live handle state (Haption base frame) ---
        self.handle_pos = None
        self.handle_rot = None
        self.handle_vel = np.zeros(6)   # 6-DOF spatial velocity (for viscous twist damping)

        # --- Robot EE state (TRIAGo base frame) ---
        self.ee_pos = None
        self.ee_rot = None
        # PER-ARM gripper reference orientation that maps to the neutral handle quat.
        # Captured from an arm's gripper the FIRST time that arm becomes active (its
        # home then == neutral), and SAVED/RESTORED across arm switches -- switching
        # away keeps the arm's reference in this dict, switching back reuses it so the
        # arm resumes its own home pose instead of resetting to neutral. It is NEVER
        # cleared/re-anchored after the first capture (see grasp_active_cb) so the home
        # stays continuously synced to the gripper through grasp execution.
        self.grip_ref_rot = {'right': None, 'left': None}

        # --- Authority handover ---
        # While shared_autonomy drives a grasp autonomously it publishes
        # grasp_active=True; we stop publishing the user twist. The home orientation
        # keeps updating every tick against the persistent reference (no re-anchor),
        # so the handle stays synced to the gripper and never snaps when teleop resumes.
        self.grasp_active = False

        # --- Parameters ---
        self.freq = 150.0
        self.dt = 1.0 / self.freq
        self.task_dim = 6.0

        # --- ROS 2 interfaces ---
        self.active_arm = 'right'
        # Handle Cartesian pose (position + orientation) drives the displacement.
        # NOTE: virtuose_server_node publishes virtuose/pose as geometry_msgs/Pose
        # (NOT PoseStamped) -- the wrong type silently receives nothing.
        self.create_subscription(Pose, 'virtuose/pose', self.handle_pose_cb, 10)
        # Handle velocity, for the viscous damping term on the commanded twist.
        self.create_subscription(Twist, 'virtuose/velocity', self.vel_cb, 10)
        # EE pose: reference orientation + the pose slots of the outgoing message.
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self.ee_cb, 10)
        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_cb, 10)
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)

        _topic_right = ('/arm_right/user_cartesian_reference' if cfg.BLENDING
                        else '/arm_right/cartesian_reference')
        _topic_left = ('/arm_left/user_cartesian_reference' if cfg.BLENDING
                       else '/arm_left/cartesian_reference')
        self.cmd_pub_right = self.create_publisher(Float64MultiArray, _topic_right, 10)
        self.cmd_pub_left = self.create_publisher(Float64MultiArray, _topic_left, 10)
        self.cmd_pub = self.cmd_pub_right

        # Live home pose broadcast to the force manager (single source of truth).
        self.home_pub = self.create_publisher(Float64MultiArray, cfg.JOYSTICK_HOME_POSE_TOPIC, 10)

        self.timer = self.create_timer(self.dt, self.control_loop)

        if not cfg.BLENDING:
            self.get_logger().warn(
                "[JOYSTICK] cfg.BLENDING=False -- this joystick node is intended for "
                "BLENDING mode. Publishing directly to /arm_*/cartesian_reference.")
        self.get_logger().info(
            f"[JOYSTICK] Joystick Mode teleop started. Home position (Haption) "
            f"= {self.home_pos.tolist()}; deadband = "
            f"{cfg.JOYSTICK_DEADBAND_LIN*100:.1f} cm / "
            f"{np.degrees(cfg.JOYSTICK_DEADBAND_ANG):.1f} deg. Waiting for handle + EE...")

    # ------------------------------------------------------------------ callbacks
    def active_arm_cb(self, msg):
        """Switch which arm the twist is published to (per-arm home saved/restored)."""
        if msg.data in ('right', 'left') and msg.data != self.active_arm:
            self.active_arm = msg.data
            self.cmd_pub = self.cmd_pub_right if msg.data == 'right' else self.cmd_pub_left
            # Drop the stale (old-arm) EE so we don't build a bad home/twist for one
            # tick; the next ee_cb refills it for the new arm. The new arm's grip_ref
            # is RESTORED from the dict (or captured on that next sample if this is
            # its first activation) -- never force-reset to neutral.
            self.ee_pos = None
            self.ee_rot = None
            restored = self.grip_ref_rot[self.active_arm] is not None
            self.get_logger().info(
                f"[JOYSTICK] Arm -> {msg.data.upper()} "
                f"({'home restored' if restored else 'first activation, home = neutral'}).")

    def grasp_active_cb(self, msg):
        """Suspend twist publishing while the grasp SM drives the arm.

        We DELIBERATELY do NOT re-anchor grip_ref_rot on the falling edge. The
        home orientation is updated every tick (even while grasp_active, see
        control_loop) as a scaled DELTA from this arm's persistent reference, so
        it already tracked the gripper smoothly all the way through the grasp
        (success, failure, or abort). Resetting the reference here would snap the
        delta to zero -> home would JUMP back to neutral and the spring would yank
        the handle. Keeping the reference means the handle stays continuously
        synchronized with the current gripper orientation across the whole grasp.
        """
        if msg.data and not self.grasp_active:
            self.grasp_active = True
            self.get_logger().info("[JOYSTICK] Grasp exec: teleop suspended.")
        elif not msg.data and self.grasp_active:
            self.grasp_active = False
            self.get_logger().info(
                "[JOYSTICK] Grasp done: teleop resuming (home stayed synced to the gripper).")

    def handle_pose_cb(self, msg):
        """Latest Haption handle pose (position in m, orientation quat), Haption base frame."""
        p = msg.position
        q = msg.orientation
        self.handle_pos = np.array([p.x, p.y, p.z])
        self.handle_rot = R.from_quat([q.x, q.y, q.z, q.w])

    def vel_cb(self, msg):
        """Latest Haption handle 6-DOF spatial velocity (Haption base frame)."""
        self.handle_vel = np.array([
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z])

    def ee_cb(self, msg):
        """Active arm's EE pose. Layout: [pos_R(3), vel_R(3), pos_L(3), vel_L(3), rpy_R(3), rpy_L(3)]."""
        if len(msg.data) < 18:
            return
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
        """Rebase the home orientation onto the ACTIVE arm's current gripper orientation.

        home_rot = map(scaled gripper delta) * neutral_rot, where the gripper delta
        R_grip * R_grip_ref^-1 (TRIAGo base frame, R_grip_ref = this arm's saved
        reference) has its ANGLE divided by JOYSTICK_ROT_HOME_SCALE and its AXIS
        mapped into the Haption base frame.
        """
        ref = self.grip_ref_rot[self.active_arm]
        if self.ee_rot is None or ref is None:
            return
        delta_triago = (self.ee_rot * ref.inv()).as_rotvec()  # base-frame axis*angle
        scaled_triago = delta_triago / cfg.JOYSTICK_ROT_HOME_SCALE
        delta_haption = _FRAME_FLIP * scaled_triago            # map axis to Haption frame
        self.home_rot = R.from_rotvec(delta_haption) * self.neutral_rot

    @staticmethod
    def _deadband_radial(vec, deadband):
        """Remove a radial deadband from a vector, continuous at the boundary.

        Returns zero inside the deadband, otherwise the vector shrunk so its
        magnitude is (||vec|| - deadband) along the same direction.
        """
        n = float(np.linalg.norm(vec))
        if n <= deadband:
            return np.zeros(3)
        return vec * ((n - deadband) / n)

    @staticmethod
    def _clamp_norm(vec, max_norm):
        n = float(np.linalg.norm(vec))
        if n > max_norm and n > 1e-9:
            return vec * (max_norm / n)
        return vec

    def _compute_user_twist(self):
        """Handle displacement from home -> Cartesian twist in the TRIAGo base frame.

        Twist magnitude is proportional to the displacement past the deadband, minus
        a viscous damping term (-DAMP * handle_velocity) that smooths quick handle
        motions. Damping is applied ONLY outside the deadband, so a handle resting or
        oscillating near home still commands exactly zero (deadband guarantee kept).
        """
        # --- Linear: displacement of the handle position from home ---
        d_lin = self.handle_pos - self.home_pos                       # Haption frame
        eff_lin = self._deadband_radial(d_lin, cfg.JOYSTICK_DEADBAND_LIN)
        if float(np.linalg.norm(eff_lin)) < 1e-9:
            v_haption = np.zeros(3)                                    # inside deadband -> zero (no damping)
        else:
            v_haption = (cfg.JOYSTICK_K_TRANS * eff_lin
                         - cfg.JOYSTICK_DAMP_LIN * self.handle_vel[0:3])
        v_triago = _FRAME_FLIP * v_haption
        v_triago = self._clamp_norm(v_triago, cfg.JOYSTICK_V_MAX_LIN)

        # --- Angular: rotational displacement of the handle from the home orientation ---
        delta_rot = (self.handle_rot * self.home_rot.inv()).as_rotvec()  # Haption frame axis*angle
        eff_ang = self._deadband_radial(delta_rot, cfg.JOYSTICK_DEADBAND_ANG)
        if float(np.linalg.norm(eff_ang)) < 1e-9:
            w_haption = np.zeros(3)                                    # inside deadband -> zero (no damping)
        else:
            w_haption = (cfg.JOYSTICK_K_ROT * eff_ang
                         - cfg.JOYSTICK_DAMP_ANG * self.handle_vel[3:6])
        w_triago = _FRAME_FLIP * w_haption
        w_triago = self._clamp_norm(w_triago, cfg.JOYSTICK_V_MAX_ANG)

        return v_triago, w_triago

    # ------------------------------------------------------------------ loop
    def control_loop(self):
        # Always keep the force manager's home target fresh (even while suspended),
        # so the spring recenters the handle onto the current gripper orientation.
        if self.ee_rot is not None:
            self._update_home_orientation()
            self._publish_home_pose()

        if self.grasp_active:
            return  # yield authority to the autonomous grasp
        if self.handle_pos is None or self.handle_rot is None or self.ee_pos is None:
            return  # need the handle + EE before commanding

        v_cmd, w_cmd = self._compute_user_twist()

        # Pose slots carry the live EE pose so downstream current_T_user == current_T_EE
        # (belief + guidance are EE-anchored in joystick mode).
        rpy_ee = self.ee_rot.as_euler('xyz')
        cmd = Float64MultiArray()
        cmd.data = [
            float(self.ee_pos[0]), float(self.ee_pos[1]), float(self.ee_pos[2]),  # 0:3 position
            float(rpy_ee[0]),      float(rpy_ee[1]),      float(rpy_ee[2]),        # 3:6 rpy
            float(v_cmd[0]),       float(v_cmd[1]),       float(v_cmd[2]),         # 6:9 linear twist
            float(w_cmd[0]),       float(w_cmd[1]),       float(w_cmd[2]),         # 9:12 angular twist
            float(self.task_dim),                                                 # 12: task-dim flag
        ]
        self.cmd_pub.publish(cmd)

    def _publish_home_pose(self):
        """Broadcast the live home pose (Haption base frame) for the force manager."""
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
