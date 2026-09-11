#!/usr/bin/env python3
"""CLUTCH sync-only baseline (F=0, B=0): renders only the F_sync tether, no predictive assistance."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench, Twist, Pose
from std_msgs.msg import Bool, Float64MultiArray, String
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from collections import deque
import matplotlib.pyplot as plt
import matplotlib

# Cross-package condition selector: single source of truth for the 2x3 study cell.
import triago_control.qp_controller.config as cfg

# TkAgg keeps Matplotlib off the ROS spin thread.
matplotlib.use('TkAgg')

# TRIAGo <-> Haption base-frame map: 180-deg rotation about Z, its own inverse, valid for
# linear and axial vectors alike (so it may be applied directly to a rotation vector).
_FRAME_FLIP = np.array([-1.0, -1.0, 1.0])

# TCP approach axis: the gripper's local +X, the convention used across the whole platform.
_APPROACH_AXIS = np.array([1.0, 0.0, 0.0])


class HapticForceManagerNoGuidance(Node):
    # Baseline force manager: F_sync tether + unified cues, no guidance/fixture/CBF forces.
    def __init__(self):
        super().__init__('haptic_force_manager_noguidance')

        # Hard-error at startup unless config.py selects the CLUTCH sync-only cell.
        cfg.validate_condition('haptic_force_manager_C',
                               control_mode=cfg.CLUTCH, feedback=False, blending=False)

        self.pos_target = None
        self.rot_target = None
        self.pos_real = None
        self.rot_real = None
        self.vel_real = np.zeros(3)     # active-arm EE linear velocity (from ee_real)
        self.vel_haption = np.zeros(6)  # handle 6D spatial velocity (Haption frame)

        # Sync spring gains, unified across all clutch cells (Kd=0: global damper supplies viscosity).
        self.Kp_sync = float(self.declare_parameter('Kp_sync', 15.0).value)  # N/m
        self.Kd_sync = 0.0         # Ns/m
        # Live-tunable so the spring/damping pair can be swept without a rebuild.
        self.Kp_sync_ang = float(self.declare_parameter('Kp_sync_ang', 0.1).value)  # Nm/rad
        self._last_err_log_time = 0.0
        # Damping is rendered by virtuose_server_node instead: a damper is only passive if applied
        # in the same tick its velocity was measured, which a separate process cannot guarantee.
        self.ENABLE_GLOBAL_DAMPING_LIN = False

        # During autonomous grasp no force is impressed; only a vibration cue is rendered.
        self.grasp_active = False
        self._grasp_start_pos = None
        self.GRASP_FOLLOW_KP = 15.0     # N/m
        self.GRASP_FOLLOW_KD = 80.0     # Ns/m
        self.GRASP_VIB_AMP = 0.009  # Nm constant square-wave buzz during the whole grasp
        self.grasp_vib_toggle = 1.0

        # Clutch press freezes the wrench at 50% (cognitive grounding).
        self.is_clutching = False
        self.was_clutching_last_frame = False
        self.f_clutch_frozen = np.zeros(6)
        self.pos_haption = None
        self.rot_haption = None
        # Live-tunable align stiffness. Reference points: clutch sync=0.1 Nm/rad, joystick
        # homing=0.75 Nm/rad; this gain is stable only below roughly 0.9 Nm/rad.
        self.K_align = float(self.declare_parameter('K_align', 0.3).value)  # Nm/rad
        # Soft ceiling below the 0.5 Nm device clip: a stiff spring that hard-clips becomes
        # bang-bang (max torque flipping sign across the target), which self-excites.
        self.MAX_ALIGN_TORQUE = 0.35  # Nm
        # Handle<->gripper orientation correspondence, captured once and never re-anchored: CLUTCH
        # teleop integrates orientation incrementally, so this anchor is the only absolute reference.
        self._align_ref_real = None
        self._align_ref_haption = None
        # Handle-local image of the gripper's approach axis, fixed once with the anchor pair.
        self._align_axis_local = None
        # True: clutch renders zero linear force plus a live torque aligning the handle with the
        # gripper. False: the whole wrench is frozen at 50% for the press duration instead.
        self.ENABLE_CLUTCH_ALIGN = True
        self.CLUTCH_ALIGN_FADE_MARGIN = 0.35  # rad: align torque fades out inside this joint-limit margin

        # Gripper delta -> handle delta gain: teleop_triago_clutch integrates the handle twist into
        # the robot reference through its K_rot, so the inverse map divides by the same factor.
        self.CLUTCH_ROT_SCALE = 1.0  # must mirror teleop_triago_clutch.K_rot
        # Startup homing pins the handle at a fixed neutral pose (joystick spring law), making the
        # handle<->gripper correspondence deterministic; gains are soft since nobody may be holding it yet.
        self.HOME_POS = np.array(cfg.JOYSTICK_NEUTRAL_POSITION_M, dtype=float)
        self.HOME_ROT = R.from_quat(cfg.JOYSTICK_NEUTRAL_ORIENTATION_XYZW)  # xyzw
        self.HOMING_KP_LIN = float(self.declare_parameter('K_home_lin', 20.0).value)  # N/m
        self.HOMING_KP_ANG = float(self.declare_parameter('K_home_ang', 0.5).value)   # Nm/rad
        # Soft ceilings far under the device clip, approached through a tanh on the magnitude so
        # the pull never slams when the handle starts on the far side of the workspace.
        self.MAX_HOMING_FORCE = 2.0    # N
        self.MAX_HOMING_TORQUE = 0.15  # Nm
        self.HOMING_FADE_S = 1.0       # whole wrench fades in over this window: t=0 is never a step
        self.HOMING_TIMEOUT_S = 5.0    # give up rather than leave teleop frozen forever
        self._homing_t0 = None
        self.HOMING_TOL_LIN = 0.02   # m
        self.HOMING_TOL_ANG = 0.10   # rad
        self.HOMING_SETTLE_TICKS = 30  # 0.2 s held inside tolerance before the phase latches off
        self._homing_ticks_in_tol = 0
        # One-shot: latches when homing ends and never re-arms; skipped when clutch-align is off.
        self._homing_done = not self.ENABLE_CLUTCH_ALIGN

        # Global viscous damping (impedance-device stability), unified across clutch cells.
        self.Kd_global_lin = 0.35  # Ns/m
        self.Kd_global_ang = 0.05  # Nms/rad
        # Caps the velocity feeding Kd_global_ang: a brief low-inertia spin transient (e.g. a
        # button press nudging the wrist gimbal) should not be amplified into a torque step.
        self.VEL_DAMP_CLAMP_ANG = 3.0  # rad/s
        # 1-pole LPF on angular velocity before the damping law: raw differentiated velocity is only
        # passive up to the device's own mechanical damping margin; filtering above resonance restores it.
        self._vel_ang_filt = np.zeros(3)
        self.VEL_FILT_ALPHA_ANG = 0.75  # ~7 Hz cutoff at 150 Hz
        # Angular damping is rendered by virtuose_server_node instead (same-tick passivity can't
        # hold across a process boundary); tune it there via ros2 param set damping_ang.
        self.ENABLE_GLOBAL_DAMPING_ANG = False

        # Device safety clip and unified authority cap.
        self.MAX_FORCE = 5.0       # N
        self.MAX_TORQUE = 0.5      # Nm
        self.MAX_TOTAL_FORCE = 5.0
        self.MAX_TOTAL_TORQUE = 0.5

        # Arm the force is computed for (follows shared autonomy's active arm).
        self.active_arm = 'right'

        # Haption joint positions and calibrated limits (for the joint-limit cue).
        self.joint_pos = np.zeros(6)
        self.joint_min = np.array([-0.785282, -1.5709, 0.792704, -2.39339, -1.02312, -2.22872])
        self.joint_max = np.array([0.784393, -0.00157491, 2.49551, 2.35378, 0.879614, 2.21327])

        self.LIMIT_OUTER = 0.10       # rad: margin where the joint-limit cue buzzes
        self.vib_toggle = 1.0         # sign flip every frame -> 75 Hz square wave

        # Joint-limit cue: same continuous while-condition-holds pattern as the
        # joystick out-of-deadzone buzz -- no timer, no re-arm state.
        self.LIMIT_VIB_AMP = 0.009  # Nm

        # Plot buffers (10 s window at 150 Hz), guarded by a lock shared with the UI thread.
        self.plot_lock = threading.Lock()
        self.plot_window_sec = 10.0
        self.buffer_size = int(150 * self.plot_window_sec)
        self.t_data = deque(maxlen=self.buffer_size)
        self.start_time = time.time()
        self.sync_F = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.sync_T = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.tot_F = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.tot_T = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.freq_data = deque(maxlen=self.buffer_size)
        self._own_freq_lpf = 0.0
        self._own_last_time = None

        # No guidance/CBF subscriptions: this baseline renders only F_sync plus cues.
        self.create_subscription(Float64MultiArray, '/arm_right/cartesian_reference', self.target_cb, 10)
        self.create_subscription(Float64MultiArray, '/arm_left/cartesian_reference', self.target_cb_left, 10)
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self.real_cb, 10)
        self.create_subscription(Twist, 'virtuose/velocity', self.vel_cb, 10)
        self.create_subscription(Bool, 'virtuose/button_right', self.button_cb, 10)
        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_cb, 10)
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)
        self.create_subscription(Float64MultiArray, 'virtuose/articular_position', self.joint_cb, 10)
        self.create_subscription(Pose, 'virtuose/pose', self.haption_pose_cb, 10)

        self.force_pub = self.create_publisher(Wrench, 'virtuose/force_cmd', 10)
        # Gates the teleop node's integration so the homing motion never drives the robot.
        self.homing_pub = self.create_publisher(Bool, 'device/homing_active', 10)

        self.dt = 1.0 / 150.0
        self.timer = self.create_timer(self.dt, self.control_loop)

        # Live Matplotlib windows: on by default, disable with -p plot:=false.
        self.plot_enabled = bool(self.declare_parameter('plot', True).value)
        if self.plot_enabled:
            self.setup_plot()
        self.get_logger().info(
            "Haptic Force Manager (NO-GUIDANCE baseline) started: F_sync only, "
            f"doubled tether (Kp_sync={self.Kp_sync}, Kp_sync_ang={self.Kp_sync_ang}).")

    # =========================
    # PLOT SETUP & UPDATE
    # =========================
    def setup_plot(self):
        """Initializes the live plot: F_sync, published total, and loop rate."""
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 2, figsize=(11, 8))
        self.fig.canvas.manager.set_window_title('Haptic Force (NO-GUIDANCE: F_sync only)')
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']

        self.lines_sync_F = []
        ax = self.axs[0, 0]
        ax.set_title("F_sync - FORCE (N)", fontsize=10)
        ax.set_ylabel("Force (N)"); ax.grid(True, linestyle='--', alpha=0.6)
        for i in range(3):
            self.lines_sync_F.append(ax.plot([], [], color=colors[i], label=f"F{labels[i]}")[0])
        ax.legend(loc='upper left', fontsize=8, ncol=3)

        self.lines_sync_T = []
        ax = self.axs[0, 1]
        ax.set_title("F_sync - TORQUE (Nm)", fontsize=10)
        ax.set_ylabel("Torque (Nm)"); ax.grid(True, linestyle='--', alpha=0.6)
        for i in range(3):
            self.lines_sync_T.append(ax.plot([], [], color=colors[i], label=f"T{labels[i]}")[0])
        ax.legend(loc='upper left', fontsize=8, ncol=3)

        self.lines_tot_F = []
        ax = self.axs[1, 0]
        ax.set_title("Published TOTAL - FORCE (N)", fontsize=10)
        ax.set_ylabel("Force (N)"); ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(self.MAX_FORCE, color='k', linestyle='--', linewidth=1.0, alpha=0.6, label='±max')
        ax.axhline(-self.MAX_FORCE, color='k', linestyle='--', linewidth=1.0, alpha=0.6)
        for i in range(3):
            self.lines_tot_F.append(ax.plot([], [], color=colors[i], label=f"F{labels[i]}")[0])
        ax.legend(loc='upper left', fontsize=8, ncol=4)

        self.lines_tot_T = []
        ax = self.axs[1, 1]
        ax.set_title("Published TOTAL - TORQUE (Nm)", fontsize=10)
        ax.set_ylabel("Torque (Nm)"); ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(self.MAX_TORQUE, color='k', linestyle='--', linewidth=1.0, alpha=0.6, label='±max')
        ax.axhline(-self.MAX_TORQUE, color='k', linestyle='--', linewidth=1.0, alpha=0.6)
        for i in range(3):
            self.lines_tot_T.append(ax.plot([], [], color=colors[i], label=f"T{labels[i]}")[0])
        ax.legend(loc='upper left', fontsize=8, ncol=4)

        ax = self.axs[2, 0]
        ax.set_title("Force Manager Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Hz"); ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 180); ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(150, color='g', linestyle='--', linewidth=1.0, alpha=0.7, label='target 150Hz')
        self.line_freq, = ax.plot([], [], color='#9467bd', linewidth=1.5, label='HFM freq')
        ax.legend(loc='upper left', fontsize=8)
        self.axs[2, 1].axis('off')

        self.fig.tight_layout()
        plt.show(block=False)

    def update_plot(self):
        """Snapshots buffers under the lock and refreshes the Matplotlib UI."""
        with self.plot_lock:
            if len(self.t_data) == 0:
                return
            t_list = list(self.t_data)
            sF = [list(self.sync_F[i]) for i in range(3)]
            sT = [list(self.sync_T[i]) for i in range(3)]
            tF = [list(self.tot_F[i]) for i in range(3)]
            tT = [list(self.tot_T[i]) for i in range(3)]
            freq_list = list(self.freq_data)

        current_t = t_list[-1]
        win = (current_t - self.plot_window_sec, current_t)

        for i in range(3):
            self.lines_sync_F[i].set_data(t_list, sF[i])
            self.lines_sync_T[i].set_data(t_list, sT[i])
            self.lines_tot_F[i].set_data(t_list, tF[i])
            self.lines_tot_T[i].set_data(t_list, tT[i])
        for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            self.axs[r, c].set_xlim(*win)
            self.axs[r, c].relim()
            self.axs[r, c].autoscale_view(scalex=False, scaley=True)

        n = min(len(t_list), len(freq_list))
        self.line_freq.set_data(t_list[:n], freq_list[:n])
        self.axs[2, 0].set_xlim(*win)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # =========================
    # CALLBACKS
    # =========================
    def haption_pose_cb(self, msg):
        """Stores the handle pose (geometry_msgs/Pose) for the homing spring and alignment torque."""
        p = msg.position
        q = msg.orientation
        self.pos_haption = np.array([p.x, p.y, p.z])
        self.rot_haption = R.from_quat([q.x, q.y, q.z, q.w])

    def button_cb(self, msg):
        """Updates the clutching state from the Virtuose button."""
        self.is_clutching = msg.data

    def grasp_active_cb(self, msg):
        """Tracks whether shared autonomy is autonomously driving a grasp."""
        self.grasp_active = bool(msg.data)

    def active_arm_cb(self, msg):
        """Switches which arm's EE data is used for force computation."""
        if msg.data in ('right', 'left') and msg.data != self.active_arm:
            self.active_arm = msg.data
            self.get_logger().info(f"[FORCE MGR] Active arm switched to {msg.data.upper()}")

    def joint_cb(self, msg):
        """Updates the 6-DoF Haption joint positions from the encoders."""
        if len(msg.data) >= 6:
            self.joint_pos = np.array(msg.data[0:6])

    def target_cb(self, msg):
        """Updates the target Cartesian pose (right arm reference)."""
        if self.active_arm != 'right':
            return
        if len(msg.data) >= 6:
            self.pos_target = np.array(msg.data[0:3])
            self.rot_target = R.from_euler('xyz', np.array(msg.data[3:6]), degrees=False)

    def target_cb_left(self, msg):
        """Updates the target Cartesian pose (left arm reference)."""
        if self.active_arm != 'left':
            return
        if len(msg.data) >= 6:
            self.pos_target = np.array(msg.data[0:3])
            self.rot_target = R.from_euler('xyz', np.array(msg.data[3:6]), degrees=False)

    def real_cb(self, msg):
        """Updates the real Cartesian pose + linear velocity of the active arm."""
        if len(msg.data) >= 18:
            if self.active_arm == 'right':
                self.pos_real = np.array(msg.data[0:3])
                self.vel_real = np.array(msg.data[3:6])
                rpy = np.array(msg.data[12:15])
            else:
                self.pos_real = np.array(msg.data[6:9])
                self.vel_real = np.array(msg.data[9:12])
                rpy = np.array(msg.data[15:18])
            self.rot_real = R.from_euler('xyz', rpy, degrees=False)

    def vel_cb(self, msg):
        """Updates the handle's raw 6D spatial velocity."""
        self.vel_haption = np.array([
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z
        ])

    # =========================
    # FORCE COMPONENT (F_sync only)
    # =========================
    def compute_F_sync(self):
        """Spring tether (position + orientation) keeping the handle synced with the real EE pose."""
        F_sync = np.zeros(6)
        if self.pos_target is None or self.pos_real is None:
            return F_sync

        # Position spring in TRIAGo frame, mapped to Haption frame (negate X, Y).
        error_pos_tiago = self.pos_real - self.pos_target
        F_spring_tiago = self.Kp_sync * error_pos_tiago
        F_spring_haption = np.array([-F_spring_tiago[0], -F_spring_tiago[1], F_spring_tiago[2]])
        F_sync[0:3] = F_spring_haption - (self.Kd_sync * self.vel_haption[0:3])

        # Orientation spring: R_err = R_real * R_target^T, same frame flip on the torque.
        if self.rot_real is not None and self.rot_target is not None:
            err_rot = R.from_matrix(
                self.rot_real.as_matrix() @ self.rot_target.as_matrix().T).as_rotvec()
            Tau_tiago = self.Kp_sync_ang * err_rot
            F_sync[3] = -Tau_tiago[0]
            F_sync[4] = -Tau_tiago[1]
            F_sync[5] = Tau_tiago[2]

        return F_sync

    def compute_F_limit_warning(self):
        """Buzzes continuously while a device joint is within LIMIT_OUTER of a bound (same pattern as the joystick out-of-deadzone cue)."""
        F_vib = np.zeros(6)

        # Closest distance to any of the 12 joint bounds.
        dist_to_min = self.joint_pos - self.joint_min
        dist_to_max = self.joint_max - self.joint_pos
        min_margin = float(np.min(np.concatenate([dist_to_min, dist_to_max])))

        if min_margin <= self.LIMIT_OUTER:
            self.vib_toggle *= -1.0
            amp = self.LIMIT_VIB_AMP
            F_vib[3] = amp * self.vib_toggle
            F_vib[4] = amp * self.vib_toggle
            F_vib[5] = amp * self.vib_toggle

        return F_vib

    # =========================
    # MAIN LOOP
    # =========================
    @staticmethod
    def _tanh_cap(vec, max_norm):
        """Saturates a vector's magnitude smoothly, preserving its direction exactly."""
        n = float(np.linalg.norm(vec))
        if n < 1e-9:
            return np.zeros(3)
        return (max_norm * np.tanh(n / max_norm)) * (vec / n)

    def _compute_homing_wrench(self):
        """Gentle startup spring to the neutral handle pose: tanh-capped and faded in from zero."""
        f = np.zeros(6)
        if self.pos_haption is None or self.rot_haption is None:
            return f

        now = time.time()
        if self._homing_t0 is None:
            self._homing_t0 = now
        # Fade in so the handle is never grabbed at full strength on the very first tick.
        fade = min(1.0, (now - self._homing_t0) / self.HOMING_FADE_S)

        # Home and handle are both in the Haption frame, so no frame mapping is needed.
        err_pos = self.HOME_POS - self.pos_haption
        err_rotvec = (self.HOME_ROT * self.rot_haption.inv()).as_rotvec()
        f[0:3] = fade * self._tanh_cap(self.HOMING_KP_LIN * err_pos, self.MAX_HOMING_FORCE)
        f[3:6] = fade * self._tanh_cap(self.HOMING_KP_ANG * err_rotvec, self.MAX_HOMING_TORQUE)

        # Settle time, not a bare threshold: the handle must hold the pose, not just cross it.
        lin_err = float(np.linalg.norm(err_pos))
        ang_err = float(np.linalg.norm(err_rotvec))
        if lin_err < self.HOMING_TOL_LIN and ang_err < self.HOMING_TOL_ANG:
            self._homing_ticks_in_tol += 1
        else:
            self._homing_ticks_in_tol = 0
        if self._homing_ticks_in_tol >= self.HOMING_SETTLE_TICKS:
            self._finish_homing(True)
        elif now - self._homing_t0 > self.HOMING_TIMEOUT_S:
            # A soft spring may never overcome friction from the far side; releasing teleop
            # matters more than reaching the neutral pose exactly.
            self._finish_homing(False)
        return f

    def _finish_homing(self, at_neutral):
        """Latches homing off and pins the handle<->gripper correspondence at (handle, live gripper)."""
        self._homing_done = True
        if self.rot_real is None:
            self.get_logger().warn(
                "[CLUTCH HOMING] Homing ended with no EE pose yet: the correspondence "
                "falls back to capture on the first clutch press.")
            return
        self._align_ref_haption = self.HOME_ROT if at_neutral else self.rot_haption
        self._align_ref_real = self.rot_real
        self._capture_align_axis()
        if at_neutral:
            self.get_logger().info(
                "[CLUTCH HOMING] Handle parked at neutral: handle<->gripper orientation "
                "correspondence pinned there, identical every run.")
        else:
            self.get_logger().warn(
                "[CLUTCH HOMING] Timed out before reaching neutral: correspondence pinned at the "
                "handle's current pose instead, so it is not reproducible across runs.")

    def _capture_align_axis(self):
        """Handle-local axis matching the gripper's approach axis: only the anchor pair relates the two frames."""
        self._align_axis_local = self._align_ref_haption.inv().apply(
            _FRAME_FLIP * self._align_ref_real.apply(_APPROACH_AXIS))

    def _approach_swing_error(self, target_rot_haption):
        """Shortest rotation aligning the handle's approach axis with the target's, leaving the twist about it free."""
        n_cur = self.rot_haption.apply(self._align_axis_local)
        n_tgt = target_rot_haption.apply(self._align_axis_local)
        axis = np.cross(n_cur, n_tgt)
        s = float(np.linalg.norm(axis))
        c = float(np.dot(n_cur, n_tgt))
        if s < 1e-9:
            if c > 0.0:
                return np.zeros(3)
            # Anti-parallel: the swing axis is degenerate, so fall back to the full error's
            # component perpendicular to the approach axis.
            err = (target_rot_haption * self.rot_haption.inv()).as_rotvec()
            return err - float(np.dot(err, n_cur)) * n_cur
        # arctan2 keeps the error in radians, so K_align and the tanh ceiling keep their meaning.
        return (axis / s) * np.arctan2(s, c)

    def _compute_clutch_align_torque(self):
        """Live torque (Haption frame) aligning the handle with the REAL gripper orientation while clutching."""
        if self.rot_haption is None or self.rot_real is None:
            return np.zeros(3)

        # Fallback capture (homing normally pins this): only DELTAS carry meaning across the two
        # frames, since their zero orientations are unrelated and cannot be mapped directly.
        if self._align_ref_real is None:
            self._align_ref_real = self.rot_real
            self._align_ref_haption = self.rot_haption
            self._capture_align_axis()
            self.get_logger().info(
                "[CLUTCH ALIGN] Handle<->gripper orientation correspondence captured "
                "(current handle pose now means 'matched'; tracks gripper deltas from here).")

        # Frame-map the delta's ROTVEC (angle-preserving under the Z-flip, unlike a matrix product),
        # then divide by CLUTCH_ROT_SCALE to invert how the teleop integrated the handle twist.
        delta_triago = ((self.rot_real * self._align_ref_real.inv()).as_rotvec()
                        / self.CLUTCH_ROT_SCALE)
        target_rot_haption = R.from_rotvec(_FRAME_FLIP * delta_triago) * self._align_ref_haption
        # 5-DOF alignment (task_dim=5 in the QP): only the approach axis is aligned, so roll about
        # it -- unreachable for the wrist and not worth tracking while clutching -- induces no torque.
        err_rotvec = self._approach_swing_error(target_rot_haption)

        # tanh on the MAGNITUDE keeps the axis exact and saturates smoothly instead of hard-clipping.
        self.K_align = float(self.get_parameter('K_align').value)
        err_norm = float(np.linalg.norm(err_rotvec))
        if err_norm < 1e-9:
            return np.zeros(3)
        tau_mag = self.MAX_ALIGN_TORQUE * np.tanh(self.K_align * err_norm / self.MAX_ALIGN_TORQUE)
        tau = tau_mag * (err_rotvec / err_norm)

        # Fade out near a device joint limit so the cue never fights the hardware stop.
        dist_to_min = self.joint_pos - self.joint_min
        dist_to_max = self.joint_max - self.joint_pos
        min_margin = float(np.min(np.concatenate([dist_to_min, dist_to_max])))
        if min_margin < self.CLUTCH_ALIGN_FADE_MARGIN:
            tau = tau * max(0.0, min_margin / self.CLUTCH_ALIGN_FADE_MARGIN)
        return tau

    def control_loop(self):
        """150 Hz: renders F_sync (or the grasp cue), applies clutch freeze + damping, clips, publishes."""
        # Broadcast the homing gate every tick: the teleop node must not integrate the handle
        # motion the homing spring itself is producing.
        gate = Bool()
        gate.data = not self._homing_done
        self.homing_pub.publish(gate)

        # Startup homing phase: only the parking spring is rendered, nothing else.
        if not self._homing_done:
            f_home = self._compute_homing_wrench()
            fn = np.linalg.norm(f_home[0:3])
            if fn > self.MAX_TOTAL_FORCE:
                f_home[0:3] *= self.MAX_TOTAL_FORCE / fn
            tn = np.linalg.norm(f_home[3:6])
            if tn > self.MAX_TOTAL_TORQUE:
                f_home[3:6] *= self.MAX_TOTAL_TORQUE / tn
            f_home[0:3] = np.clip(f_home[0:3], -self.MAX_FORCE, self.MAX_FORCE)
            f_home[3:6] = np.clip(f_home[3:6], -self.MAX_TORQUE, self.MAX_TORQUE)
            home_msg = Wrench()
            home_msg.force.x, home_msg.force.y, home_msg.force.z = (
                float(f_home[0]), float(f_home[1]), float(f_home[2]))
            home_msg.torque.x, home_msg.torque.y, home_msg.torque.z = (
                float(f_home[3]), float(f_home[4]), float(f_home[5]))
            self.force_pub.publish(home_msg)
            return

        self.Kp_sync = float(self.get_parameter('Kp_sync').value)
        self.Kp_sync_ang = float(self.get_parameter('Kp_sync_ang').value)
        f_sync = self.compute_F_sync()

        if self.grasp_active:
            # Autonomous grasp: no force impressed, only the vibration cue added below.
            self._grasp_start_pos = None
            f_total_normal = np.zeros(6)
        else:
            self._grasp_start_pos = None
            f_total_normal = f_sync.copy()

        # Authority cap: proportional rescale bounds the assistive wrench to MAX_TOTAL_*.
        fn = np.linalg.norm(f_total_normal[0:3])
        if fn > self.MAX_TOTAL_FORCE:
            f_total_normal[0:3] *= self.MAX_TOTAL_FORCE / fn
        tn = np.linalg.norm(f_total_normal[3:6])
        if tn > self.MAX_TOTAL_TORQUE:
            f_total_normal[3:6] *= self.MAX_TOTAL_TORQUE / tn

        # Clutch press: either free repositioning (align mode) or the wrench frozen at 50%.
        if self.is_clutching:
            if self.ENABLE_CLUTCH_ALIGN:
                # Clutch = free repositioning: no linear force at all, only a live alignment
                # torque recomputed every tick against the current gripper orientation.
                self.was_clutching_last_frame = True
                f_total = np.zeros(6)
                f_total[3:6] = self._compute_clutch_align_torque()
            else:
                # On the press edge: freeze the wrench at 50% (cognitive grounding).
                if not self.was_clutching_last_frame:
                    self.f_clutch_frozen = f_total_normal / 2.0
                    self.was_clutching_last_frame = True
                f_total = self.f_clutch_frozen.copy()
        else:
            f_total = f_total_normal
            self.was_clutching_last_frame = False

        # Global viscous damping; skipped during grasp so only the cue is felt.
        if not self.grasp_active:
            if self.ENABLE_GLOBAL_DAMPING_LIN:
                f_total[0:3] -= self.Kd_global_lin * self.vel_haption[0:3]
            self._vel_ang_filt = (self.VEL_FILT_ALPHA_ANG * self._vel_ang_filt
                                  + (1.0 - self.VEL_FILT_ALPHA_ANG) * self.vel_haption[3:6])
            if self.ENABLE_GLOBAL_DAMPING_ANG:
                ang_vel = self._vel_ang_filt
                ang_vel_norm = np.linalg.norm(ang_vel)
                if ang_vel_norm > self.VEL_DAMP_CLAMP_ANG:
                    ang_vel = ang_vel * (self.VEL_DAMP_CLAMP_ANG / ang_vel_norm)
                f_total[3:6] -= self.Kd_global_ang * ang_vel

        # Cues injected last so they ride on top of any frozen wrench and toggle every frame.
        f_vib = self.compute_F_limit_warning()
        if self.grasp_active:
            self.grasp_vib_toggle *= -1.0
            gb = self.GRASP_VIB_AMP * self.grasp_vib_toggle
            f_total[3] += gb
            f_total[4] += gb
            f_total[5] += gb
        else:
            f_total[3:6] += f_vib[3:6]

        # Device safety clip.
        f_total[0:3] = np.clip(f_total[0:3], -self.MAX_FORCE, self.MAX_FORCE)
        f_total[3:6] = np.clip(f_total[3:6], -self.MAX_TORQUE, self.MAX_TORQUE)

        # Debug log: breaks the published torque down by contributor (sync/damping/vibration).
        now_log = time.time()
        if now_log - self._last_err_log_time > 0.5:
            self._last_err_log_time = now_log
            vel_ang_deg = np.degrees(np.linalg.norm(self.vel_haption[3:6]))
            vel_ang_filt_deg = np.degrees(np.linalg.norm(self._vel_ang_filt))
            sync_tau = float(np.linalg.norm(f_sync[3:6]))
            damp_tau_raw = float(np.linalg.norm(self.Kd_global_ang * self.vel_haption[3:6]))
            damp_tau_filt = float(np.linalg.norm(self.Kd_global_ang * self._vel_ang_filt))
            vib_tau = float(np.linalg.norm(f_vib[3:6]))
            pub_tau = float(np.linalg.norm(f_total[3:6]))
            pub_f = float(np.linalg.norm(f_total[0:3]))
            self.get_logger().info(
                f"[DBG] vel_ang={vel_ang_deg:.1f} deg/s (filt={vel_ang_filt_deg:.1f})  "
                f"sync_tau={sync_tau:.3f}  damp_tau={damp_tau_filt:.3f} (raw={damp_tau_raw:.3f})  "
                f"vib_tau={vib_tau:.3f}  -> PUB_TAU={pub_tau:.3f} Nm  PUB_F={pub_f:.3f} N  "
                f"clutch={self.is_clutching}")

        msg = Wrench()
        msg.force.x, msg.force.y, msg.force.z = float(f_total[0]), float(f_total[1]), float(f_total[2])
        msg.torque.x, msg.torque.y, msg.torque.z = float(f_total[3]), float(f_total[4]), float(f_total[5])
        self.force_pub.publish(msg)

        # Buffer for plotting.
        t = time.time() - self.start_time
        with self.plot_lock:
            self.t_data.append(t)
            for i in range(3):
                self.sync_F[i].append(f_sync[i])
                self.sync_T[i].append(f_sync[i + 3])
                self.tot_F[i].append(f_total[i])
                self.tot_T[i].append(f_total[i + 3])
            now = time.time()
            if self._own_last_time is not None:
                dt_own = now - self._own_last_time
                if dt_own > 1e-6:
                    self._own_freq_lpf = 0.9 * self._own_freq_lpf + 0.1 * (1.0 / dt_own)
            self._own_last_time = now
            self.freq_data.append(self._own_freq_lpf)


def main(args=None):
    """Spins ROS on a daemon thread and drives Matplotlib on the main thread."""
    rclpy.init(args=args)
    node = HapticForceManagerNoGuidance()

    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True,
        name='rclpy-spin',
    )
    spin_thread.start()

    try:
        if node.plot_enabled:
            while rclpy.ok():
                node.update_plot()
                plt.pause(0.1)
        else:
            spin_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
