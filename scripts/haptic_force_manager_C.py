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

# Cross-package condition selector: single source of truth for the 2x2x2 study cell.
import triago_control.qp_controller.config as cfg

# TkAgg keeps Matplotlib off the ROS spin thread.
matplotlib.use('TkAgg')


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
        self.Kp_sync = 30.0        # N/m
        self.Kd_sync = 0.0         # Ns/m
        self.Kp_sync_ang = 0.9     # Nm/rad

        # During autonomous grasp no force is impressed; only a vibration cue is rendered.
        self.grasp_active = False
        self._grasp_start_pos = None
        self.GRASP_FOLLOW_KP = 30.0     # N/m
        self.GRASP_FOLLOW_KD = 160.0    # Ns/m
        self.GRASP_VIB_AMP = 0.07       # Nm constant square-wave buzz during the whole grasp
        self.grasp_vib_toggle = 1.0

        # Clutch press freezes the wrench at 50% (cognitive grounding).
        self.is_clutching = False
        self.was_clutching_last_frame = False
        self.f_clutch_frozen = np.zeros(6)
        self.rot_haption = None
        self.K_align = 10.0  # Nm/rad clutch orientation-alignment stiffness
        # Disabled: the alignment error mixes robot-base and device frames, making the torque non-restorative.
        self.ENABLE_CLUTCH_ALIGN = False

        # Global viscous damping (impedance-device stability), unified across clutch cells.
        self.Kd_global_lin = 0.7   # Ns/m
        self.Kd_global_ang = 0.1   # Nms/rad

        # Device safety clip and unified authority cap (currently equal).
        self.MAX_FORCE = 10.0      # N
        self.MAX_TORQUE = 1.0      # Nm
        self.MAX_TOTAL_FORCE = 10.0
        self.MAX_TOTAL_TORQUE = 1.0

        # Arm the force is computed for (follows shared autonomy's active arm).
        self.active_arm = 'right'

        # Haption joint positions and calibrated limits (for the joint-limit cue).
        self.joint_pos = np.zeros(6)
        self.joint_min = np.array([-0.804283, -1.65038, 0.728283, -3.02431, -1.28196, -2.05398])
        self.joint_max = np.array([0.781944, -0.0654231, 2.49752, 2.82038, 1.04722, 2.09453])

        self.LIMIT_OUTER = 0.25       # rad: margin where the cue can fire
        self.LIMIT_INNER = 0.15       # rad: margin of maximum vibration
        self.AMP_MIN = 0.05           # Nm
        self.AMP_MAX = 0.07           # Nm
        self.vib_toggle = 1.0         # sign flip every frame -> 75 Hz square wave

        # Joint-limit "clutch advice": one-shot burst, re-armed only by a full clutch cycle.
        self.LIMIT_VIB_DURATION = 1.0   # s
        self.LIMIT_VIB_AMP = 0.07       # Nm
        self.limit_vib_armed = True
        self.limit_vib_active = False
        self.limit_vib_start_time = 0.0
        self._vib_clutch_prev = False

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

        self.dt = 1.0 / 150.0
        self.timer = self.create_timer(self.dt, self.control_loop)

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
        """Stores the handle orientation (geometry_msgs/Pose) for the clutch alignment torque."""
        q = msg.orientation
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
        """One-shot 1 s torque burst when a device joint nears a limit; re-armed by a full clutch cycle."""
        F_vib = np.zeros(6)
        now = time.time()

        # Re-arm on a completed clutch cycle (press -> release).
        if self._vib_clutch_prev and not self.is_clutching:
            self.limit_vib_armed = True
        self._vib_clutch_prev = self.is_clutching

        # Closest distance to any of the 12 joint bounds.
        dist_to_min = self.joint_pos - self.joint_min
        dist_to_max = self.joint_max - self.joint_pos
        min_margin = float(np.min(np.concatenate([dist_to_min, dist_to_max])))

        # Fire only if armed and not already playing.
        if (min_margin <= self.LIMIT_OUTER
                and self.limit_vib_armed
                and not self.limit_vib_active):
            self.limit_vib_active = True
            self.limit_vib_start_time = now
            self.limit_vib_armed = False

        # The burst always plays its full duration, even if the operator clutches midway.
        if self.limit_vib_active:
            if (now - self.limit_vib_start_time) <= self.LIMIT_VIB_DURATION:
                self.vib_toggle *= -1.0
                amp = self.LIMIT_VIB_AMP
                F_vib[3] = amp * self.vib_toggle
                F_vib[4] = amp * self.vib_toggle
                F_vib[5] = amp * self.vib_toggle
            else:
                self.limit_vib_active = False

        return F_vib

    # =========================
    # MAIN LOOP
    # =========================
    def control_loop(self):
        """150 Hz: renders F_sync (or the grasp cue), applies clutch freeze + damping, clips, publishes."""
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

        # Clutch press: freeze the wrench at 50% (cognitive grounding).
        if self.is_clutching:
            if not self.was_clutching_last_frame:
                self.f_clutch_frozen = f_total_normal / 2.0
                self.was_clutching_last_frame = True
            f_total = self.f_clutch_frozen.copy()

            # Alignment torque toward the target orientation, faded near device joint limits (disabled).
            if self.ENABLE_CLUTCH_ALIGN and self.rot_haption is not None and self.rot_target is not None:
                error_rot_matrix = self.rot_target.as_matrix() @ self.rot_haption.as_matrix().T
                error_rot_vec = R.from_matrix(error_rot_matrix).as_rotvec()
                tau_align_base = self.K_align * error_rot_vec
                tau_align_haption = np.array([
                    -tau_align_base[0], -tau_align_base[1], tau_align_base[2]])
                dist_to_min = self.joint_pos - self.joint_min
                dist_to_max = self.joint_max - self.joint_pos
                min_margin = np.min(np.concatenate([dist_to_min, dist_to_max]))
                fade_margin = 0.35
                if min_margin < fade_margin:
                    scale = max(0.0, min_margin / fade_margin)
                    tau_align_haption *= scale
                f_total[3:6] += tau_align_haption
        else:
            f_total = f_total_normal
            self.was_clutching_last_frame = False

        # Global viscous damping; skipped during grasp so only the cue is felt.
        if not self.grasp_active:
            f_total[0:3] -= self.Kd_global_lin * self.vel_haption[0:3]
            f_total[3:6] -= self.Kd_global_ang * self.vel_haption[3:6]

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
        while rclpy.ok():
            node.update_plot()
            plt.pause(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
