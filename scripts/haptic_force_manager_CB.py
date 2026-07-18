#!/usr/bin/env python3
"""CLUTCH guided blending (F=0, B=1): F_sync tethers the handle to the blended reference; no guidance forces."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench, Twist
from std_msgs.msg import Bool, Float64MultiArray, Float64, String
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from collections import deque
from geometry_msgs.msg import Pose  # virtuose/pose is Pose, not PoseStamped
import matplotlib.pyplot as plt
import matplotlib

# Cross-package condition selector: single source of truth for the 2x2x2 study cell.
import triago_control.qp_controller.config as cfg

# TkAgg keeps Matplotlib off the ROS spin thread.
matplotlib.use('TkAgg')

class HapticForceManagerCB(Node):
    # Blending-only force manager: shared autonomy blends the reference, the handle feels only F_sync.
    def __init__(self):
        super().__init__('haptic_force_manager_cb')

        # Hard-error at startup unless config.py selects the CLUTCH guided-blending cell.
        cfg.validate_condition('haptic_force_manager_CB',
                               control_mode=cfg.CLUTCH, feedback=False, blending=True)

        self.pos_target = None
        self.rot_target = None
        self.pos_real = None
        self.rot_real = None
        self.vel_real = np.zeros(3)     # active-arm EE linear velocity (from ee_real)
        self.vel_haption = np.zeros(6)  # handle 6D spatial velocity (Haption frame)

        # CBF telemetry (evaluated for plots only; F_cbf is excluded from the rendered wrench).
        self.grad_cbf_right = np.zeros(6)
        self.lambda_cbf = 0.0
        self.lambda_cbf_f = 0.0
        self.CBF_LAMBDA_ALPHA = 0.05  # LPF coefficient on lambda
        self.CBF_GAIN_BOOST = 1.2

        self.f_cbf_filtered = np.zeros(6)
        self.alpha_cbf = 0.15          # LPF: keep 85% of the old value each tick

        self.MAX_CBF_FORCE = 15.0      # N
        self.MAX_CBF_TORQUE = 1.0      # Nm

        # Shared-autonomy inference state (goal names, beliefs, per-goal user policies).
        self.goal_names = []
        self.goal_probs = []
        self.user_policies = []

        # F_guide gains (kept for reference; guidance is never applied in this cell).
        self.D_guide_lin = 33.6   # Ns/m
        self.D_guide_ang = 0.54   # Nms/rad
        self.MAX_GUIDE_FORCE  = 4.2   # N
        self.MAX_GUIDE_TORQUE = 0.10  # Nm

        # Guidance gates (kept for reference; guidance is never applied in this cell).
        self.GUIDE_PROX_FAR  = 0.60   # m
        self.GUIDE_PROX_NEAR = 0.10   # m

        # Debug: guidance-only mode is meaningless in CB (guidance deleted).
        self.DEBUG_ONLY_GUIDE = False

        self.GUIDE_CONF_LO   = 0.30
        self.GUIDE_CONF_HI   = 0.90
        self.alpha_guide     = 0.15
        self.f_guide_filtered = np.zeros(6)

        # Position virtual fixture gains (kept for reference; F_fixture is not rendered).
        self.fix_goal_pos = None
        self.fix_goal_rot = None
        self.fix_confidence = 0.0
        self.K_fix_force  = 38.016    # N/m
        self.K_fix_torque = 0.2376    # Nm/rad
        self.MAX_FIX_FORCE  = 5.7024  # N
        self.MAX_FIX_TORQUE = 0.396   # Nm
        self.FIX_CONF_LO = 0.55
        self.FIX_CONF_HI = 0.85
        self.alpha_fix = 0.15
        self.f_fix_filtered = np.zeros(6)
        # Near-goal orientation-assist shaping (fixture-related, not rendered).
        self.FIX_TORQUE_NEAR = 0.05        # m
        self.FIX_TORQUE_FAR  = 0.12        # m
        self.FIX_TORQUE_NEAR_BOOST = 0.20
        self.K_FIX_TORQUE_DAMP = 0.06      # Nms/rad

        # Clutch press freezes the wrench at 50% (cognitive grounding).
        self.is_clutching = False
        self.was_clutching_last_frame = False
        self.f_clutch_frozen = np.zeros(6)
        self.K_align = 10.0  # Nm/rad clutch orientation-alignment stiffness
        # Disabled: the alignment error mixes robot-base and device frames, making the torque non-restorative.
        self.ENABLE_CLUTCH_ALIGN = False
        self.rot_haption = None


        # Legacy virtual-fixture stiffness values (not used).
        self.K_guide_force = 90.0   # N/m
        self.K_guide_torque = 0.3   # Nm/rad

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

        # Sync spring gains, unified across all clutch cells (Kd=0: global damper supplies viscosity).
        self.Kp_sync = 30.0       # N/m
        self.Kd_sync = 0.0
        self.Kp_sync_ang = 0.9    # Nm/rad

        # Adaptive sync-share parameters (kept for reference; attenuation is not applied).
        self.SYNC_SHARE_AT_FULL = 0.5
        self.SYNC_FULL_POS_ERR  = 0.30     # m
        self.SYNC_FULL_ANG_ERR  = np.pi / 2
        self.SYNC_SHARE_CAP     = 0.85

        # During autonomous grasp no force is impressed; only a vibration cue is rendered.
        self.grasp_active = False
        self._grasp_start_pos = None
        self.GRASP_SYNC_BOOST = 6.0
        self.GRASP_FOLLOW_KP = 30.0    # N/m
        self.GRASP_FOLLOW_KD = 160.0   # Ns/m
        self.GRASP_VIB_AMP = 0.07      # Nm constant square-wave buzz during the whole grasp
        self.grasp_vib_toggle = 1.0
        self.K_cbf_force = 2.0
        self.K_cbf_torque = 0.1
        self.MAX_FORCE = 10.0
        self.MAX_TORQUE = 1.0

        # Authority cap: proportional rescale so assistance can never overpower the operator.
        self.MAX_TOTAL_FORCE  = 10.0  # N
        self.MAX_TOTAL_TORQUE = 1.0   # Nm

        # Plot buffers (10 s window at 150 Hz), guarded by a lock shared with the UI thread.
        self.plot_lock = threading.Lock()
        self.plot_window_sec = 10.0
        self.buffer_size = int(150 * self.plot_window_sec)
        self.t_data = deque(maxlen=self.buffer_size)
        self.start_time = time.time()

        # Final published total wrench (= F_sync + damping + cues in this cell).
        self.tot_F = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.tot_T = [deque(maxlen=self.buffer_size) for _ in range(3)]
        # Blend telemetry buffers (alpha + user/policy share).
        self.alpha_data = deque(maxlen=self.buffer_size)
        self.user_pct_data = deque(maxlen=self.buffer_size)
        self.policy_pct_data = deque(maxlen=self.buffer_size)
        # Own loop-rate tracker.
        self._own_freq_data = deque(maxlen=self.buffer_size)
        self._own_last_time = None
        self._own_freq_lpf = 0.0
        self._last_blend_alpha = 0.0
        self._last_blend_user_pct = 0.0
        self._last_blend_policy_pct = 0.0
        self.create_subscription(Float64MultiArray, '/arm_right/cartesian_reference', self.target_cb, 10)
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self.real_cb, 10)
        self.create_subscription(Twist, 'virtuose/velocity', self.vel_cb, 10)
        self.create_subscription(Float64MultiArray, '/collision_constraints', self.cbf_gradient_cb, 10)
        self.create_subscription(Float64MultiArray, '/qp_debug/lambda_cbf', self.lambda_cb, 10)
        self.create_subscription(Float64MultiArray, 'virtuose/articular_position', self.joint_cb, 10)
        self.create_subscription(Bool, 'virtuose/button_right', self.button_cb, 10)
        self.create_subscription(Pose, 'virtuose/pose', self.haption_pose_cb, 10)
        # Shared-autonomy inference state.
        self.create_subscription(String, '/shared_autonomy/goal_names', self.goal_names_cb, 10)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/goal_probabilities', self.goal_probs_cb, 10)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/user_policy', self.user_policy_cb, 10)
        # Blend telemetry: [alpha, v_user(6), v_policy(6), v_blend(6)] = 19 floats.
        self.create_subscription(Float64MultiArray, '/shared_autonomy/blend_debug', self.blend_debug_cb, 10)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/active_goal_pose', self.goal_pose_cb, 10)
        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_cb, 10)
        # Arm switching is decided solely by shared autonomy; this node follows.
        self.active_arm = 'right'
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)
        self.create_subscription(Float64MultiArray, '/arm_left/cartesian_reference', self.target_cb_left, 10)

        self.force_pub = self.create_publisher(Wrench, 'virtuose/force_cmd', 10)

        self.dt = 1.0 / 150.0
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.setup_plot()
        self.get_logger().info("Haptic Force Manager (CB: guided blending, NO feedback) started.")

    # =========================
    # PLOT SETUP & UPDATE
    # =========================
    def setup_plot(self):
        """Initializes the live plot: total published wrench, blend alpha + share, loop rate."""
        plt.ion()
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']

        self.fig, self.axs = plt.subplots(3, 2, figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('CB: Clutch Guided-Blending (F_sync only)')

        ax = self.axs[0, 0]
        ax.set_title("Published FORCE (N)", fontsize=10, fontweight='bold')
        ax.set_ylabel("N"); ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(self.MAX_FORCE, color='k', linestyle='--', linewidth=1.0, alpha=0.6, label='±max')
        ax.axhline(-self.MAX_FORCE, color='k', linestyle='--', linewidth=1.0, alpha=0.6)
        self.lines_tot_F = [ax.plot([], [], color=colors[i], label=f"F{labels[i]}")[0] for i in range(3)]
        ax.legend(loc='upper left', fontsize=8, ncol=4)
        ax = self.axs[0, 1]
        ax.set_title("Published TORQUE (Nm)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Nm"); ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(self.MAX_TORQUE, color='k', linestyle='--', linewidth=1.0, alpha=0.6, label='±max')
        ax.axhline(-self.MAX_TORQUE, color='k', linestyle='--', linewidth=1.0, alpha=0.6)
        self.lines_tot_T = [ax.plot([], [], color=colors[i], label=f"T{labels[i]}")[0] for i in range(3)]
        ax.legend(loc='upper left', fontsize=8, ncol=4)

        ax = self.axs[1, 0]
        ax.set_title("Blending authority α (0=user, 1=policy)", fontsize=10, fontweight='bold')
        ax.set_ylabel("α"); ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color='#888', linestyle=':', linewidth=0.8)
        ax.grid(True, linestyle='--', alpha=0.6)
        self.line_alpha, = ax.plot([], [], color='#ff7f0e', linewidth=1.5, label='α')
        ax.legend(loc='upper left', fontsize=8)
        ax = self.axs[1, 1]
        ax.set_title("Blend share: (1-α)·v_user  vs  α·v_policy", fontsize=10, fontweight='bold')
        ax.set_ylabel("%"); ax.set_ylim(-5, 105)
        ax.grid(True, linestyle='--', alpha=0.6)
        self.line_user_pct, = ax.plot([], [], color='#1f77b4', linewidth=1.4, label='user %')
        self.line_policy_pct, = ax.plot([], [], color='#ff7f0e', linewidth=1.4, label='policy %')
        ax.legend(loc='upper left', fontsize=8, ncol=2)

        ax = self.axs[2, 0]
        ax.set_title("Force Manager Frequency (Hz)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Hz"); ax.set_xlabel("Time (s)"); ax.set_ylim(0, 180)
        ax.grid(True, linestyle='--', alpha=0.6)
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
            tF = [list(self.tot_F[i]) for i in range(3)]
            tT = [list(self.tot_T[i]) for i in range(3)]
            alpha_list = list(self.alpha_data)
            upct_list = list(self.user_pct_data)
            ppct_list = list(self.policy_pct_data)
            freq_list = list(self._own_freq_data)

        win = (t_list[-1] - self.plot_window_sec, t_list[-1])
        for i in range(3):
            self.lines_tot_F[i].set_data(t_list, tF[i])
            self.lines_tot_T[i].set_data(t_list, tT[i])
        for c in range(2):
            self.axs[0, c].set_xlim(*win)
            self.axs[0, c].relim()
            self.axs[0, c].autoscale_view(scalex=False, scaley=True)

        n = min(len(t_list), len(alpha_list))
        self.line_alpha.set_data(t_list[:n], alpha_list[:n])
        self.axs[1, 0].set_xlim(*win)
        n2 = min(len(t_list), len(upct_list), len(ppct_list))
        self.line_user_pct.set_data(t_list[:n2], upct_list[:n2])
        self.line_policy_pct.set_data(t_list[:n2], ppct_list[:n2])
        self.axs[1, 1].set_xlim(*win)

        nf = min(len(t_list), len(freq_list))
        self.line_freq.set_data(t_list[:nf], freq_list[:nf])
        self.axs[2, 0].set_xlim(*win)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # =========================
    # CALLBACKS
    # =========================
    def blend_debug_cb(self, msg):
        """Processes blend telemetry [alpha, v_user(6), v_policy(6), v_blend(6)] into share percentages."""
        if len(msg.data) < 13:
            return
        a = float(msg.data[0])
        vu = np.linalg.norm(msg.data[1:7])
        vp = np.linalg.norm(msg.data[7:13])
        u_weight = (1.0 - a) * vu
        p_weight = a * vp
        total = u_weight + p_weight
        self._last_blend_alpha = a
        self._last_blend_user_pct = 100.0 * u_weight / total if total > 1e-9 else 0.0
        self._last_blend_policy_pct = 100.0 * p_weight / total if total > 1e-9 else 0.0

    def haption_pose_cb(self, msg):
        """Stores the handle orientation (geometry_msgs/Pose) for the clutch alignment torque."""
        q = msg.orientation
        self.rot_haption = R.from_quat([q.x, q.y, q.z, q.w])

    def button_cb(self, msg):
        """Updates the clutching state from the Virtuose button."""
        self.is_clutching = msg.data

    def goal_names_cb(self, msg):
        """Updates the list of active goal names from the inference engine."""
        self.goal_names = msg.data.split(',')

    def goal_probs_cb(self, msg):
        """Updates the array of goal probabilities."""
        self.goal_probs = list(msg.data)

    def user_policy_cb(self, msg):
        """Updates the flattened per-goal user-frame policy twists."""
        self.user_policies = list(msg.data)

    def goal_pose_cb(self, msg):
        """Updates the active goal pose + belief b_max; layout [x,y,z,rpy(3),confidence]."""
        if len(msg.data) >= 7:
            self.fix_goal_pos = np.array(msg.data[0:3])
            self.fix_goal_rot = R.from_euler('xyz', np.array(msg.data[3:6]), degrees=False)
            self.fix_confidence = float(msg.data[6])

    def grasp_active_cb(self, msg):
        """Tracks whether shared autonomy is autonomously driving a grasp."""
        self.grasp_active = bool(msg.data)

    def active_arm_cb(self, msg):
        """Switches which arm's EE data is used for force computation."""
        if msg.data in ('right', 'left') and msg.data != self.active_arm:
            self.active_arm = msg.data
            self.get_logger().info(f"[FORCE MGR] Active arm switched to {msg.data.upper()}")

    def target_cb(self, msg):
        """Updates the target Cartesian pose (right arm reference)."""
        if self.active_arm != 'right':
            return
        if len(msg.data) >= 6:
            self.pos_target = np.array(msg.data[0:3])
            rpy = np.array(msg.data[3:6])
            self.rot_target = R.from_euler('xyz', rpy, degrees=False)

    def target_cb_left(self, msg):
        """Updates the target Cartesian pose (left arm reference)."""
        if self.active_arm != 'left':
            return
        if len(msg.data) >= 6:
            self.pos_target = np.array(msg.data[0:3])
            rpy = np.array(msg.data[3:6])
            self.rot_target = R.from_euler('xyz', rpy, degrees=False)

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
        """Updates the handle's raw 6D spatial velocity (unfiltered: an LPF here adds destabilizing lag)."""
        self.vel_haption = np.array([
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z
        ])

    def cbf_gradient_cb(self, msg):
        """Updates the right arm's Cartesian CBF gradient; layout [b_col_r, b_col_l, J_R(6), J_L(6)]."""
        if len(msg.data) >= 14:
            self.grad_cbf_right = np.array(msg.data[2:8])

    def lambda_cb(self, msg):
        """Updates the right arm's CBF shadow price; layout [lambda_R, lambda_L]."""
        if len(msg.data) < 1:
            return
        lambda_r = float(msg.data[0])
        self.lambda_cbf = lambda_r
        self.lambda_cbf_f = ((1.0 - self.CBF_LAMBDA_ALPHA) * self.lambda_cbf_f
                             + self.CBF_LAMBDA_ALPHA * max(0.0, lambda_r))

    def joint_cb(self, msg):
        """Updates the 6-DoF Haption joint positions from the encoders."""
        if len(msg.data) >= 6:
            self.joint_pos = np.array(msg.data[0:6])

    # =========================
    # FORCE COMPONENTS
    # =========================

    def compute_F_sync(self):
        """Spring tether (position + orientation) keeping the handle synced with the real EE pose."""
        F_sync = np.zeros(6)
        if self.pos_target is None or self.pos_real is None:
            return F_sync

        # Position spring in TRIAGo frame, mapped to Haption frame (negate X, Y).
        error_pos_tiago = self.pos_real - self.pos_target
        F_spring_tiago = self.Kp_sync * error_pos_tiago

        F_spring_haption = np.zeros(3)
        F_spring_haption[0] = -F_spring_tiago[0]
        F_spring_haption[1] = -F_spring_tiago[1]
        F_spring_haption[2] =  F_spring_tiago[2]

        F_damped_haption = F_spring_haption - (self.Kd_sync * self.vel_haption[0:3])
        F_sync[0:3] = F_damped_haption

        # Orientation spring toward the REAL EE orientation reins the handle back if the reference runs away.
        if self.rot_real is not None and self.rot_target is not None:
            err_rot = R.from_matrix(
                self.rot_real.as_matrix() @ self.rot_target.as_matrix().T).as_rotvec()
            Tau_tiago = self.Kp_sync_ang * err_rot
            F_sync[3] = -Tau_tiago[0]
            F_sync[4] = -Tau_tiago[1]
            F_sync[5] =  Tau_tiago[2]

        return F_sync

    def compute_F_cbf(self):
        """Repulsive CBF wrench: gradient x shadow price, tanh-saturated and LPF'd (telemetry only)."""
        # Free space: decay the residual smoothly to zero instead of snapping.
        if self.lambda_cbf <= 0.0:
            self.f_cbf_filtered = (1.0 - self.alpha_cbf) * self.f_cbf_filtered
            return self.f_cbf_filtered

        F_cbf_triago = self.grad_cbf_right * self.lambda_cbf
        F_cbf_triago[0:3] *= self.K_cbf_force
        F_cbf_triago[3:6] *= self.K_cbf_torque

        # Tanh bends the unbounded CBF spike into a bounded, comfortable curve.
        F_cbf_triago[0:3] = self.MAX_CBF_FORCE * np.tanh(F_cbf_triago[0:3] / self.MAX_CBF_FORCE)
        F_cbf_triago[3:6] = self.MAX_CBF_TORQUE * np.tanh(F_cbf_triago[3:6] / self.MAX_CBF_TORQUE)

        # Map TRIAGo -> Haption frame (negate X, Y for both force and torque).
        F_cbf_raw_haption = np.zeros(6)
        F_cbf_raw_haption[0] = -F_cbf_triago[0]
        F_cbf_raw_haption[1] = -F_cbf_triago[1]
        F_cbf_raw_haption[2] =  F_cbf_triago[2]
        F_cbf_raw_haption[3] = -F_cbf_triago[3]
        F_cbf_raw_haption[4] = -F_cbf_triago[4]
        F_cbf_raw_haption[5] =  F_cbf_triago[5]

        self.f_cbf_filtered = (self.alpha_cbf * F_cbf_raw_haption) + ((1.0 - self.alpha_cbf) * self.f_cbf_filtered)

        return self.f_cbf_filtered

    def _smoothstep(self, p, lo=0.70, hi=1.0):
        """C1-continuous ramp from 0 at p=lo to 1 at p=hi."""
        if p <= lo:
            return 0.0
        x = min((p - lo) / (hi - lo), 1.0)
        return 3.0 * x**2 - 2.0 * x**3

    @staticmethod
    def _sync_share_factor(sync_mag, push_mag, share):
        """Attenuation factor (<=1) so the sync spring reaches at least `share` of the total magnitude."""
        if share <= 1e-3 or push_mag < 1e-6 or sync_mag < 1e-6:
            return 1.0
        max_push = sync_mag * (1.0 - share) / share
        if push_mag > max_push:
            return float(np.clip(max_push / push_mag, 0.0, 1.0))
        return 1.0

    # goal_names: comma-joined goal list; goal_probabilities: aligned simplex; user_policy: n_goals x 6 twists.

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
        """150 Hz: renders F_sync only (blended-reference tether), applies clutch/grasp handling, publishes."""
        f_sync = self.compute_F_sync()
        f_cbf = self.compute_F_cbf()
        f_vib = self.compute_F_limit_warning()

        # Grasp execution takes precedence over every other mode (including DEBUG).
        if self.grasp_active:
            # Autonomous grasp: no force impressed, only the vibration cue added below.
            self._grasp_start_pos = None
            f_total_normal = np.zeros(6)
            f_cbf_s = np.zeros(6)
            f_guide_s = np.zeros(6)
            f_fix_s = np.zeros(6)
        elif self.DEBUG_ONLY_GUIDE:
            # Guidance-only debug path is meaningless in CB (guidance deleted).
            self._grasp_start_pos = None
            f_total_normal = np.zeros(6)
            f_cbf_s = np.zeros(6)
            f_guide_s = np.zeros(6)
            f_fix_s = np.zeros(6)
        else:
            self._grasp_start_pos = None
            # CB renders only F_sync: guidance and fixture deleted, F_cbf telemetry-only.
            # The blended reference from shared autonomy is what the tether targets.
            f_cbf_s = np.zeros(6)
            f_guide_s = np.zeros(6)
            f_fix_s = np.zeros(6)

            # f_vib is injected AFTER clutch/grasp branching so the cue is never frozen by the clutch.
            f_total_normal = f_sync + f_cbf_s + f_guide_s + f_fix_s

            # Authority cap: proportional rescale only when the magnitude exceeds the bound.
            f_norm = np.linalg.norm(f_total_normal[0:3])
            if f_norm > self.MAX_TOTAL_FORCE:
                f_total_normal[0:3] *= self.MAX_TOTAL_FORCE / f_norm
            t_norm = np.linalg.norm(f_total_normal[3:6])
            if t_norm > self.MAX_TOTAL_TORQUE:
                f_total_normal[3:6] *= self.MAX_TOTAL_TORQUE / t_norm

        # ========================================================
        # CLUTCH HANDLING
        # ========================================================
        if self.DEBUG_ONLY_GUIDE:
            f_total = f_total_normal.copy()
        elif self.is_clutching:
            # On the press edge: freeze the wrench at 50% (cognitive grounding).
            if not self.was_clutching_last_frame:
                self.f_clutch_frozen = f_total_normal / 2.0
                self.was_clutching_last_frame = True

            f_total = self.f_clutch_frozen.copy()

            # Alignment torque toward the frozen target orientation (disabled: frame-mixing).
            if self.ENABLE_CLUTCH_ALIGN and self.rot_haption is not None and self.rot_target is not None:

                # R_error = R_target * R_haption^T.
                error_rot_matrix = self.rot_target.as_matrix() @ self.rot_haption.as_matrix().T
                error_rot_vec = R.from_matrix(error_rot_matrix).as_rotvec()

                tau_align_base = self.K_align * error_rot_vec

                # Map to the Haption frame (180-deg Z-flip).
                tau_align_haption = np.zeros(3)
                tau_align_haption[0] = -tau_align_base[0]
                tau_align_haption[1] = -tau_align_base[1]
                tau_align_haption[2] =  tau_align_base[2]

                # Fade to zero within 0.35 rad of a device joint limit.
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

        # Global viscous damping, constant and unified across all clutch cells.
        if not self.DEBUG_ONLY_GUIDE and not self.grasp_active:
            Kd_global_lin = 0.7
            Kd_global_ang = 0.1
            f_total[0:3] -= Kd_global_lin * self.vel_haption[0:3]
            f_total[3:6] -= Kd_global_ang * self.vel_haption[3:6]

        # Cues injected last so they ride on top of any frozen wrench and toggle every frame.
        if self.grasp_active:
            self.grasp_vib_toggle *= -1.0
            gb = self.GRASP_VIB_AMP * self.grasp_vib_toggle
            f_total[3] += gb
            f_total[4] += gb
            f_total[5] += gb
        else:
            f_total[3:6] += f_vib[3:6]

        # ========================================================
        # CLIPPING & PUBLISHING
        # ========================================================
        f_total[0:3] = np.clip(f_total[0:3], -self.MAX_FORCE, self.MAX_FORCE)
        f_total[3:6] = np.clip(f_total[3:6], -self.MAX_TORQUE, self.MAX_TORQUE)
        msg = Wrench()
        msg.force.x, msg.force.y, msg.force.z = float(f_total[0]), float(f_total[1]), float(f_total[2])
        msg.torque.x, msg.torque.y, msg.torque.z = float(f_total[3]), float(f_total[4]), float(f_total[5])
        self.force_pub.publish(msg)

        # Buffer for plotting (total wrench + blend telemetry).
        t = time.time() - self.start_time
        with self.plot_lock:
            self.t_data.append(t)
            for i in range(3):
                self.tot_F[i].append(f_total[i])
                self.tot_T[i].append(f_total[i + 3])
            self.alpha_data.append(self._last_blend_alpha)
            self.user_pct_data.append(self._last_blend_user_pct)
            self.policy_pct_data.append(self._last_blend_policy_pct)
            now = time.time()
            if self._own_last_time is not None:
                dt_own = now - self._own_last_time
                if dt_own > 1e-6:
                    self._own_freq_lpf = 0.9 * self._own_freq_lpf + 0.1 * (1.0 / dt_own)
            self._own_last_time = now
            self._own_freq_data.append(self._own_freq_lpf)

def main(args=None):
    """Spins ROS on a daemon thread and drives Matplotlib on the main thread."""
    rclpy.init(args=args)
    node = HapticForceManagerCB()

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
