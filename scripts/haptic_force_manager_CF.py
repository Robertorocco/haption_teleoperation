#!/usr/bin/env python3
"""CLUTCH guided feedback (F=1, B=0): F_sync tether + feed-forward F_guide velocity-field guidance."""

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

class HapticForceManager(Node):
    # Virtual-Fixture force manager: renders F_sync + gated F_guide on the handle.
    def __init__(self):
        super().__init__('haptic_force_manager')

        # Hard-error at startup unless config.py selects the CLUTCH guided-feedback cell.
        cfg.validate_condition('haptic_force_manager_CF',
                               control_mode=cfg.CLUTCH, feedback=True, blending=False)

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

        # F_guide: feed-forward velocity field -- policy speed shaped into force, tanh-saturated.
        self.D_guide_lin = 43.68   # Ns/m  magnitude-shaping gain (translation)
        self.D_guide_ang = 0.702   # Nms/rad magnitude-shaping gain (rotation)
        self.MAX_GUIDE_FORCE  = 5.46   # N   guidance force saturation
        self.MAX_GUIDE_TORQUE = 0.13  # Nm  guidance torque saturation

        # Proximity gate: guidance silenced far from the goal, where the goal manifold still swings.
        self.GUIDE_PROX_FAR  = 0.60   # m: beyond this the device is free (gate = 0)
        self.GUIDE_PROX_NEAR = 0.10   # m: at/below this full guidance (gate = 1)

        # Debug: True renders only F_guide (no sync, no damping, no clutch handling).
        self.DEBUG_ONLY_GUIDE = False

        # Confidence gate on the active-goal belief b_max, unified across all guidance cells.
        self.GUIDE_CONF_LO   = 0.30   # below: transparent
        self.GUIDE_CONF_HI   = 0.90   # at/above: full guidance gain
        # LPF on the guidance wrench guarantees C0 continuity across noisy belief samples.
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


        # Legacy virtual-fixture stiffness values (not used by the velocity-field guidance).
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
        self.Kp_sync = 30.0      # N/m
        self.Kd_sync = 0.0
        self.Kp_sync_ang = 0.9   # Nm/rad

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

        self.f_data = {
            'Sync':  {'F': [deque(maxlen=self.buffer_size) for _ in range(3)], 'T': [deque(maxlen=self.buffer_size) for _ in range(3)]},
            'CBF':   {'F': [deque(maxlen=self.buffer_size) for _ in range(3)], 'T': [deque(maxlen=self.buffer_size) for _ in range(3)]},
            'Guide': {'F': [deque(maxlen=self.buffer_size) for _ in range(3)], 'T': [deque(maxlen=self.buffer_size) for _ in range(3)]},
            'Limit': {'F': [deque(maxlen=self.buffer_size) for _ in range(3)], 'T': [deque(maxlen=self.buffer_size) for _ in range(3)]}
        }

        # f_total window: final published wrench + per-source % breakdown.
        self.ftot_data = {'F': [deque(maxlen=self.buffer_size) for _ in range(3)],
                          'T': [deque(maxlen=self.buffer_size) for _ in range(3)]}
        self.pct_force  = {'Sync': deque(maxlen=self.buffer_size),
                           'CBF':  deque(maxlen=self.buffer_size),
                           'Guide': deque(maxlen=self.buffer_size)}
        self.pct_torque = {'Sync': deque(maxlen=self.buffer_size),
                           'CBF':  deque(maxlen=self.buffer_size),
                           'Guide': deque(maxlen=self.buffer_size)}

        # Inference-rate tracker (from goal_probs arrival rate).
        self._sa_freq_data = deque(maxlen=self.buffer_size)
        self._sa_last_time = None
        self._sa_freq_lpf = 0.0

        # Own loop-rate tracker.
        self._own_freq_data = deque(maxlen=self.buffer_size)
        self._own_last_time = None
        self._own_freq_lpf = 0.0
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
        self.get_logger().info("Haptic Force Manager (tutorial) started.")

    # =========================
    # PLOT SETUP & UPDATE
    # =========================
    def setup_plot(self):
        """Initializes the live Matplotlib windows (superposition + total wrench)."""
        plt.ion()
        self.fig, self.axs = plt.subplots(4, 2, figsize=(12, 9))
        self.fig.canvas.manager.set_window_title('Haptic Force Superposition')

        self.lines = {}
        categories = ['Sync', 'CBF', 'Guide', 'Limit']
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']

        for row, cat in enumerate(categories):
            self.lines[cat] = {'F': [], 'T': []}

            ax_f = self.axs[row, 0]
            ax_f.set_title(f"{cat} Wrench - FORCE (N)", fontsize=10, pad=3)
            ax_f.set_ylabel("Force (N)")
            ax_f.grid(True, linestyle='--', alpha=0.6)
            for i in range(3):
                line, = ax_f.plot([], [], color=colors[i], label=f"F{labels[i]}")
                self.lines[cat]['F'].append(line)
            ax_f.legend(loc='upper left', fontsize=8)

            ax_t = self.axs[row, 1]
            ax_t.set_title(f"{cat} Wrench - TORQUE (Nm)", fontsize=10, pad=3)
            ax_t.set_ylabel("Torque (Nm)")
            ax_t.grid(True, linestyle='--', alpha=0.6)
            for i in range(3):
                line, = ax_t.plot([], [], color=colors[i], label=f"T{labels[i]}")
                self.lines[cat]['T'].append(line)
            ax_t.legend(loc='upper left', fontsize=8)

        for col in range(2):
            self.axs[3, col].set_xlabel("Time (s)")

        self.fig.tight_layout()

        # f_total window: published wrench + contribution shares + loop rate.
        self.fig_tot, self.axs_tot = plt.subplots(5, 1, figsize=(9, 12))
        self.fig_tot.canvas.manager.set_window_title('Total Wrench (published to device)')
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']
        src_colors = {'Sync': '#1f77b4', 'CBF': '#d62728', 'Guide': '#2ca02c'}

        ax = self.axs_tot[0]
        ax.set_title("f_total — FORCE components (N)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Force (N)")
        ax.grid(True, linestyle='--', alpha=0.6)
        self.lines_ftot_F = [ax.plot([], [], color=colors[i], label=f"F{labels[i]}")[0] for i in range(3)]
        ax.axhline( self.MAX_TOTAL_FORCE, color='k', linestyle='--', linewidth=1.0, alpha=0.7, label='±max')
        ax.axhline(-self.MAX_TOTAL_FORCE, color='k', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.legend(loc='upper left', fontsize=8, ncol=4)

        ax = self.axs_tot[1]
        ax.set_title("f_total — TORQUE components (Nm)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Torque (Nm)")
        ax.grid(True, linestyle='--', alpha=0.6)
        self.lines_ftot_T = [ax.plot([], [], color=colors[i], label=f"T{labels[i]}")[0] for i in range(3)]
        ax.axhline( self.MAX_TOTAL_TORQUE, color='k', linestyle='--', linewidth=1.0, alpha=0.7, label='±max')
        ax.axhline(-self.MAX_TOTAL_TORQUE, color='k', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.legend(loc='upper left', fontsize=8, ncol=4)

        ax = self.axs_tot[2]
        ax.set_title("Force contribution share (%)", fontsize=10, fontweight='bold')
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)
        self.lines_pctF = {k: ax.plot([], [], color=src_colors[k], label=k)[0]
                           for k in ['Sync', 'CBF', 'Guide']}
        ax.legend(loc='upper left', fontsize=8, ncol=3)

        ax = self.axs_tot[3]
        ax.set_title("Torque contribution share (%)", fontsize=10, fontweight='bold')
        ax.set_ylabel("%")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)
        self.lines_pctT = {k: ax.plot([], [], color=src_colors[k], label=k)[0]
                           for k in ['Sync', 'CBF', 'Guide']}
        ax.legend(loc='upper left', fontsize=8, ncol=3)

        ax = self.axs_tot[4]
        ax.set_title("Haptic Force Manager Frequency (Hz)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Hz")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 180)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(150, color='g', linestyle='--', linewidth=1.0, alpha=0.7, label='target 150Hz')
        self.line_sa_freq, = ax.plot([], [], color='#9467bd', linewidth=1.5, label='HFM freq')
        ax.legend(loc='upper left', fontsize=8)

        self.fig_tot.tight_layout()
        plt.show(block=False)


    def update_plot(self):
        """Snapshots buffers under the lock and refreshes the Matplotlib UI."""
        with self.plot_lock:
            if len(self.t_data) == 0:
                return
            t_list = list(self.t_data)
            f_lists = {
                cat: {
                    'F': [list(self.f_data[cat]['F'][i]) for i in range(3)],
                    'T': [list(self.f_data[cat]['T'][i]) for i in range(3)]
                } for cat in ['Sync', 'CBF', 'Guide', 'Limit']
            }
            ftot_F = [list(self.ftot_data['F'][i]) for i in range(3)]
            ftot_T = [list(self.ftot_data['T'][i]) for i in range(3)]
            pctF = {k: list(self.pct_force[k]) for k in ['Sync', 'CBF', 'Guide']}
            pctT = {k: list(self.pct_torque[k]) for k in ['Sync', 'CBF', 'Guide']}

        # Matplotlib updates happen outside the lock to avoid stalling the ROS loop.
        current_t = t_list[-1]
        win = (current_t - self.plot_window_sec, current_t)

        for row, cat in enumerate(['Sync', 'CBF', 'Guide', 'Limit']):
            for i in range(3):
                self.lines[cat]['F'][i].set_data(t_list, f_lists[cat]['F'][i])
            self.axs[row, 0].set_xlim(*win)
            self.axs[row, 0].relim()
            self.axs[row, 0].autoscale_view(scalex=False, scaley=True)

            for i in range(3):
                self.lines[cat]['T'][i].set_data(t_list, f_lists[cat]['T'][i])
            self.axs[row, 1].set_xlim(*win)
            self.axs[row, 1].relim()
            self.axs[row, 1].autoscale_view(scalex=False, scaley=True)

        for i in range(3):
            self.lines_ftot_F[i].set_data(t_list, ftot_F[i])
            self.lines_ftot_T[i].set_data(t_list, ftot_T[i])
        for k in ['Sync', 'CBF', 'Guide']:
            self.lines_pctF[k].set_data(t_list, pctF[k])
            self.lines_pctT[k].set_data(t_list, pctT[k])

        self.axs_tot[0].set_xlim(*win)
        self.axs_tot[0].set_ylim(-self.MAX_TOTAL_FORCE * 1.4, self.MAX_TOTAL_FORCE * 1.4)
        self.axs_tot[1].set_xlim(*win)
        self.axs_tot[1].set_ylim(-self.MAX_TOTAL_TORQUE * 1.4, self.MAX_TOTAL_TORQUE * 1.4)
        self.axs_tot[2].set_xlim(*win)
        self.axs_tot[3].set_xlim(*win)

        with self.plot_lock:
            own_freq_list = list(self._own_freq_data)
        if own_freq_list:
            # Trim to the common length: the two buffers may differ by one sample under the lock.
            n = min(len(t_list), len(own_freq_list))
            self.line_sa_freq.set_data(t_list[:n], own_freq_list[:n])
        self.axs_tot[4].set_xlim(*win)

        self.fig_tot.canvas.draw_idle()

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

    def compute_F_guide(self):
        """Feed-forward velocity-field guidance: belief-blended policy twist shaped into a gated force."""
        n_goals = len(self.goal_names) if self.goal_names else 0
        n_policies = len(self.user_policies)

        if (n_goals == 0
                or len(self.goal_probs) != n_goals
                or n_policies != n_goals * 6):
            self.f_guide_filtered = (1.0 - self.alpha_guide) * self.f_guide_filtered
            return self.f_guide_filtered.copy()

        # pi_blend = sum_k P(k) * pi_k: belief-weighted policy twist (robot frame).
        probs = np.array(self.goal_probs)
        policies = np.array(self.user_policies).reshape(n_goals, 6)
        pi_blend = probs @ policies

        # Confidence gate on b_max (max posterior), zero during autonomous grasp execution.
        alpha = self._smoothstep(self.fix_confidence, lo=self.GUIDE_CONF_LO, hi=self.GUIDE_CONF_HI)

        # Proximity gate: reference-to-goal distance, silencing guidance while the goal still swings.
        if self.fix_goal_pos is not None and self.pos_target is not None:
            d_goal = float(np.linalg.norm(self.fix_goal_pos - self.pos_target))
            prox = np.clip(
                (self.GUIDE_PROX_FAR - d_goal)
                / max(self.GUIDE_PROX_FAR - self.GUIDE_PROX_NEAR, 1e-6), 0.0, 1.0)
            prox_gate = 3.0 * prox ** 2 - 2.0 * prox ** 3   # smoothstep
        else:
            prox_gate = 0.0   # no goal/reference info -> no guidance

        gain = alpha * prox_gate

        # Map the policy twist into the Haption frame (180-deg Z-flip, matching the clutch teleop).
        v_field = np.array([
            -pi_blend[0], -pi_blend[1],  pi_blend[2],
            -pi_blend[3], -pi_blend[4],  pi_blend[5],
        ])

        # Feed-forward: F = MAX*tanh(D*v_field/MAX), gain-scaled after the tanh; no handle-velocity
        # feedback (a virtual damper at this gain exceeds the 150 Hz passivity limit). Vanishes at
        # the goal since pi_blend -> 0 there.
        F_guide_raw = np.zeros(6)
        F_guide_raw[0:3] = self.MAX_GUIDE_FORCE * np.tanh(self.D_guide_lin * v_field[0:3] / self.MAX_GUIDE_FORCE)
        F_guide_raw[3:6] = self.MAX_GUIDE_TORQUE * np.tanh(self.D_guide_ang * v_field[3:6] / self.MAX_GUIDE_TORQUE)
        F_guide_raw = gain * F_guide_raw

        self.f_guide_filtered = (self.alpha_guide * F_guide_raw
                                 + (1.0 - self.alpha_guide) * self.f_guide_filtered)
        return self.f_guide_filtered.copy()

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
        """150 Hz: superposes F_sync + F_guide, applies clutch/grasp handling, caps, clips, publishes."""
        f_sync = self.compute_F_sync()
        f_cbf = self.compute_F_cbf()
        f_guide = self.compute_F_guide()
        f_vib = self.compute_F_limit_warning()

        # Grasp execution takes precedence over every other mode (including DEBUG).
        if self.grasp_active:
            # Autonomous grasp: no force impressed, only the vibration cue added below.
            self._grasp_start_pos = None
            f_total_normal = np.zeros(6)
            f_cbf_s = np.zeros(6)
            f_guide_s = np.zeros(6)
        elif self.DEBUG_ONLY_GUIDE:
            # Guidance-only debug: F_guide alone.
            self._grasp_start_pos = None
            f_total_normal = f_guide
            f_cbf_s = np.zeros(6)
            f_guide_s = f_guide.copy()
        else:
            self._grasp_start_pos = None
            # Fairness: no adaptive sync-share, no F_fixture, F_cbf excluded (telemetry only).
            f_cbf_s = np.zeros(6)
            f_guide_s = f_guide

            # f_vib is injected AFTER clutch/grasp branching so the cue is never frozen by the clutch.
            f_total_normal = f_sync + f_cbf_s + f_guide_s

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
            # Debug mode: skip clutching and global damping, raw F_guide only.
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

        # Buffer for plotting.
        t = time.time() - self.start_time
        guide_comb = f_guide_s
        components = {'Sync': f_sync, 'CBF': f_cbf_s, 'Guide': guide_comb, 'Limit': f_vib}

        # Per-source contribution share (% of summed component magnitudes).
        nF = {'Sync': np.linalg.norm(f_sync[0:3]),
              'CBF':  np.linalg.norm(f_cbf[0:3]),
              'Guide': np.linalg.norm(guide_comb[0:3])}
        nT = {'Sync': np.linalg.norm(f_sync[3:6]),
              'CBF':  np.linalg.norm(f_cbf[3:6]),
              'Guide': np.linalg.norm(guide_comb[3:6])}
        sF = sum(nF.values())
        sT = sum(nT.values())

        with self.plot_lock:
            self.t_data.append(t)
            for cat, force_vec in components.items():
                for i in range(3):
                    self.f_data[cat]['F'][i].append(force_vec[i])
                    self.f_data[cat]['T'][i].append(force_vec[i + 3])
            for i in range(3):
                self.ftot_data['F'][i].append(f_total[i])
                self.ftot_data['T'][i].append(f_total[i + 3])
            for k in ['Sync', 'CBF', 'Guide']:
                self.pct_force[k].append(100.0 * nF[k] / sF if sF > 1e-9 else 0.0)
                self.pct_torque[k].append(100.0 * nT[k] / sT if sT > 1e-9 else 0.0)
            self._sa_freq_data.append(self._sa_freq_lpf)
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
    node = HapticForceManager()

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
