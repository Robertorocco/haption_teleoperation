#!/usr/bin/env python3
"""CLUTCH full guidance (F=1, B=1): same wrench as CF, with F_sync tethered to the blended reference."""

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

# TRIAGo <-> Haption base-frame map: 180-deg rotation about Z, its own inverse, valid for
# linear and axial vectors alike (so it may be applied directly to a rotation vector).
_FRAME_FLIP = np.array([-1.0, -1.0, 1.0])

# TCP approach axis: the gripper's local +X, the convention used across the whole platform.
_APPROACH_AXIS = np.array([1.0, 0.0, 0.0])

class HapticForceManagerFull(Node):
    # Full-guidance force manager: F_sync + gated F_guide, while shared autonomy blends the reference.
    def __init__(self):
        super().__init__('haptic_force_manager_full')

        # Hard-error at startup unless config.py selects the CLUTCH full-guidance cell.
        cfg.validate_condition('haptic_force_manager_CFB',
                               control_mode=cfg.CLUTCH, feedback=True, blending=True)

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

        self.MAX_CBF_FORCE = 7.5       # N
        self.MAX_CBF_TORQUE = 0.5      # Nm

        # Shared-autonomy inference state (goal names, beliefs, per-goal user policies).
        self.goal_names = []
        self.goal_probs = []
        self.user_policies = []

        # F_guide: feed-forward velocity field -- policy speed shaped into force, tanh-saturated.
        self.D_guide_lin = 21.84 * 1.2   # Ns/m  translation magnitude-shaping gain
        self.D_guide_ang = 0.351 * 4.2   # Nms/rad rotation magnitude-shaping gain
        self.MAX_GUIDE_FORCE  = 2.73 * 1.2   # N   guidance force saturation
        self.MAX_GUIDE_TORQUE = 0.065 * 4.2  # Nm  guidance torque saturation

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
        self.K_fix_force  = 19.008    # N/m
        self.K_fix_torque = 0.1188    # Nm/rad
        self.MAX_FIX_FORCE  = 2.8512  # N
        self.MAX_FIX_TORQUE = 0.198   # Nm
        self.FIX_CONF_LO = 0.55
        self.FIX_CONF_HI = 0.85
        self.alpha_fix = 0.15
        self.f_fix_filtered = np.zeros(6)
        # Near-goal orientation-assist shaping (fixture-related, not rendered).
        self.FIX_TORQUE_NEAR = 0.05        # m
        self.FIX_TORQUE_FAR  = 0.12        # m
        self.FIX_TORQUE_NEAR_BOOST = 0.20
        self.K_FIX_TORQUE_DAMP = 0.03      # Nms/rad

        # Clutch press freezes the wrench at 50% (cognitive grounding).
        self.is_clutching = False
        self.was_clutching_last_frame = False
        self.f_clutch_frozen = np.zeros(6)
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
        self.pos_haption = None
        self.rot_haption = None


        # Virtual-fixture stiffness, unused by the velocity-field guidance.
        self.K_guide_force = 45.0   # N/m
        self.K_guide_torque = 0.15  # Nm/rad

        # Haption joint positions and calibrated limits (for the joint-limit cue).
        self.joint_pos = np.zeros(6)
        self.joint_min = np.array([-0.785282, -1.5709, 0.792704, -2.39339, -1.02312, -2.22872])
        self.joint_max = np.array([0.784393, -0.00157491, 2.49551, 2.35378, 0.879614, 2.21327])

        self.LIMIT_OUTER = 0.10       # rad: margin where the joint-limit cue buzzes
        self.vib_toggle = 1.0         # sign flip every frame -> 75 Hz square wave

        # Joint-limit cue: same continuous while-condition-holds pattern as the
        # joystick out-of-deadzone buzz -- no timer, no re-arm state.
        self.LIMIT_VIB_AMP = 0.009  # Nm

        # Sync spring gains, unified across all clutch cells (Kd=0: global damper supplies viscosity).
        # Live-tunable so the spring pair can be swept without a rebuild.
        self.Kp_sync = float(self.declare_parameter('Kp_sync', 15.0).value)      # N/m
        self.Kd_sync = 0.0
        self.Kp_sync_ang = float(self.declare_parameter('Kp_sync_ang', 0.1).value)  # Nm/rad
        # Damping is rendered by virtuose_server_node instead (same-tick passivity can't hold
        # across a process boundary); tune it there via ros2 param set damping_lin/damping_ang.
        self.ENABLE_GLOBAL_DAMPING = False

        # Adaptive sync-share parameters (kept for reference; attenuation is not applied).
        self.SYNC_SHARE_AT_FULL = 0.5
        self.SYNC_FULL_POS_ERR  = 0.30     # m
        self.SYNC_FULL_ANG_ERR  = np.pi / 2
        self.SYNC_SHARE_CAP     = 0.85

        # During autonomous grasp no force is impressed; only a vibration cue is rendered.
        self.grasp_active = False
        self._grasp_start_pos = None
        self.GRASP_SYNC_BOOST = 6.0
        self.GRASP_FOLLOW_KP = 15.0    # N/m
        self.GRASP_FOLLOW_KD = 80.0    # Ns/m
        self.GRASP_VIB_AMP = 0.009  # Nm constant square-wave buzz during the whole grasp
        self.grasp_vib_toggle = 1.0
        self.K_cbf_force = 1.0
        self.K_cbf_torque = 0.05
        self.MAX_FORCE = 5.0
        self.MAX_TORQUE = 0.5

        # Authority cap: proportional rescale so assistance can never overpower the operator.
        self.MAX_TOTAL_FORCE  = 5.0   # N
        self.MAX_TOTAL_TORQUE = 0.5   # Nm

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

        # Blend telemetry buffers (alpha + user/policy share), shared design with every B=1 cell.
        self.alpha_data = deque(maxlen=self.buffer_size)
        self.user_pct_data = deque(maxlen=self.buffer_size)
        self.policy_pct_data = deque(maxlen=self.buffer_size)
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
        self.create_subscription(Float64MultiArray, '/shared_autonomy/active_goal_pose', self.goal_pose_cb, 10)
        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_cb, 10)
        # Arm switching is decided solely by shared autonomy; this node follows.
        self.active_arm = 'right'
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)
        self.create_subscription(Float64MultiArray, '/arm_left/cartesian_reference', self.target_cb_left, 10)
        # Blend telemetry: [alpha, v_user(6), v_policy(6), v_blend(6)] = 19 floats.
        self.create_subscription(Float64MultiArray, '/shared_autonomy/blend_debug', self.blend_debug_cb, 10)

        self.force_pub = self.create_publisher(Wrench, 'virtuose/force_cmd', 10)
        # Gates the teleop node's integration so the homing motion never drives the robot.
        self.homing_pub = self.create_publisher(Bool, 'device/homing_active', 10)

        self.dt = 1.0 / 150.0
        self.timer = self.create_timer(self.dt, self.control_loop)

        # Live Matplotlib windows: on by default, disable with -p plot:=false.
        self.plot_enabled = bool(self.declare_parameter('plot', True).value)
        if self.plot_enabled:
            self.setup_plot()
        self.get_logger().info("Haptic Force Manager (full: feedback + blending) started.")

    # =========================
    # PLOT SETUP & UPDATE
    # =========================
    def setup_plot(self):
        """Initializes the live Matplotlib windows (superposition + total wrench + blend telemetry)."""
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

        # Blend-telemetry window: authority alpha + user/policy share.
        self.fig_bl, self.axs_bl = plt.subplots(2, 1, figsize=(10, 5))
        self.fig_bl.canvas.manager.set_window_title('Blending Telemetry')
        ax = self.axs_bl[0]
        ax.set_title("Blending authority α (0=user, 1=policy)", fontsize=10, fontweight='bold')
        ax.set_ylabel("α"); ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color='#888', linestyle=':', linewidth=0.8)
        ax.grid(True, linestyle='--', alpha=0.6)
        self.line_alpha, = ax.plot([], [], color='#ff7f0e', linewidth=1.5, label='α')
        ax.legend(loc='upper left', fontsize=8)
        ax = self.axs_bl[1]
        ax.set_title("Blend share: (1-α)·v_user  vs  α·v_policy", fontsize=10, fontweight='bold')
        ax.set_ylabel("%"); ax.set_xlabel("Time (s)"); ax.set_ylim(-5, 105)
        ax.grid(True, linestyle='--', alpha=0.6)
        self.line_user_pct, = ax.plot([], [], color='#1f77b4', linewidth=1.4, label='user %')
        self.line_policy_pct, = ax.plot([], [], color='#ff7f0e', linewidth=1.4, label='policy %')
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        self.fig_bl.tight_layout()

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
            alpha_list = list(self.alpha_data)
            upct_list = list(self.user_pct_data)
            ppct_list = list(self.policy_pct_data)

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

        # Blend-telemetry window.
        na = min(len(t_list), len(alpha_list))
        self.line_alpha.set_data(t_list[:na], alpha_list[:na])
        self.axs_bl[0].set_xlim(*win)
        ns = min(len(t_list), len(upct_list), len(ppct_list))
        self.line_user_pct.set_data(t_list[:ns], upct_list[:ns])
        self.line_policy_pct.set_data(t_list[:ns], ppct_list[:ns])
        self.axs_bl[1].set_xlim(*win)
        self.fig_bl.canvas.draw_idle()

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

        # F = MAX*tanh(D*v_field/MAX), gain-scaled after the tanh; no handle-velocity feedback (a
        # virtual damper at this gain would exceed the 150 Hz passivity limit). Vanishes at the goal.
        F_guide_raw = np.zeros(6)
        F_guide_raw[0:3] = self.MAX_GUIDE_FORCE * np.tanh(self.D_guide_lin * v_field[0:3] / self.MAX_GUIDE_FORCE)
        F_guide_raw[3:6] = self.MAX_GUIDE_TORQUE * np.tanh(self.D_guide_ang * v_field[3:6] / self.MAX_GUIDE_TORQUE)
        F_guide_raw = gain * F_guide_raw

        self.f_guide_filtered = (self.alpha_guide * F_guide_raw
                                 + (1.0 - self.alpha_guide) * self.f_guide_filtered)
        return self.f_guide_filtered.copy()

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
        """150 Hz: superposes F_sync + F_guide, applies clutch/grasp handling, caps, clips, publishes."""
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

        # Global viscous damping, constant and unified across all clutch cells.
        if self.ENABLE_GLOBAL_DAMPING and not self.DEBUG_ONLY_GUIDE and not self.grasp_active:
            Kd_global_lin = 0.35
            Kd_global_ang = 0.05
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
            # Blend telemetry.
            self.alpha_data.append(self._last_blend_alpha)
            self.user_pct_data.append(self._last_blend_user_pct)
            self.policy_pct_data.append(self._last_blend_policy_pct)

def main(args=None):
    """Spins ROS on a daemon thread and drives Matplotlib on the main thread."""
    rclpy.init(args=args)
    node = HapticForceManagerFull()

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
