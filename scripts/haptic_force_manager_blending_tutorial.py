#!/usr/bin/env python3
"""Haptic Force Manager — BLENDING TUTORIAL.

Architecture (finalized 2026-07-03, see triago_control/qp_controller/config.py
cfg.BLENDING docstring for the full topic-routing rationale):

  Haption -> teleop_triago_clutch.py -> /arm_right/user_cartesian_reference
                                                  |
                                                  v
                          main_shared_autonomy.py (SOLE owner of blending):
                            - belief inference (unchanged)
                            - alpha = compute_alpha(b_max)   [cfg.ALPHA_MAX/GAMMA]
                            - v_blend = (1-alpha)*v_user + alpha*pi_policy
                            - PERSISTENT integration of v_blend every tick
                            - publishes /shared_autonomy/blend_debug (telemetry)
                                                  |
                                                  v
                          /arm_right/cartesian_reference  (SOLE writer now)
                                                  |
                                                  v
                                        main_qp_controller.py (QP CLF-CBF)

  This node (haptic_force_manager_blending_tutorial.py) does NOT compute any
  blending or integration itself anymore -- that would risk a second,
  independently-drifting copy of the same math. It only:
    1. Renders F_sync: pulls the Haption handle toward the REAL robot EE, so
       the operator feels the full displacement caused by autonomy authority
       (no F_guide / F_fixture / F_cbf at the haptic level).
    2. Plots the AUTHORITY SHARE using the exact numbers main_shared_autonomy.py
       used to command the robot, read verbatim off /shared_autonomy/blend_debug
       (single source of truth -- never recomputed here).

Force output:
  f_total = F_sync (boosted Kp/Kp_ang) + F_limit_warning + global_damping

Plot:
  Window 1: F_sync wrench (3 force + 3 torque)
  Window 2: f_total (published wrench), node frequency
  Window 3 (NEW): Authority Share -- ||v_user|| vs ||v_policy|| vs ||v_blend||,
                  alpha vs time, and EE/user position divergence
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench, Twist
from std_msgs.msg import Bool, Float64MultiArray, Float64, String
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from collections import deque
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import matplotlib

# Single source of truth for the shared-autonomy blending flag (also read by
# main_shared_autonomy.py and teleop_triago_clutch.py).
import triago_control.qp_controller.config as cfg

matplotlib.use('TkAgg')



class HapticForceManagerBlending(Node):
    def __init__(self):
        """Initializes the Haptic Force Manager (Blending Tutorial)."""
        super().__init__('haptic_force_manager_blending')

        # --- State Variables ---
        self.pos_target = None       # User's raw cartesian reference position
        self.rot_target = None       # User's raw cartesian reference orientation
        self.pos_real = None         # Real EE position (from QP controller)
        self.rot_real = None         # Real EE orientation
        self.vel_real = np.zeros(3)  # Real EE linear velocity
        self.vel_haption = np.zeros(6)  # Haption handle velocity (raw)

        # --- Authority-share telemetry (read verbatim from main_shared_autonomy.py
        # via /shared_autonomy/blend_debug -- NEVER recomputed here). Layout:
        #   [alpha, v_user(6), v_policy(6), v_blend(6)]
        self.alpha = 0.0
        self.v_user = np.zeros(6)
        self.v_policy = np.zeros(6)
        self.v_blend = np.zeros(6)

        # --- Clutching ---
        self.is_clutching = False
        self.was_clutching_last_frame = False
        self.f_clutch_frozen = np.zeros(6)
        self.rot_haption = None
        self.K_align = 10.0          # Nm/rad rotational stiffness for clutch alignment

        # --- Articular Limit Variables ---
        self.joint_pos = np.zeros(6)
        self.joint_min = np.array([-0.804283, -1.65038, 0.728283, -3.02431, -1.28196, -2.05398])
        self.joint_max = np.array([0.781944, -0.0654231, 2.49752, 2.82038, 1.04722, 2.09453])


        # --- Sync Force Parameters ---
        # F_sync is the ONLY force the operator ever feels (no more F_guide /
        # F_fixture / F_cbf at the haptic level), so it must be strong enough on
        # its own to convey EVERY divergence between the user's raw reference and
        # the robot's real EE -- including the divergence CAUSED by the blended
        # reference pulling the robot toward the assistive trajectory. Boosted
        # 3.5x from the original single-force-of-several tuning (10.0 -> 35.0)
        # so a few-cm reference/real gap is clearly felt, while staying well
        # under the MAX_FORCE=10N safety clip for realistic tracking errors.
        self.Kp_sync = 35.0          # N/m   [was 10.0]
        self.Kd_sync = 0.0           # kept at 0: global damping (below) already covers this
        self.Kp_sync_ang = 1.0       # Nm/rad orientation sync spring [was 0.3]

        # --- Global Force Limits ---
        self.MAX_FORCE = 10.0
        self.MAX_TORQUE = 1.0

        # --- Grasp-execution coupling ---
        self.grasp_active = False
        self._grasp_start_pos = None
        self.GRASP_SYNC_BOOST = 6.0
        self.GRASP_FOLLOW_KP = 30.0
        self.GRASP_FOLLOW_KD = 160.0


        # --- Data Buffers & Synchronization ---
        self.plot_lock = threading.Lock()
        self.plot_window_sec = 10.0
        self.buffer_size = int(150 * self.plot_window_sec)
        self.t_data = deque(maxlen=self.buffer_size)
        self.start_time = time.time()

        # Per-component wrench history (only Sync in this architecture)
        self.f_data = {
            'Sync': {'F': [deque(maxlen=self.buffer_size) for _ in range(3)],
                     'T': [deque(maxlen=self.buffer_size) for _ in range(3)]},
        }

        # Total published wrench
        self.ftot_data = {'F': [deque(maxlen=self.buffer_size) for _ in range(3)],
                          'T': [deque(maxlen=self.buffer_size) for _ in range(3)]}

        # Authority-share history (from /shared_autonomy/blend_debug)
        self.alpha_data = deque(maxlen=self.buffer_size)
        self.v_user_norm_data = deque(maxlen=self.buffer_size)
        self.v_policy_norm_data = deque(maxlen=self.buffer_size)
        self.v_blend_norm_data = deque(maxlen=self.buffer_size)
        self.pos_divergence_data = deque(maxlen=self.buffer_size)  # ||pos_real - pos_target||

        # Own node frequency tracker
        self._own_freq_data = deque(maxlen=self.buffer_size)
        self._own_last_time = None
        self._own_freq_lpf = 0.0


        # --- Subscribers ---
        # Topic routing depends on cfg.BLENDING (single source of truth, shared
        # with main_shared_autonomy.py and teleop_triago_clutch.py):
        #   BLENDING=False: /arm_*/cartesian_reference carries the pure user pose
        #     (teleop_triago_clutch.py publishes it directly).
        #   BLENDING=True: /arm_*/cartesian_reference now carries the BLENDED
        #     pose (main_shared_autonomy.py is the sole publisher); the pure user
        #     pose lives on /arm_*/user_cartesian_reference instead. F_sync must
        #     read the PURE USER pose here (not the blended one) so the operator
        #     feels the divergence caused by the autonomy authority, rather than
        #     syncing to a target that already includes their own contribution.
        _user_ref_topic_right = ('/arm_right/user_cartesian_reference' if cfg.BLENDING
                                 else '/arm_right/cartesian_reference')
        _user_ref_topic_left = ('/arm_left/user_cartesian_reference' if cfg.BLENDING
                                else '/arm_left/cartesian_reference')
        self.create_subscription(Float64MultiArray, _user_ref_topic_right, self.target_cb, 10)
        self.create_subscription(Float64MultiArray, _user_ref_topic_left, self.target_cb_left, 10)
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self.real_cb, 10)
        self.create_subscription(Twist, 'virtuose/velocity', self.vel_cb, 10)
        self.create_subscription(Float64MultiArray, 'virtuose/articular_position', self.joint_cb, 10)
        self.create_subscription(Bool, 'virtuose/button_right', self.button_cb, 10)
        self.create_subscription(PoseStamped, 'virtuose/pose', self.haption_pose_cb, 10)
        # Authority-share telemetry (single source of truth; see class docstring)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/blend_debug', self.blend_debug_cb, 10)
        # Grasp execution flag
        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_cb, 10)
        # Arm switch
        self.active_arm = 'right'
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)

        # --- Publishers ---
        self.force_pub = self.create_publisher(Wrench, 'virtuose/force_cmd', 10)

        # --- Timers ---
        self.dt = 1.0 / 150.0
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.setup_plot()
        self.get_logger().info(
            f"Haptic Force Manager (BLENDING tutorial) started. "
            f"cfg.BLENDING={cfg.BLENDING} -> reading user reference from "
            f"'{_user_ref_topic_right}' (right) / '{_user_ref_topic_left}' (left).")


    # =========================
    # PLOT SETUP & UPDATE
    # =========================
    def setup_plot(self):
        """Initializes the live Matplotlib windows: F_sync, f_total, Authority Share."""
        plt.ion()

        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']

        # --- Window 1: F_sync wrench components ---
        self.fig1, self.axs1 = plt.subplots(2, 1, figsize=(10, 5))
        self.fig1.canvas.manager.set_window_title('Haptic Force: F_sync Only')

        ax = self.axs1[0]
        ax.set_title("F_sync — FORCE (N)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Force (N)")
        ax.grid(True, linestyle='--', alpha=0.6)
        self.lines_sync_F = []
        for i in range(3):
            line, = ax.plot([], [], color=colors[i], label=f"F{labels[i]}")
            self.lines_sync_F.append(line)
        ax.legend(loc='upper left', fontsize=8)

        ax = self.axs1[1]
        ax.set_title("F_sync — TORQUE (Nm)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Torque (Nm)")
        ax.set_xlabel("Time (s)")
        ax.grid(True, linestyle='--', alpha=0.6)
        self.lines_sync_T = []
        for i in range(3):
            line, = ax.plot([], [], color=colors[i], label=f"T{labels[i]}")
            self.lines_sync_T.append(line)
        ax.legend(loc='upper left', fontsize=8)
        self.fig1.tight_layout()


        # --- Window 2: f_total + node frequency ---
        self.fig2, self.axs2 = plt.subplots(3, 1, figsize=(10, 7))
        self.fig2.canvas.manager.set_window_title('Total Wrench (published to device)')

        ax = self.axs2[0]
        ax.set_title("f_total — FORCE (N)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Force (N)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(self.MAX_FORCE, color='k', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.axhline(-self.MAX_FORCE, color='k', linestyle='--', linewidth=1.0, alpha=0.7)
        self.lines_ftot_F = []
        for i in range(3):
            line, = ax.plot([], [], color=colors[i], label=f"F{labels[i]}")
            self.lines_ftot_F.append(line)
        ax.legend(loc='upper left', fontsize=8, ncol=3)

        ax = self.axs2[1]
        ax.set_title("f_total — TORQUE (Nm)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Torque (Nm)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(self.MAX_TORQUE, color='k', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.axhline(-self.MAX_TORQUE, color='k', linestyle='--', linewidth=1.0, alpha=0.7)
        self.lines_ftot_T = []
        for i in range(3):
            line, = ax.plot([], [], color=colors[i], label=f"T{labels[i]}")
            self.lines_ftot_T.append(line)
        ax.legend(loc='upper left', fontsize=8, ncol=3)

        ax = self.axs2[2]
        ax.set_title("Node Frequency (Hz)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Hz")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 180)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(150, color='g', linestyle='--', linewidth=1.0, alpha=0.7, label='target 150Hz')
        self.line_freq, = ax.plot([], [], color='#1f77b4', linewidth=1.5, label='HFM freq')
        ax.legend(loc='upper left', fontsize=8)
        self.fig2.tight_layout()


        # --- Window 3 (NEW): Authority Share ---
        # Answers directly: "how much of the demanded twist/pose comes from the
        # user handle vs. the shared-autonomy policy?" All data here is read
        # VERBATIM from /shared_autonomy/blend_debug -- the exact numbers
        # main_shared_autonomy.py used to command the robot (never recomputed).
        self.fig3, self.axs3 = plt.subplots(3, 1, figsize=(10, 8))
        self.fig3.canvas.manager.set_window_title('Authority Share (User vs Policy vs Blend)')

        ax = self.axs3[0]
        ax.set_title("Twist magnitude — ||v_user|| vs ||v_policy|| vs ||v_blend||",
                    fontsize=10, fontweight='bold')
        ax.set_ylabel("Speed (m/s, rad/s)")
        ax.grid(True, linestyle='--', alpha=0.6)
        self.line_v_user, = ax.plot([], [], color='#1f77b4', linewidth=1.6, label='||v_user||')
        self.line_v_policy, = ax.plot([], [], color='#2ca02c', linewidth=1.6, label='||v_policy||')
        self.line_v_blend, = ax.plot([], [], color='#d62728', linewidth=2.0, label='||v_blend|| (commanded)')
        ax.legend(loc='upper left', fontsize=8, ncol=3)

        ax = self.axs3[1]
        ax.set_title("Blending Factor alpha [-]", fontsize=10, fontweight='bold')
        ax.set_ylabel("alpha")
        ax.set_ylim(-0.05, 1.0)
        ax.grid(True, linestyle='--', alpha=0.6)
        # alpha's hard ceiling is cfg.ALPHA_MAX (2026-07-06: alpha is now
        # belief * distance-gate, both in [0,1] -- proximity boost removed).
        ax.axhline(cfg.ALPHA_MAX, color='orange', linestyle='--', linewidth=1.3,
                   alpha=0.7, label=f'ALPHA_MAX={cfg.ALPHA_MAX}')
        self.line_alpha, = ax.plot([], [], color='#9b59b6', linewidth=2.0, label='alpha')
        ax.legend(loc='upper left', fontsize=7, ncol=1)

        ax = self.axs3[2]
        ax.set_title("Position Divergence ||pos_real - pos_user|| [m]",
                    fontsize=10, fontweight='bold')
        ax.set_ylabel("m")
        ax.set_xlabel("Time (s)")
        ax.grid(True, linestyle='--', alpha=0.6)
        self.line_pos_div, = ax.plot([], [], color='#e67e22', linewidth=1.8, label='divergence')
        ax.legend(loc='upper left', fontsize=8)
        self.fig3.tight_layout()

        plt.show(block=False)


    def update_plot(self):
        """Safely captures data and updates Matplotlib."""
        with self.plot_lock:
            if len(self.t_data) == 0:
                return
            t_list = list(self.t_data)
            sync_F = [list(self.f_data['Sync']['F'][i]) for i in range(3)]
            sync_T = [list(self.f_data['Sync']['T'][i]) for i in range(3)]
            ftot_F = [list(self.ftot_data['F'][i]) for i in range(3)]
            ftot_T = [list(self.ftot_data['T'][i]) for i in range(3)]
            freq_list = list(self._own_freq_data)
            alpha_list = list(self.alpha_data)
            v_user_list = list(self.v_user_norm_data)
            v_policy_list = list(self.v_policy_norm_data)
            v_blend_list = list(self.v_blend_norm_data)
            pos_div_list = list(self.pos_divergence_data)

        current_t = t_list[-1]
        win = (current_t - self.plot_window_sec, current_t)

        # Window 1: F_sync
        for i in range(3):
            self.lines_sync_F[i].set_data(t_list, sync_F[i])
            self.lines_sync_T[i].set_data(t_list, sync_T[i])
        for ax in self.axs1:
            ax.set_xlim(*win)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        # Window 2: f_total + freq
        for i in range(3):
            self.lines_ftot_F[i].set_data(t_list, ftot_F[i])
            self.lines_ftot_T[i].set_data(t_list, ftot_T[i])
        n_f = min(len(t_list), len(freq_list))
        self.line_freq.set_data(t_list[:n_f], freq_list[:n_f])
        self.axs2[0].set_xlim(*win)
        self.axs2[0].set_ylim(-self.MAX_FORCE * 1.3, self.MAX_FORCE * 1.3)
        self.axs2[1].set_xlim(*win)
        self.axs2[1].set_ylim(-self.MAX_TORQUE * 1.3, self.MAX_TORQUE * 1.3)
        self.axs2[2].set_xlim(*win)


        # Window 3: Authority Share
        n_a = min(len(t_list), len(v_user_list), len(v_policy_list), len(v_blend_list))
        self.line_v_user.set_data(t_list[:n_a], v_user_list[:n_a])
        self.line_v_policy.set_data(t_list[:n_a], v_policy_list[:n_a])
        self.line_v_blend.set_data(t_list[:n_a], v_blend_list[:n_a])
        self.axs3[0].set_xlim(*win)
        self.axs3[0].relim()
        self.axs3[0].autoscale_view(scalex=False, scaley=True)

        n_al = min(len(t_list), len(alpha_list))
        self.line_alpha.set_data(t_list[:n_al], alpha_list[:n_al])
        self.axs3[1].set_xlim(*win)

        n_pd = min(len(t_list), len(pos_div_list))
        self.line_pos_div.set_data(t_list[:n_pd], pos_div_list[:n_pd])
        self.axs3[2].set_xlim(*win)
        self.axs3[2].relim()
        self.axs3[2].autoscale_view(scalex=False, scaley=True)

        self.fig1.canvas.draw_idle()
        self.fig2.canvas.draw_idle()
        self.fig3.canvas.draw_idle()
        self.fig1.canvas.flush_events()


    # =========================
    # CALLBACKS
    # =========================
    def haption_pose_cb(self, msg):
        """Updates the real Cartesian orientation of the Virtuose handle."""
        q = msg.pose.orientation
        self.rot_haption = R.from_quat([q.x, q.y, q.z, q.w])

    def button_cb(self, msg):
        """Updates the clutching state from the Virtuose button."""
        self.is_clutching = msg.data

    def blend_debug_cb(self, msg):
        """Authority-share telemetry from main_shared_autonomy.py.

        Layout (19 floats): [alpha, v_user(6), v_policy(6), v_blend(6)].
        This is the SINGLE SOURCE OF TRUTH for what actually commanded the
        robot this tick -- never recomputed locally, so the plot can never
        drift from reality.
        """
        if len(msg.data) >= 19:
            self.alpha = float(msg.data[0])
            self.v_user = np.array(msg.data[1:7])
            self.v_policy = np.array(msg.data[7:13])
            self.v_blend = np.array(msg.data[13:19])

    def grasp_active_cb(self, msg):
        """Tracks whether the shared-autonomy node is autonomously driving a grasp."""
        self.grasp_active = bool(msg.data)

    def active_arm_cb(self, msg):
        """Switches which arm's EE data is used."""
        if msg.data in ('right', 'left') and msg.data != self.active_arm:
            self.active_arm = msg.data
            self.get_logger().info(f"[BLEND] Active arm switched to {msg.data.upper()}")

    def target_cb(self, msg):
        """Updates the user's raw cartesian reference (right arm)."""
        if self.active_arm != 'right':
            return
        if len(msg.data) >= 6:
            self.pos_target = np.array(msg.data[0:3])
            self.rot_target = R.from_euler('xyz', np.array(msg.data[3:6]), degrees=False)

    def target_cb_left(self, msg):
        """Updates the user's raw cartesian reference (left arm)."""
        if self.active_arm != 'left':
            return
        if len(msg.data) >= 6:
            self.pos_target = np.array(msg.data[0:3])
            self.rot_target = R.from_euler('xyz', np.array(msg.data[3:6]), degrees=False)


    def real_cb(self, msg):
        """Updates the real EE position and orientation of the active arm."""
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
        """Updates the current 6D spatial velocity of the Haption handle."""
        self.vel_haption = np.array([
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z
        ])

    def joint_cb(self, msg):
        """Updates the 6-DoF Haption joint positions."""
        if len(msg.data) >= 6:
            self.joint_pos = np.array(msg.data[0:6])


    # =========================
    # FORCE COMPONENTS
    # =========================
    def compute_F_sync(self):
        """Distance-decaying tether pulling the handle toward the real robot EE.

        A constant-stiffness spring (F = Kp * error) has a magnitude that GROWS
        with distance -- so when the operator tries to move far from the EE it
        fights them hard and, because that induced handle motion is read back as
        user twist, injects a FAKE intention into the blending loop. Instead the
        linear force MAGNITUDE now follows a 1/d-shaped law that is MAXIMUM near
        the reference and MINIMUM far away:

            |F|(d) = A/d + B,   with  |F|(SYNC_D_MIN) = SYNC_F_MAX
                                      |F|(SYNC_D_MAX) = SYNC_F_MIN
            A = (F_MAX - F_MIN) / (1/D_MIN - 1/D_MAX)
            B = F_MIN - A / D_MAX

        d = ||pos_real - pos_target|| is clamped to [SYNC_D_MIN, SYNC_D_MAX], so
        the force saturates to SYNC_F_MAX within 3cm (the lock/settle zone) and
        to SYNC_F_MIN beyond 30cm (free-roam zone -- the operator can move the
        handle freely to express intent without the tether fighting them).
        Direction is the unit vector toward the EE. The orientation torque is
        scaled by the SAME normalised distance weight so it too relaxes far out.

        self.pos_target/rot_target hold the PURE USER reference (see the
        topic-routing note in __init__).
        """
        F_sync = np.zeros(6)
        if self.pos_target is None or self.pos_real is None:
            return F_sync

        error_pos_tiago = self.pos_real - self.pos_target   # points toward the EE
        d = float(np.linalg.norm(error_pos_tiago))
        if d < 1e-6:
            return F_sync   # already matched -- no tether force

        # 1/d-shaped magnitude hitting SYNC_F_MAX at SYNC_D_MIN and SYNC_F_MIN at
        # SYNC_D_MAX (see docstring). d clamped so it saturates outside that band.
        d_c = float(np.clip(d, cfg.SYNC_D_MIN, cfg.SYNC_D_MAX))
        A = (cfg.SYNC_F_MAX - cfg.SYNC_F_MIN) / (1.0 / cfg.SYNC_D_MIN - 1.0 / cfg.SYNC_D_MAX)
        B = cfg.SYNC_F_MIN - A / cfg.SYNC_D_MAX
        f_mag = A / d_c + B

        # Normalised distance weight in [0,1]: 1 close (SYNC_F_MAX), 0 far (SYNC_F_MIN).
        w = float(np.clip((f_mag - cfg.SYNC_F_MIN) / max(cfg.SYNC_F_MAX - cfg.SYNC_F_MIN, 1e-9),
                          0.0, 1.0))

        F_spring_tiago = f_mag * (error_pos_tiago / d)   # magnitude * unit direction

        # Map TRIAGo frame -> Haption frame (180 deg Z-flip)
        F_spring_haption = np.array([-F_spring_tiago[0], -F_spring_tiago[1], F_spring_tiago[2]])
        F_sync[0:3] = F_spring_haption - (self.Kd_sync * self.vel_haption[0:3])

        # Orientation sync spring, scaled by the same distance weight so it also
        # relaxes when the operator is far from the reference.
        if self.rot_real is not None and self.rot_target is not None:
            err_rot = R.from_matrix(
                self.rot_real.as_matrix() @ self.rot_target.as_matrix().T).as_rotvec()
            Tau_tiago = self.Kp_sync_ang * w * err_rot
            F_sync[3] = -Tau_tiago[0]
            F_sync[4] = -Tau_tiago[1]
            F_sync[5] =  Tau_tiago[2]

        return F_sync


    # =========================
    # MAIN LOOP
    # =========================
    def control_loop(self):
        """Computes F_sync, applies clutch/damping, publishes the wrench, buffers plot data.

        NOTE: this node no longer computes any blending or reference integration
        -- that is now exclusively main_shared_autonomy.py's job (single source
        of truth). This loop only renders force feedback and plots telemetry.
        """
        f_sync = self.compute_F_sync()

        # --- Grasp-execution override ---
        if self.grasp_active:
            if self.pos_real is not None:
                if self._grasp_start_pos is None:
                    self._grasp_start_pos = self.pos_real.copy()
                err = self.pos_real - self._grasp_start_pos
                F_follow = self.GRASP_FOLLOW_KP * err + self.GRASP_FOLLOW_KD * self.vel_real
                F_haption = np.zeros(6)
                F_haption[0] = -F_follow[0]
                F_haption[1] = -F_follow[1]
                F_haption[2] =  F_follow[2]
                f_total = F_haption
            else:
                f_total = np.zeros(6)
        else:
            self._grasp_start_pos = None
            f_total = f_sync.copy()


        # --- Clutching Architecture ---
        if self.is_clutching and not self.grasp_active:
            if not self.was_clutching_last_frame:
                self.f_clutch_frozen = f_total / 2.0
                self.was_clutching_last_frame = True
            f_total = self.f_clutch_frozen.copy()

            # Haptic alignment guidance (orientation only, during clutch)
            if self.rot_haption is not None and self.rot_target is not None:
                error_rot_matrix = self.rot_target.as_matrix() @ self.rot_haption.as_matrix().T
                error_rot_vec = R.from_matrix(error_rot_matrix).as_rotvec()
                tau_align_base = self.K_align * error_rot_vec
                tau_align_haption = np.zeros(3)
                tau_align_haption[0] = -tau_align_base[0]
                tau_align_haption[1] = -tau_align_base[1]
                tau_align_haption[2] =  tau_align_base[2]

                # Fade near joint limits
                dist_to_min = self.joint_pos - self.joint_min
                dist_to_max = self.joint_max - self.joint_pos
                min_margin = np.min(np.concatenate([dist_to_min, dist_to_max]))
                fade_margin = 0.35
                if min_margin < fade_margin:
                    scale = max(0.0, min_margin / fade_margin)
                    tau_align_haption *= scale

                f_total[3:6] += tau_align_haption
        else:
            self.was_clutching_last_frame = False

        # --- Global Damping ---
        Kd_global_lin = 0.7
        Kd_global_ang = 0.1
        f_total[0:3] -= Kd_global_lin * self.vel_haption[0:3]
        f_total[3:6] -= Kd_global_ang * self.vel_haption[3:6]

        # --- Clipping & Publishing ---
        f_total[0:3] = np.clip(f_total[0:3], -self.MAX_FORCE, self.MAX_FORCE)
        f_total[3:6] = np.clip(f_total[3:6], -self.MAX_TORQUE, self.MAX_TORQUE)

        msg = Wrench()
        msg.force.x = float(f_total[0])
        msg.force.y = float(f_total[1])
        msg.force.z = float(f_total[2])
        msg.torque.x = float(f_total[3])
        msg.torque.y = float(f_total[4])
        msg.torque.z = float(f_total[5])
        self.force_pub.publish(msg)


        # --- Buffer Data for Plotting ---
        t = time.time() - self.start_time
        pos_div = (float(np.linalg.norm(self.pos_real - self.pos_target))
                  if (self.pos_real is not None and self.pos_target is not None) else 0.0)
        with self.plot_lock:
            self.t_data.append(t)
            for i in range(3):
                self.f_data['Sync']['F'][i].append(f_sync[i])
                self.f_data['Sync']['T'][i].append(f_sync[i + 3])
            for i in range(3):
                self.ftot_data['F'][i].append(f_total[i])
                self.ftot_data['T'][i].append(f_total[i + 3])

            # Authority-share telemetry (verbatim from /shared_autonomy/blend_debug)
            self.alpha_data.append(self.alpha)
            self.v_user_norm_data.append(float(np.linalg.norm(self.v_user[0:3])))
            self.v_policy_norm_data.append(float(np.linalg.norm(self.v_policy[0:3])))
            self.v_blend_norm_data.append(float(np.linalg.norm(self.v_blend[0:3])))
            self.pos_divergence_data.append(pos_div)

            # Node frequency tracking
            now = time.time()
            if self._own_last_time is not None:
                dt_own = now - self._own_last_time
                if dt_own > 1e-6:
                    self._own_freq_lpf = 0.9 * self._own_freq_lpf + 0.1 * (1.0 / dt_own)
            self._own_last_time = now
            self._own_freq_data.append(self._own_freq_lpf)



def main(args=None):
    """Initializes ROS, spins the node on a daemon thread, and drives Matplotlib on main thread."""
    rclpy.init(args=args)
    node = HapticForceManagerBlending()

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
