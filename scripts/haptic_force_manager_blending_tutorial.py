#!/usr/bin/env python3
"""Haptic Force Manager — BLENDING TUTORIAL.

Architecture:
  - The ONLY force rendered on the Haption device is F_sync (spring-damper
    tether keeping the operator synced with the real robot EE).
  - NO F_guide, NO F_fixture, NO F_cbf at the haptic level.
  - Instead, the user's Cartesian reference is BLENDED with the belief-weighted
    assistive trajectory BEFORE being published to the QP CLF-CBF controller.
  - The blended reference is what the green gripper (RViz) visualizes.

Blending:
  ref_blended = (1 - alpha) * ref_user + alpha * ref_assistive

  where:
    ref_assistive = belief-weighted sum of per-goal assistive twists (integrated),
                    ensuring continuity in the reference as beliefs shift.
    alpha = ALPHA_MAX * max_belief   (continuous function of the peak belief,
            capped so the user ALWAYS retains at least 20% authority).
    ALPHA_MAX = 0.80  ->  at 100% belief the autonomy contributes 80%.

Force output:
  f_total = F_sync + F_limit_warning + global_damping

Plot:
  Window 1: F_sync wrench (3 force + 3 torque)
  Window 2: f_total (published wrench), alpha blending factor, node frequency
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

        # --- Blending Parameters ---
        self.ALPHA_MAX = 0.80        # Max autonomy authority (user keeps >= 20%)
        self.alpha = 0.0             # Current blending factor (updated every tick)
        self.alpha_lpf = 0.0         # Low-pass filtered alpha for smooth transitions
        self.ALPHA_LPF_COEFF = 0.08  # LPF coefficient (lower = smoother)


        # --- Inference State Variables ---
        self.goal_names = []
        self.goal_probs = []
        self.user_policies = []      # Flattened array of per-goal optimal twists (n_goals * 6)

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
        self.Kp_sync = 10.0
        self.Kd_sync = 0.0
        self.Kp_sync_ang = 0.3       # Nm/rad orientation sync spring

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

        # Alpha blending factor history
        self.alpha_data = deque(maxlen=self.buffer_size)

        # Own node frequency tracker
        self._own_freq_data = deque(maxlen=self.buffer_size)
        self._own_last_time = None
        self._own_freq_lpf = 0.0


        # --- Subscribers ---
        self.create_subscription(Float64MultiArray, '/arm_right/cartesian_reference', self.target_cb, 10)
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self.real_cb, 10)
        self.create_subscription(Twist, 'virtuose/velocity', self.vel_cb, 10)
        self.create_subscription(Float64MultiArray, 'virtuose/articular_position', self.joint_cb, 10)
        self.create_subscription(Bool, 'virtuose/button_right', self.button_cb, 10)
        self.create_subscription(PoseStamped, 'virtuose/pose', self.haption_pose_cb, 10)
        # Shared autonomy inference state
        self.create_subscription(String, '/shared_autonomy/goal_names', self.goal_names_cb, 10)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/goal_probabilities', self.goal_probs_cb, 10)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/user_policy', self.user_policy_cb, 10)
        # Grasp execution flag
        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_cb, 10)
        # Arm switch
        self.active_arm = 'right'
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)
        self.create_subscription(Float64MultiArray, '/arm_left/cartesian_reference', self.target_cb_left, 10)

        # --- Publishers ---
        self.force_pub = self.create_publisher(Wrench, 'virtuose/force_cmd', 10)
        # Blended cartesian reference output (replaces the user's raw reference)
        self.blended_ref_pub = self.create_publisher(
            Float64MultiArray, '/arm_right/blended_cartesian_reference', 10)

        # --- Timers ---
        self.dt = 1.0 / 150.0
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.setup_plot()
        self.get_logger().info("Haptic Force Manager (BLENDING tutorial) started.")


    # =========================
    # PLOT SETUP & UPDATE
    # =========================
    def setup_plot(self):
        """Initializes the live Matplotlib windows (simplified: Sync only + alpha)."""
        plt.ion()

        # --- Window 1: F_sync wrench components ---
        self.fig1, self.axs1 = plt.subplots(2, 1, figsize=(10, 5))
        self.fig1.canvas.manager.set_window_title('Haptic Force: F_sync Only')

        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']

        # Row 0: Sync Force
        ax = self.axs1[0]
        ax.set_title("F_sync — FORCE (N)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Force (N)")
        ax.grid(True, linestyle='--', alpha=0.6)
        self.lines_sync_F = []
        for i in range(3):
            line, = ax.plot([], [], color=colors[i], label=f"F{labels[i]}")
            self.lines_sync_F.append(line)
        ax.legend(loc='upper left', fontsize=8)

        # Row 1: Sync Torque
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


        # --- Window 2: f_total + alpha + frequency ---
        self.fig2, self.axs2 = plt.subplots(4, 1, figsize=(10, 9))
        self.fig2.canvas.manager.set_window_title('Total Wrench + Blending Alpha')

        # Row 0: f_total Force
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

        # Row 1: f_total Torque
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


        # Row 2: Alpha blending factor
        ax = self.axs2[2]
        ax.set_title("Blending Factor (alpha)", fontsize=10, fontweight='bold')
        ax.set_ylabel("alpha [-]")
        ax.set_ylim(-0.05, 1.0)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(self.ALPHA_MAX, color='orange', linestyle='--', linewidth=1.5,
                   alpha=0.8, label=f'ALPHA_MAX={self.ALPHA_MAX}')
        self.line_alpha, = ax.plot([], [], color='#9b59b6', linewidth=2.0, label='alpha')
        ax.legend(loc='upper left', fontsize=8)

        # Row 3: Node frequency
        ax = self.axs2[3]
        ax.set_title("Node Frequency (Hz)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Hz")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 180)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(150, color='g', linestyle='--', linewidth=1.0, alpha=0.7, label='target 150Hz')
        self.line_freq, = ax.plot([], [], color='#1f77b4', linewidth=1.5, label='HFM freq')
        ax.legend(loc='upper left', fontsize=8)

        self.fig2.tight_layout()
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
            alpha_list = list(self.alpha_data)
            freq_list = list(self._own_freq_data)

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

        # Window 2: f_total + alpha + freq
        for i in range(3):
            self.lines_ftot_F[i].set_data(t_list, ftot_F[i])
            self.lines_ftot_T[i].set_data(t_list, ftot_T[i])
        n = min(len(t_list), len(alpha_list))
        self.line_alpha.set_data(t_list[:n], alpha_list[:n])
        n_f = min(len(t_list), len(freq_list))
        self.line_freq.set_data(t_list[:n_f], freq_list[:n_f])

        self.axs2[0].set_xlim(*win)
        self.axs2[0].set_ylim(-self.MAX_FORCE * 1.3, self.MAX_FORCE * 1.3)
        self.axs2[1].set_xlim(*win)
        self.axs2[1].set_ylim(-self.MAX_TORQUE * 1.3, self.MAX_TORQUE * 1.3)
        self.axs2[2].set_xlim(*win)
        self.axs2[3].set_xlim(*win)

        self.fig1.canvas.draw_idle()
        self.fig2.canvas.draw_idle()
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

    def goal_names_cb(self, msg):
        """Updates the list of active goal names."""
        self.goal_names = msg.data.split(',')

    def goal_probs_cb(self, msg):
        """Updates the array of goal probabilities."""
        self.goal_probs = list(msg.data)

    def user_policy_cb(self, msg):
        """Updates the flattened array of per-goal optimal spatial twists."""
        self.user_policies = list(msg.data)

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
        """Spring-damper tether: keeps the operator synced with the real robot EE."""
        F_sync = np.zeros(6)
        if self.pos_target is None or self.pos_real is None:
            return F_sync

        # Position spring: pulls the handle toward the real EE
        error_pos_tiago = self.pos_real - self.pos_target
        F_spring_tiago = self.Kp_sync * error_pos_tiago

        # Map TRIAGo frame -> Haption frame (180 deg Z-flip)
        F_spring_haption = np.zeros(3)
        F_spring_haption[0] = -F_spring_tiago[0]
        F_spring_haption[1] = -F_spring_tiago[1]
        F_spring_haption[2] =  F_spring_tiago[2]

        F_damped_haption = F_spring_haption - (self.Kd_sync * self.vel_haption[0:3])
        F_sync[0:3] = F_damped_haption

        # Orientation sync spring
        if self.rot_real is not None and self.rot_target is not None:
            err_rot = R.from_matrix(
                self.rot_real.as_matrix() @ self.rot_target.as_matrix().T).as_rotvec()
            Tau_tiago = self.Kp_sync_ang * err_rot
            F_sync[3] = -Tau_tiago[0]
            F_sync[4] = -Tau_tiago[1]
            F_sync[5] =  Tau_tiago[2]

        return F_sync


    # =========================
    # BLENDING LOGIC
    # =========================
    def compute_alpha(self):
        """Compute the blending factor as a continuous function of the peak belief.

        alpha = ALPHA_MAX * max(beliefs)

        Properties:
          - When all beliefs are uniform (1/N), alpha is small -> mostly user.
          - As belief concentrates on one goal (max -> 1.0), alpha -> ALPHA_MAX.
          - User always retains at least (1 - ALPHA_MAX) = 20% authority.
          - Continuous and differentiable in the probabilities.
        """
        if not self.goal_probs or len(self.goal_probs) == 0:
            return 0.0

        max_belief = max(self.goal_probs)
        # Scale so alpha=0 at uniform and alpha=ALPHA_MAX at certainty.
        # With N goals, uniform max_belief = 1/N. Map [1/N, 1] -> [0, ALPHA_MAX].
        n_goals = len(self.goal_probs)
        uniform_max = 1.0 / n_goals if n_goals > 0 else 1.0
        if max_belief <= uniform_max:
            return 0.0
        # Linear mapping from [uniform, 1] -> [0, ALPHA_MAX]
        raw_alpha = self.ALPHA_MAX * (max_belief - uniform_max) / (1.0 - uniform_max)
        return float(np.clip(raw_alpha, 0.0, self.ALPHA_MAX))

    def compute_blended_reference(self):
        """Compute the belief-weighted assistive twist and blend with user reference.

        Returns:
            blended_pos (np.ndarray[3]): blended position reference
            blended_rpy (np.ndarray[3]): blended orientation (euler xyz)
            alpha_used (float): the alpha factor actually applied this tick
        Returns (None, None, 0.0) if blending cannot be performed.
        """
        if self.pos_target is None or self.rot_target is None:
            return None, None, 0.0

        n_goals = len(self.goal_names) if self.goal_names else 0
        n_policies = len(self.user_policies)

        # If no valid inference data, pass through the user reference unchanged
        if (n_goals == 0
                or len(self.goal_probs) != n_goals
                or n_policies != n_goals * 6):
            return self.pos_target.copy(), self.rot_target.as_euler('xyz'), 0.0


        # Compute alpha
        alpha_raw = self.compute_alpha()
        # Low-pass filter for smoothness (no sudden jumps in authority)
        self.alpha_lpf = (self.ALPHA_LPF_COEFF * alpha_raw
                          + (1.0 - self.ALPHA_LPF_COEFF) * self.alpha_lpf)
        alpha = self.alpha_lpf

        # Compute the belief-weighted assistive twist (robot frame)
        probs = np.array(self.goal_probs)
        policies = np.array(self.user_policies).reshape(n_goals, 6)
        # pi_assistive = sum_k P(k) * pi_k  (convex combination -> continuous)
        pi_assistive = probs @ policies  # shape (6,)

        # Integrate the assistive twist into a position/orientation offset
        # relative to the user's current reference (one tick of dt)
        dp_assist = pi_assistive[0:3] * self.dt   # small position delta
        dw_assist = pi_assistive[3:6] * self.dt   # small orientation delta

        # Assistive reference = user_ref + assistive_delta (robot frame)
        pos_assist = self.pos_target + dp_assist
        # Orientation: compose a small rotation dR onto the user's orientation
        dR = R.from_rotvec(dw_assist)
        rot_assist = dR * self.rot_target

        # Blend: ref_blended = (1 - alpha) * user + alpha * assistive
        blended_pos = (1.0 - alpha) * self.pos_target + alpha * pos_assist
        # Orientation blending via SLERP (alpha=0 -> user, alpha=1 -> assistive)
        # For small alpha and small dw_assist, linear RPY blend is acceptable,
        # but SLERP is correct for all cases.
        from scipy.spatial.transform import Slerp
        key_rots = R.concatenate([self.rot_target, rot_assist])
        slerp = Slerp([0.0, 1.0], key_rots)
        rot_blended = slerp(alpha)
        blended_rpy = rot_blended.as_euler('xyz')

        return blended_pos, blended_rpy, alpha


    # =========================
    # MAIN LOOP
    # =========================
    def control_loop(self):
        """Computes forces, performs blending, publishes wrench and blended reference."""
        # --- Compute the sole force: F_sync ---
        f_sync = self.compute_F_sync()

        # --- Compute blended reference and publish ---
        blended_pos, blended_rpy, alpha = self.compute_blended_reference()
        self.alpha = alpha

        if blended_pos is not None and blended_rpy is not None:
            ref_msg = Float64MultiArray()
            # Publish in the same format as the user's cartesian_reference:
            # [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, ...]
            # We pass position + orientation; velocities are kept from the user's
            # original reference (the QP integrator handles velocity tracking).
            ref_msg.data = list(blended_pos) + list(blended_rpy)
            self.blended_ref_pub.publish(ref_msg)

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
        with self.plot_lock:
            self.t_data.append(t)
            for i in range(3):
                self.f_data['Sync']['F'][i].append(f_sync[i])
                self.f_data['Sync']['T'][i].append(f_sync[i + 3])
            for i in range(3):
                self.ftot_data['F'][i].append(f_total[i])
                self.ftot_data['T'][i].append(f_total[i + 3])
            self.alpha_data.append(self.alpha)

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
