#!/usr/bin/env python3
"""Haptic Force Manager -- JOYSTICK GUIDED FEEDBACK (JF: Feedback only, no Blending).

The guided-feedback cell of the velocity-control column: ONLY the assistive
FEEDBACK channel (F) is active on the spring-centered joystick; reference-level
blending (channel B) is OFF. It is a copy of the JFB (full-guidance) manager with
the study guard flipped to (JOYSTICK, feedback=True, blending=False).

Because blending is performed at the REFERENCE level by main_shared_autonomy (NOT
by the force manager), the wrench rendered on the handle is IDENTICAL to JFB:
the SUPERPOSITION of

  1. F_home  -- the restorative centering spring toward the (dynamic) home pose
                (identical to the JB manager), and
  2. F_guide -- the belief-weighted velocity-field guidance force (feed-forward),
                calibrated so it saturates at the DEADZONE-EXIT force.

The only functional difference vs JFB is that ASSIST_BLENDING=False, so
main_shared_autonomy does NOT own /arm_*/cartesian_reference -- the joystick teleop
drives the raw (un-blended) reference, and the operator feels the guidance forces
on the handle without any autonomous reference arbitration. The guidance therefore
BIASES the handle toward the inferred goal (a "preferred direction") but every bit
of robot motion still comes from the operator's own twist.

Design of the F_guide / F_home overlap (unchanged from JFB):
  The guidance saturates at the DEADZONE-EXIT force -- the force that, against the
  centering spring, displaces the handle exactly to the edge of the joystick
  deadband:
        F_exit_lin = KP_LIN * DEADBAND_LIN ,   Tau_exit_ang = KP_ANG * DEADBAND_ANG
  MAX_GUIDE_{FORCE,TORQUE} = GUIDE_K * F_exit, and the guidance is scaled LINEARLY
  by  gain = confidence(b_max) x proximity(ref->goal). With GUIDE_K < 1 (currently
  0.55) the guidance is a pure BIAS -- it can no longer clear the deadband on its
  own, even at gain=1, so the operator always drives.
  F_guide is FEED-FORWARD (built from v_field only, no -handle_vel term): CFB's
  velocity-field self-damping is a virtual damper too strong for the 150 Hz device
  passivity limit; handle damping comes from the centering spring's KD + friction.

Final wrench (Haption base frame): F_home + F_guide, clipped to +/- MAX_FORCE /
MAX_TORQUE, published on virtuose/force_cmd.

Plots (two windows, per the JF plan):
  1. CONTRIBUTIONS -- homing (F_home) vs assistance (F_guide) magnitude on the
     handle (force and torque), each as its OWN line, with dashed lines for the
     maxima (guidance saturation MAX_GUIDE_* and the device clip MAX_FORCE/TORQUE).
  2. FEEDBACK SHARE -- what fraction of the total wrench is homing vs assistance
     (% of |F_home|+|F_guide|), for force and torque separately.
"""

import threading
import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench, Twist, Pose
from std_msgs.msg import Float64MultiArray, String, Bool

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Single source of truth for the joystick home pose, spring gains, deadbands and
# the experiment-condition selector.
import triago_control.qp_controller.config as cfg


class HapticForceManagerJF(Node):
    def __init__(self):
        super().__init__('haptic_force_manager_jf')

        # Fail loudly if launched under the wrong study condition. This is the
        # JOYSTICK "Guided feedback" manager: assistive feedback (channel F) ON,
        # reference blending (channel B) OFF. The handle renders the centering
        # spring + the belief-weighted guidance force, but the reference is NOT
        # blended (main_shared_autonomy does not own /arm_*/cartesian_reference).
        cfg.validate_condition('haptic_force_manager_JF',
                               control_mode=cfg.JOYSTICK, feedback=True, blending=False)

        # --- Home pose (Haption base frame) -- neutral until teleop publishes ---
        self.home_pos = np.array(cfg.JOYSTICK_NEUTRAL_POSITION_M, dtype=float)
        self.home_rot = R.from_quat(cfg.JOYSTICK_NEUTRAL_ORIENTATION_XYZW)  # xyzw

        # --- Live handle state (Haption base frame) ---
        self.handle_pos = None
        self.handle_rot = None
        self.handle_vel = np.zeros(6)

        # --- Home centering-spring gains (same as JB) ---
        self.KP_LIN = cfg.JOYSTICK_SPRING_KP_LIN
        self.KD_LIN = cfg.JOYSTICK_SPRING_KD_LIN
        self.KP_ANG = cfg.JOYSTICK_SPRING_KP_ANG
        self.KD_ANG = cfg.JOYSTICK_SPRING_KD_ANG

        # --- Device safety clip ---
        self.MAX_FORCE = 10.0
        self.MAX_TORQUE = 1.0

        # --- Guidance (F_guide) inference inputs (from main_shared_autonomy) ---
        self.goal_names = []
        self.goal_probs = []
        self.user_policies = []
        self.fix_goal_pos = None      # active goal position (base frame)
        self.fix_confidence = 0.0     # b_max: active-goal belief (0 during grasp exec)
        self.pos_target = None        # the reference the QP tracks
        self.active_arm = 'right'

        # --- Guidance tuning, DEFINED RELATIVE TO THE HOME SPRING ---------------
        # Saturation = deadzone-exit force x GUIDE_K. GUIDE_K < 1 -> the guidance
        # BIASES the handle toward the goal but does NOT clear the deadband on its
        # own (even at gain=1); the operator feels a preferred direction yet still
        # drives.
        self.GUIDE_K = 0.55
        self.MAX_GUIDE_FORCE  = self.GUIDE_K * self.KP_LIN * cfg.JOYSTICK_DEADBAND_LIN   # ~2.85 N
        self.MAX_GUIDE_TORQUE = self.GUIDE_K * self.KP_ANG * cfg.JOYSTICK_DEADBAND_ANG   # ~0.21 Nm
        # FEED-FORWARD magnitude-shaping gains (policy speed -> force), NOT velocity
        # feedback: F_guide is built from v_field ONLY (no -handle_vel term), so
        # these are not virtual dampers and don't threaten device passivity.
        self.D_guide_lin = 400.0   # N per (m/s) of policy speed (feed-forward shaping)
        self.D_guide_ang = 30.0    # Nm per (rad/s) of policy speed (feed-forward shaping)
        self.alpha_guide = 0.15    # LPF on the guidance wrench (C0 continuity)
        self.f_guide_filtered = np.zeros(6)
        # gain = confidence(b_max) x proximity(ref->goal).
        # UNIFIED across ALL guidance cells (JF / JFB / CF / CFB): the task, belief
        # function and policy are identical for both teleop modes, so the guidance
        # activation margins MUST match everywhere:
        #   proximity : full <= 0.10 m, dead > 0.60 m
        #   confidence: dead < 0.30,   full >= 0.90
        self.GUIDE_CONF_LO = 0.30   # b_max below this -> guidance dead
        self.GUIDE_CONF_HI = 0.90   # b_max at/above this -> full confidence gate
        self.GUIDE_PROX_NEAR = 0.10  # m: ref->goal distance at/below this -> full proximity gate
        self.GUIDE_PROX_FAR  = 0.60  # m: at/beyond this -> guidance dead

        # --- Out-of-deadzone vibration cue (same as J / JB / JFB) ---
        # A constant, low-amplitude, zero-mean buzz on the three torque axes,
        # rendered the WHOLE time the handle sits OUTSIDE the joystick deadband --
        # i.e. exactly while a non-zero twist is being commanded.
        self.VIB_AMP = 0.05           # Nm  constant torque amplitude while outside the deadband
        self.vib_toggle = 1.0         # per-frame sign toggle (~75 Hz square wave)

        # --- Grasp-execution vibration cue (UNIFIED with clutch cells) ---
        self.grasp_active = False
        self.GRASP_VIB_AMP = 0.07    # Nm  constant buzz during autonomous grasp
        self.grasp_vib_toggle = 1.0

        # --- Subscribers ---
        # NOTE: virtuose_server_node publishes virtuose/pose as geometry_msgs/Pose.
        self.create_subscription(Pose, 'virtuose/pose', self.handle_pose_cb, 10)
        self.create_subscription(Twist, 'virtuose/velocity', self.vel_cb, 10)
        self.create_subscription(
            Float64MultiArray, cfg.JOYSTICK_HOME_POSE_TOPIC, self.home_pose_cb, 10)
        # Guidance inference state (belief-weighted policy field + active goal).
        self.create_subscription(String, '/shared_autonomy/goal_names', self.goal_names_cb, 10)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/goal_probabilities', self.goal_probs_cb, 10)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/user_policy', self.user_policy_cb, 10)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/active_goal_pose', self.goal_pose_cb, 10)
        # The reference the QP tracks -> proximity distance ref->goal. With B=0 this
        # is the raw teleop reference (not a blended one), which is still valid for
        # the proximity gate.
        self.create_subscription(Float64MultiArray, '/arm_right/cartesian_reference', self.target_cb, 10)
        self.create_subscription(Float64MultiArray, '/arm_left/cartesian_reference', self.target_cb_left, 10)
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)
        # Grasp-execution flag: vibrate during autonomous grasp.
        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_cb, 10)

        # --- Publisher ---
        self.force_pub = self.create_publisher(Wrench, 'virtuose/force_cmd', 10)

        # --- Plot buffers (home + guide wrench components; norms/share derived) ---
        self.plot_lock = threading.Lock()
        self.plot_window_sec = 10.0
        self.buffer_size = int(150 * self.plot_window_sec)
        self.start_time = time.time()
        self.t_data = deque(maxlen=self.buffer_size)
        self.fh_data = [deque(maxlen=self.buffer_size) for _ in range(6)]  # home wrench
        self.fg_data = [deque(maxlen=self.buffer_size) for _ in range(6)]  # guide wrench

        self.dt = 1.0 / 150.0
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.setup_plot()
        self.get_logger().info(
            f"[HFM-JF] Joystick GUIDED-FEEDBACK manager started (F=1, B=0). "
            f"F_home spring KP_LIN={self.KP_LIN}, KP_ANG={self.KP_ANG}; "
            f"F_guide saturates at exit force MAX_GUIDE_FORCE={self.MAX_GUIDE_FORCE:.2f} N / "
            f"MAX_GUIDE_TORQUE={self.MAX_GUIDE_TORQUE:.3f} Nm (GUIDE_K={self.GUIDE_K}). "
            f"Reference NOT blended (B=0).")

    # ------------------------------------------------------------------ callbacks
    def handle_pose_cb(self, msg):
        p = msg.position
        q = msg.orientation
        self.handle_pos = np.array([p.x, p.y, p.z])
        self.handle_rot = R.from_quat([q.x, q.y, q.z, q.w])

    def vel_cb(self, msg):
        self.handle_vel = np.array([
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z])

    def home_pose_cb(self, msg):
        """Live home pose from teleop_triago_joystick.py: [pos(3), quat_xyzw(4)]."""
        if len(msg.data) >= 7:
            self.home_pos = np.array(msg.data[0:3])
            self.home_rot = R.from_quat(np.array(msg.data[3:7]))

    def goal_names_cb(self, msg):
        self.goal_names = msg.data.split(',')

    def goal_probs_cb(self, msg):
        self.goal_probs = list(msg.data)

    def user_policy_cb(self, msg):
        """Flattened optimal spatial twists (per goal) evaluated from the reference."""
        self.user_policies = list(msg.data)

    def goal_pose_cb(self, msg):
        """Active goal pose + confidence: [x,y,z, r,p,y, b_max]. b_max drives the
        confidence gate; it is 0 during autonomous grasp execution (guidance off)."""
        if len(msg.data) >= 7:
            self.fix_goal_pos = np.array(msg.data[0:3])
            self.fix_confidence = float(msg.data[6])

    def target_cb(self, msg):
        if self.active_arm != 'right':
            return
        if len(msg.data) >= 3:
            self.pos_target = np.array(msg.data[0:3])

    def target_cb_left(self, msg):
        if self.active_arm != 'left':
            return
        if len(msg.data) >= 3:
            self.pos_target = np.array(msg.data[0:3])

    def active_arm_cb(self, msg):
        if msg.data in ('right', 'left') and msg.data != self.active_arm:
            self.active_arm = msg.data
            self.get_logger().info(f"[HFM-JF] Active arm switched to {msg.data.upper()}")

    def grasp_active_cb(self, msg):
        """Tracks whether the shared-autonomy node is autonomously driving a grasp."""
        self.grasp_active = bool(msg.data)

    # ------------------------------------------------------------------ helpers
    def _smoothstep(self, p, lo, hi):
        """C1-continuous ramp from 0 at p=lo to 1 at p=hi."""
        if p <= lo:
            return 0.0
        x = min((p - lo) / (hi - lo), 1.0)
        return 3.0 * x ** 2 - 2.0 * x ** 3

    # ------------------------------------------------------------------ forces
    def compute_spring(self):
        """Restorative spring-damper wrench (Haption base frame) toward home
        (identical to the JB manager)."""
        f = np.zeros(6)
        if self.handle_pos is None or self.handle_rot is None:
            return f
        f[0:3] = self.KP_LIN * (self.home_pos - self.handle_pos) - self.KD_LIN * self.handle_vel[0:3]
        err_rotvec = (self.home_rot * self.handle_rot.inv()).as_rotvec()
        f[3:6] = self.KP_ANG * err_rotvec - self.KD_ANG * self.handle_vel[3:6]
        return f

    def compute_F_guide(self):
        """Velocity-field guidance (copied from JFB/CFB.compute_F_guide), calibrated
        to overlap the home spring.

        Structure:
          1. pi_blend = Sum_k P(k) * pi_k          (belief-weighted policy twist)
          2. v_field  = map_180Z(pi_blend)         (robot -> Haption frame)
          3. FEED-FORWARD force F = MAX * tanh(D * v_field / MAX) (direction from
             the policy field, magnitude saturated at the exit force), tanh-sat, LPF'd.

        gain = confidence(b_max) x proximity(ref->goal), applied AFTER the tanh
        (linear 'fraction of the exit force'), so gain=1 exits the deadband and
        gain<1 only biases inside it.
        """
        n_goals = len(self.goal_names) if self.goal_names else 0
        n_policies = len(self.user_policies)
        if (n_goals == 0
                or len(self.goal_probs) != n_goals
                or n_policies != n_goals * 6):
            self.f_guide_filtered = (1.0 - self.alpha_guide) * self.f_guide_filtered
            return self.f_guide_filtered.copy()

        probs = np.array(self.goal_probs)
        policies = np.array(self.user_policies).reshape(n_goals, 6)
        pi_blend = probs @ policies

        # Confidence gate: the ACTIVE-goal belief b_max. Dead below LO, full at/above HI.
        conf_gate = self._smoothstep(self.fix_confidence,
                                     lo=self.GUIDE_CONF_LO, hi=self.GUIDE_CONF_HI)

        # Proximity gate: distance from the REFERENCE to the active goal.
        # Full at/below NEAR, dead at/beyond FAR (smoothstep between).
        if self.fix_goal_pos is not None and self.pos_target is not None:
            d_goal = float(np.linalg.norm(self.fix_goal_pos - self.pos_target))
            prox = np.clip(
                (self.GUIDE_PROX_FAR - d_goal)
                / max(self.GUIDE_PROX_FAR - self.GUIDE_PROX_NEAR, 1e-6), 0.0, 1.0)
            prox_gate = 3.0 * prox ** 2 - 2.0 * prox ** 3   # smoothstep
        else:
            prox_gate = 0.0   # no goal / reference info -> no guidance (safe)

        gain = conf_gate * prox_gate

        # Map the policy twist (robot frame) into the Haption frame (180 deg Z-flip),
        # matching teleop_triago_joystick's frame convention.
        v_field = np.array([
            -pi_blend[0], -pi_blend[1],  pi_blend[2],
            -pi_blend[3], -pi_blend[4],  pi_blend[5],
        ])

        # FEED-FORWARD guidance: shape the magnitude from the (smooth, LPF'd) policy
        # field v_field ONLY -- do NOT feed the handle velocity back (that virtual
        # damper is too strong for the 150 Hz device). Direction from v_field,
        # magnitude saturated at the exit force, then linearly gain-scaled.
        F_dir = np.zeros(6)
        F_dir[0:3] = self.MAX_GUIDE_FORCE * np.tanh(self.D_guide_lin * v_field[0:3] / self.MAX_GUIDE_FORCE)
        F_dir[3:6] = self.MAX_GUIDE_TORQUE * np.tanh(self.D_guide_ang * v_field[3:6] / self.MAX_GUIDE_TORQUE)
        F_guide_raw = gain * F_dir

        # Temporal smoothing (LPF)
        self.f_guide_filtered = (self.alpha_guide * F_guide_raw
                                 + (1.0 - self.alpha_guide) * self.f_guide_filtered)
        return self.f_guide_filtered.copy()

    # ------------------------------------------------------------------ loop
    def control_loop(self):
        f_home = self.compute_spring()
        f_guide = self.compute_F_guide()
        f_total = f_home + f_guide

        # --- Out-of-deadzone vibration cue (same as J / JB / JFB) ---
        if self.handle_pos is not None and self.handle_rot is not None:
            lin_disp = float(np.linalg.norm(self.home_pos - self.handle_pos))
            ang_disp = float(np.linalg.norm((self.home_rot * self.handle_rot.inv()).as_rotvec()))
            if (lin_disp > cfg.JOYSTICK_DEADBAND_LIN
                    or ang_disp > cfg.JOYSTICK_DEADBAND_ANG):
                self.vib_toggle *= -1.0
                buzz = self.VIB_AMP * self.vib_toggle
                f_total[3] += buzz
                f_total[4] += buzz
                f_total[5] += buzz

        # --- Grasp vibration cue (0.07 Nm buzz during autonomous grasp) ---
        if self.grasp_active:
            self.grasp_vib_toggle *= -1.0
            gb = self.GRASP_VIB_AMP * self.grasp_vib_toggle
            f_total[3] += gb
            f_total[4] += gb
            f_total[5] += gb

        f_total[0:3] = np.clip(f_total[0:3], -self.MAX_FORCE, self.MAX_FORCE)
        f_total[3:6] = np.clip(f_total[3:6], -self.MAX_TORQUE, self.MAX_TORQUE)

        msg = Wrench()
        msg.force.x, msg.force.y, msg.force.z = float(f_total[0]), float(f_total[1]), float(f_total[2])
        msg.torque.x, msg.torque.y, msg.torque.z = float(f_total[3]), float(f_total[4]), float(f_total[5])
        self.force_pub.publish(msg)

        t = time.time() - self.start_time
        with self.plot_lock:
            self.t_data.append(t)
            for i in range(6):
                self.fh_data[i].append(f_home[i])
                self.fg_data[i].append(f_guide[i])

    # ------------------------------------------------------------------ plotting
    def setup_plot(self):
        plt.ion()

        # --- Window 1: CONTRIBUTIONS -- homing vs assistance magnitude ----------
        # The magnitude each channel puts on the handle: ||F_home|| (homing spring)
        # vs ||F_guide|| (assistance), for force and torque separately, each its
        # OWN line. Dashed lines mark the maxima: the guidance saturation
        # (MAX_GUIDE_*) and the device clip (MAX_FORCE / MAX_TORQUE).
        self.fig_c, self.axs_c = plt.subplots(2, 1, figsize=(10, 7))
        self.fig_c.canvas.manager.set_window_title('JF: Homing vs Assistance contributions')

        ax = self.axs_c[0]
        ax.set_title('FORCE contribution on handle (N)', fontsize=10, fontweight='bold')
        ax.set_ylabel('|F| (N)'); ax.grid(True, linestyle='--', alpha=0.6)
        self.l_home_F, = ax.plot([], [], color='#2980b9', linewidth=1.8, label='homing  ||F_home||')
        self.l_guide_F, = ax.plot([], [], color='#e67e22', linewidth=1.8, label='assistance  ||F_guide||')
        ax.axhline(self.MAX_GUIDE_FORCE, color='#e67e22', linestyle='--', linewidth=1.2,
                   alpha=0.8, label=f'guide max = {self.MAX_GUIDE_FORCE:.2f} N')
        ax.axhline(self.MAX_FORCE, color='k', linestyle='--', linewidth=1.0,
                   alpha=0.6, label=f'device clip = {self.MAX_FORCE:.0f} N')
        ax.legend(loc='upper left', fontsize=8, ncol=2)

        ax = self.axs_c[1]
        ax.set_title('TORQUE contribution on handle (Nm)', fontsize=10, fontweight='bold')
        ax.set_ylabel('|T| (Nm)'); ax.set_xlabel('Time (s)'); ax.grid(True, linestyle='--', alpha=0.6)
        self.l_home_T, = ax.plot([], [], color='#2980b9', linewidth=1.8, label='homing  ||T_home||')
        self.l_guide_T, = ax.plot([], [], color='#e67e22', linewidth=1.8, label='assistance  ||T_guide||')
        ax.axhline(self.MAX_GUIDE_TORQUE, color='#e67e22', linestyle='--', linewidth=1.2,
                   alpha=0.8, label=f'guide max = {self.MAX_GUIDE_TORQUE:.3f} Nm')
        ax.axhline(self.MAX_TORQUE, color='k', linestyle='--', linewidth=1.0,
                   alpha=0.6, label=f'device clip = {self.MAX_TORQUE:.1f} Nm')
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        self.fig_c.tight_layout()

        # --- Window 2: FEEDBACK SHARE -- homing vs assistance -------------------
        # What fraction of the total feedback is homing vs assistance:
        #   share_home  = ||F_home||  / (||F_home|| + ||F_guide||) * 100
        #   share_guide = ||F_guide|| / (||F_home|| + ||F_guide||) * 100
        # (the two sum to 100%), for force and torque separately.
        self.fig_s, self.axs_s = plt.subplots(2, 1, figsize=(10, 7))
        self.fig_s.canvas.manager.set_window_title('JF: Feedback share (homing vs assistance)')

        ax = self.axs_s[0]
        ax.set_title('FORCE share (%)', fontsize=10, fontweight='bold')
        ax.set_ylabel('% of |F_home|+|F_guide|'); ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(50, color='k', linestyle=':', linewidth=0.8, alpha=0.5)
        self.l_share_home_F, = ax.plot([], [], color='#2980b9', linewidth=1.8, label='homing %')
        self.l_share_guide_F, = ax.plot([], [], color='#e67e22', linewidth=1.8, label='assistance %')
        ax.legend(loc='upper left', fontsize=8, ncol=2)

        ax = self.axs_s[1]
        ax.set_title('TORQUE share (%)', fontsize=10, fontweight='bold')
        ax.set_ylabel('% of |T_home|+|T_guide|'); ax.set_xlabel('Time (s)'); ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(50, color='k', linestyle=':', linewidth=0.8, alpha=0.5)
        self.l_share_home_T, = ax.plot([], [], color='#2980b9', linewidth=1.8, label='homing %')
        self.l_share_guide_T, = ax.plot([], [], color='#e67e22', linewidth=1.8, label='assistance %')
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        self.fig_s.tight_layout()

        plt.show(block=False)

    def update_plot(self):
        with self.plot_lock:
            if len(self.t_data) == 0:
                return
            t = list(self.t_data)
            fh = [list(self.fh_data[i]) for i in range(6)]
            fg = [list(self.fg_data[i]) for i in range(6)]

        n = min([len(t)] + [len(x) for x in fh] + [len(x) for x in fg])
        if n == 0:
            return
        t = t[:n]
        fh_a = np.array([x[:n] for x in fh])   # (6, n): home wrench components
        fg_a = np.array([x[:n] for x in fg])   # (6, n): guide wrench components

        # Per-channel magnitudes (force = axes 0..2, torque = axes 3..5).
        home_F = np.linalg.norm(fh_a[0:3], axis=0)
        guide_F = np.linalg.norm(fg_a[0:3], axis=0)
        home_T = np.linalg.norm(fh_a[3:6], axis=0)
        guide_T = np.linalg.norm(fg_a[3:6], axis=0)

        # Shares (%). Masked so the 0/0 at rest (both ~0) stays 0 without a warning.
        denom_F = home_F + guide_F
        denom_T = home_T + guide_T
        share_home_F = np.divide(100.0 * home_F, denom_F,
                                 out=np.zeros_like(home_F), where=denom_F > 1e-9)
        share_guide_F = np.divide(100.0 * guide_F, denom_F,
                                  out=np.zeros_like(guide_F), where=denom_F > 1e-9)
        share_home_T = np.divide(100.0 * home_T, denom_T,
                                 out=np.zeros_like(home_T), where=denom_T > 1e-9)
        share_guide_T = np.divide(100.0 * guide_T, denom_T,
                                  out=np.zeros_like(guide_T), where=denom_T > 1e-9)

        win = (t[-1] - self.plot_window_sec, t[-1])

        # Contributions window.
        self.l_home_F.set_data(t, home_F)
        self.l_guide_F.set_data(t, guide_F)
        self.l_home_T.set_data(t, home_T)
        self.l_guide_T.set_data(t, guide_T)
        for ax in self.axs_c:
            ax.set_xlim(*win)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        # Share window.
        self.l_share_home_F.set_data(t, share_home_F)
        self.l_share_guide_F.set_data(t, share_guide_F)
        self.l_share_home_T.set_data(t, share_home_T)
        self.l_share_guide_T.set_data(t, share_guide_T)
        for ax in self.axs_s:
            ax.set_xlim(*win)

        self.fig_c.canvas.draw_idle()
        self.fig_s.canvas.draw_idle()
        self.fig_c.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = HapticForceManagerJF()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True, name='rclpy-spin')
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
