#!/usr/bin/env python3
"""Haptic Force Manager -- JOYSTICK FULL GUIDANCE (JFB: Feedback + Blending).

The full-guidance cell of the velocity-control column: BOTH assistance channels
active on the spring-centered joystick. It renders the SUPERPOSITION of

  1. F_home  -- the restorative centering spring toward the (dynamic) home pose
                (identical to the JB manager), and
  2. F_guide -- the belief-weighted velocity-field guidance force, copied VERBATIM
                (structure) from the CFB manager (compute_F_guide), only RETUNED so
                that it is defined RELATIVE TO the home spring rather than by
                hand-picked gains.

Design of the F_guide / F_home overlap (this is the whole point of JFB):
  The guidance saturates at the DEADZONE-EXIT force -- the force that, against the
  centering spring, displaces the handle exactly to the edge of the joystick
  deadband:
        F_exit_lin = KP_LIN * DEADBAND_LIN ,   Tau_exit_ang = KP_ANG * DEADBAND_ANG
  MAX_GUIDE_{FORCE,TORQUE} = GUIDE_K * F_exit, and the guidance is scaled LINEARLY
  by  gain = confidence(b_max) x proximity(ref->goal). With GUIDE_K < 1 (currently
  0.55) the guidance is a pure BIAS -- it can no longer clear the deadband on its
  own, even at gain=1:
    * gain = 1  (confident AND near the goal): the handle is pushed toward the goal
      but stays INSIDE the deadband -> a clear "preferred direction": the operator
      clears the deadband easily going WITH the guidance (toward the goal), and to
      oppose it they must overcome spring + guidance. The operator still drives.
    * gain in (0,1): the same bias, weaker.
    * gain = 0 (unsure, or far): pure joystick, no bias.
  (GUIDE_K >= 1 would let the guidance clear the deadband autonomously; it was
  reduced to 0.55 because full-authority guidance felt like constantly fighting
  forces rather than being gently biased.)
  NOTE: unlike CFB, F_guide here is FEED-FORWARD (built from v_field only, no
  -handle_vel term). CFB's velocity-field self-damping is a virtual damper whose
  coefficient, at the value needed to reach the exit force on this device, exceeds
  the 150 Hz impedance-device passivity limit and excites a hard high-frequency
  limit cycle. The feed-forward force still vanishes at the goal (v_field -> 0);
  handle damping is provided by the centering spring's KD + device friction.

Final wrench (Haption base frame): F_home + F_guide, clipped to +/- MAX_FORCE /
MAX_TORQUE, published on virtuose/force_cmd.
"""

import threading
import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench, Twist, Pose
from std_msgs.msg import Float64MultiArray, String

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Single source of truth for the joystick home pose, spring gains, deadbands and
# the experiment-condition selector.
import triago_control.qp_controller.config as cfg


class HapticForceManagerJFB(Node):
    def __init__(self):
        super().__init__('haptic_force_manager_jfb')

        # Fail loudly if launched under the wrong study condition. This is the
        # JOYSTICK "Full guidance" manager: assistive feedback (channel F) ON AND
        # reference blending (channel B) ON. The handle renders the centering
        # spring + the belief-weighted guidance force.
        cfg.validate_condition('haptic_force_manager_JFB',
                               control_mode=cfg.JOYSTICK, feedback=True, blending=True)

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
        self.pos_target = None        # the (blended) reference the QP tracks
        self.active_arm = 'right'

        # --- Guidance tuning, DEFINED RELATIVE TO THE HOME SPRING ---------------
        # Saturation = deadzone-exit force x GUIDE_K. GUIDE_K < 1 -> the guidance
        # BIASES the handle toward the goal but does NOT clear the deadband on its
        # own (even at gain=1); the operator feels a preferred direction yet still
        # drives. Reduced 1.1 -> 0.55 (50%) because full-authority guidance felt
        # like fighting forces rather than being gently biased.
        self.GUIDE_K = 0.55
        self.MAX_GUIDE_FORCE  = self.GUIDE_K * self.KP_LIN * cfg.JOYSTICK_DEADBAND_LIN   # ~2.85 N
        self.MAX_GUIDE_TORQUE = self.GUIDE_K * self.KP_ANG * cfg.JOYSTICK_DEADBAND_ANG   # ~0.21 Nm
        # FEED-FORWARD magnitude-shaping gains (policy speed -> force), NOT velocity
        # feedback: F_guide is built from v_field ONLY (no -handle_vel term), so
        # these are not virtual dampers and don't threaten device passivity. Set
        # HIGH so the tanh reaches the (exit-force) saturation for any meaningful
        # policy speed (~cm/s), fading only in the final approach where v_field->0.
        self.D_guide_lin = 400.0   # N per (m/s) of policy speed (feed-forward shaping)
        self.D_guide_ang = 30.0    # Nm per (rad/s) of policy speed (feed-forward shaping)
        self.alpha_guide = 0.15    # LPF on the guidance wrench (C0 continuity)
        self.f_guide_filtered = np.zeros(6)
        # gain = confidence(b_max) x proximity(ref->goal):
        self.GUIDE_CONF_LO = 0.60   # b_max below this -> guidance dead
        self.GUIDE_CONF_HI = 0.90   # b_max at/above this -> full confidence gate
        self.GUIDE_PROX_NEAR = 0.10  # m: ref->goal distance at/below this -> full proximity gate
        self.GUIDE_PROX_FAR  = 0.30  # m: at/beyond this -> guidance dead

        # --- Out-of-deadzone vibration cue (same as J / JB) ---
        # A constant, low-amplitude, zero-mean buzz on the three torque axes,
        # rendered the WHOLE time the handle sits OUTSIDE the joystick deadband --
        # i.e. exactly while a non-zero twist is being commanded. It makes the
        # operator always AWARE of WHEN they are impressing a command (whether from
        # their own push or from the guidance biasing the handle past the deadband).
        self.VIB_AMP = 0.05           # Nm  constant torque amplitude while outside the deadband
        self.vib_toggle = 1.0         # per-frame sign toggle (~75 Hz square wave)

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
        # The (blended) reference the QP tracks -> proximity distance ref->goal.
        self.create_subscription(Float64MultiArray, '/arm_right/cartesian_reference', self.target_cb, 10)
        self.create_subscription(Float64MultiArray, '/arm_left/cartesian_reference', self.target_cb_left, 10)
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)

        # --- Publisher ---
        self.force_pub = self.create_publisher(Wrench, 'virtuose/force_cmd', 10)

        # --- Plot buffers (home + guide wrench components; total/norms derived) ---
        self.plot_lock = threading.Lock()
        self.plot_window_sec = 10.0
        self.buffer_size = int(150 * self.plot_window_sec)
        self.start_time = time.time()
        self.t_data = deque(maxlen=self.buffer_size)
        self.fh_data = [deque(maxlen=self.buffer_size) for _ in range(6)]  # home wrench
        self.fg_data = [deque(maxlen=self.buffer_size) for _ in range(6)]  # guide wrench
        self.pos_err_data = deque(maxlen=self.buffer_size)  # ||home_pos - handle_pos|| (deadzone plot)
        self.ang_err_data = deque(maxlen=self.buffer_size)  # geodesic home<->handle    (deadzone plot)

        self.dt = 1.0 / 150.0
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.setup_plot()
        self.get_logger().info(
            f"[HFM-JFB] Joystick FULL-guidance manager started. "
            f"F_home spring KP_LIN={self.KP_LIN}, KP_ANG={self.KP_ANG}; "
            f"F_guide saturates at exit force MAX_GUIDE_FORCE={self.MAX_GUIDE_FORCE:.2f} N / "
            f"MAX_GUIDE_TORQUE={self.MAX_GUIDE_TORQUE:.3f} Nm (GUIDE_K={self.GUIDE_K}). "
            f"gain = conf[{self.GUIDE_CONF_LO},{self.GUIDE_CONF_HI}] x "
            f"prox[{self.GUIDE_PROX_NEAR},{self.GUIDE_PROX_FAR}] m.")

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
            self.get_logger().info(f"[HFM-JFB] Active arm switched to {msg.data.upper()}")

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
        """Velocity-field guidance (copied from CFB.compute_F_guide), retuned to
        overlap the home spring.

        Structure:
          1. pi_blend = Sum_k P(k) * pi_k          (belief-weighted policy twist)
          2. v_field  = map_180Z(pi_blend)         (robot -> Haption frame)
          3. FEED-FORWARD force F = MAX * tanh(D * v_field / MAX) (direction from
             the policy field, magnitude saturated at the exit force), tanh-sat, LPF'd.

        Retuned vs CFB (per the JFB plan):
          * FEED-FORWARD (v_field only, NO -handle_vel term): CFB's velocity-field
            self-damping is a virtual damper too strong for the 150 Hz device
            passivity limit and caused a hard limit cycle -- removed here (handle
            damping comes from the spring's KD + device friction);
          * saturation MAX_GUIDE_{FORCE,TORQUE} = deadzone-exit force (not free gains);
          * gain applied AFTER the tanh (linear 'fraction of the exit force'), so
            gain=1 exits the deadband and gain<1 only biases inside it;
          * confidence gate uses b_max (active-goal belief) not the entropy measure;
          * proximity gate ramps over [GUIDE_PROX_NEAR, GUIDE_PROX_FAR] = [0.10, 0.30] m.
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

        # Confidence gate: the ACTIVE-goal belief b_max (so "0.6" reads as "60% sure"),
        # NOT the entropy measure CFB uses. Dead below LO, full at/above HI.
        conf_gate = self._smoothstep(self.fix_confidence,
                                     lo=self.GUIDE_CONF_LO, hi=self.GUIDE_CONF_HI)

        # Proximity gate: distance from the (blended) REFERENCE to the active goal.
        # Full at/below NEAR, dead at/beyond FAR (smoothstep between). Same formula
        # as CFB, only the FAR threshold is tightened (1.0 -> 0.30 m).
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

        # FEED-FORWARD guidance (NOT the CFB velocity field): shape the magnitude
        # from the (smooth, LPF'd) policy field v_field ONLY -- do NOT feed the
        # handle velocity back. CFB's `-D*handle_vel` term is a VIRTUAL DAMPER of
        # coefficient D; at the D needed here to reach the exit force (~400 Ns/m)
        # it is ~100x above what a 150 Hz impedance device can render passively,
        # which excited a hard high-frequency limit cycle. Dropping it makes
        # F_guide a smooth goal-directed force that merely shifts the centering
        # spring's equilibrium (stable); the handle is damped by the spring's own
        # KD + device friction (exactly like the JB manager). Direction from
        # v_field, magnitude saturated at the exit force, then linearly gain-scaled.
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

        # --- Out-of-deadzone vibration cue (same as J / JB) ---
        # Buzz whenever the handle displacement from home exceeds EITHER deadband,
        # i.e. exactly when the teleop is commanding a non-zero twist -- so the
        # operator always feels WHEN a command is being impressed.
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

        f_total[0:3] = np.clip(f_total[0:3], -self.MAX_FORCE, self.MAX_FORCE)
        f_total[3:6] = np.clip(f_total[3:6], -self.MAX_TORQUE, self.MAX_TORQUE)

        msg = Wrench()
        msg.force.x, msg.force.y, msg.force.z = float(f_total[0]), float(f_total[1]), float(f_total[2])
        msg.torque.x, msg.torque.y, msg.torque.z = float(f_total[3]), float(f_total[4]), float(f_total[5])
        self.force_pub.publish(msg)

        t = time.time() - self.start_time
        pos_err = (float(np.linalg.norm(self.home_pos - self.handle_pos))
                   if self.handle_pos is not None else 0.0)
        ang_err = (float(np.linalg.norm((self.home_rot * self.handle_rot.inv()).as_rotvec()))
                   if self.handle_rot is not None else 0.0)
        with self.plot_lock:
            self.t_data.append(t)
            for i in range(6):
                self.fh_data[i].append(f_home[i])
                self.fg_data[i].append(f_guide[i])
            self.pos_err_data.append(pos_err)
            self.ang_err_data.append(ang_err)

    # ------------------------------------------------------------------ plotting
    def setup_plot(self):
        plt.ion()
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']

        # --- Window 1: DEADZONE condition (same clarity as the JB manager) ------
        self.fig_dz, self.ax_dz = plt.subplots(1, 1, figsize=(10, 4))
        self.fig_dz.canvas.manager.set_window_title('JFB: Handle displacement from home (deadzone)')
        ax = self.ax_dz
        ax.set_title('Displacement from home vs deadband', fontsize=10, fontweight='bold')
        ax.set_ylabel('m  /  rad'); ax.set_xlabel('Time (s)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(cfg.JOYSTICK_DEADBAND_LIN, color='r', linestyle=':', linewidth=1.2,
                   alpha=0.8, label=f'lin deadband = {cfg.JOYSTICK_DEADBAND_LIN} m')
        ax.axhline(cfg.JOYSTICK_DEADBAND_ANG, color='b', linestyle=':', linewidth=1.2,
                   alpha=0.8, label=f'ang deadband = {cfg.JOYSTICK_DEADBAND_ANG:.3f} rad')
        self.line_pos_err, = ax.plot([], [], color='#e67e22', linewidth=1.6, label='||pos - home|| (m)')
        self.line_ang_err, = ax.plot([], [], color='#9b59b6', linewidth=1.6, label='ang gap (rad)')
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        self.fig_dz.tight_layout()

        # --- Window 2: GUIDANCE SHARE of the wrench (home share = 100% - this) --
        # Per-axis fraction of |wrench| contributed by F_guide vs F_home. 3 lines
        # per subplot (X/Y/Z). The home share is exactly 100% minus each line.
        self.fig_sh, self.axs_sh = plt.subplots(2, 1, figsize=(10, 7))
        self.fig_sh.canvas.manager.set_window_title('JFB: Guidance share of the wrench')
        ax = self.axs_sh[0]
        ax.set_title('Guidance FORCE share per axis (%)  [home = 100% - this]', fontsize=10, fontweight='bold')
        ax.set_ylabel('% of |F| on axis'); ax.set_ylim(0, 100); ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(50, color='k', linestyle=':', linewidth=0.8, alpha=0.5)
        self.l_shF = [ax.plot([], [], color=colors[i], linewidth=1.6, label=f'guide F{labels[i]}')[0] for i in range(3)]
        ax.legend(loc='upper left', fontsize=8, ncol=3)
        ax = self.axs_sh[1]
        ax.set_title('Guidance TORQUE share per axis (%)  [home = 100% - this]', fontsize=10, fontweight='bold')
        ax.set_ylabel('% of |T| on axis'); ax.set_xlabel('Time (s)'); ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(50, color='k', linestyle=':', linewidth=0.8, alpha=0.5)
        self.l_shT = [ax.plot([], [], color=colors[i], linewidth=1.6, label=f'guide T{labels[i]}')[0] for i in range(3)]
        ax.legend(loc='upper left', fontsize=8, ncol=3)
        self.fig_sh.tight_layout()

        plt.show(block=False)

    def update_plot(self):
        with self.plot_lock:
            if len(self.t_data) == 0:
                return
            t = list(self.t_data)
            fh = [list(self.fh_data[i]) for i in range(6)]
            fg = [list(self.fg_data[i]) for i in range(6)]
            pos_err = list(self.pos_err_data)
            ang_err = list(self.ang_err_data)

        n = min([len(t)] + [len(x) for x in fh] + [len(x) for x in fg]
                + [len(pos_err), len(ang_err)])
        if n == 0:
            return
        t = t[:n]
        fh_a = np.abs(np.array([x[:n] for x in fh]))   # (6, n): |home wrench|
        fg_a = np.abs(np.array([x[:n] for x in fg]))   # (6, n): |guide wrench|
        denom = fh_a + fg_a
        # Guidance share per axis (%): |F_guide| / (|F_guide| + |F_home|). Use
        # np.divide with a mask so the division is ONLY evaluated where denom>0 --
        # this avoids the harmless 0/0 (both forces ~0, e.g. at home with guidance
        # off) that np.where would still compute and warn about. Masked-out
        # entries stay 0 (pre-filled by out=).
        share = np.divide(100.0 * fg_a, denom,
                          out=np.zeros_like(fg_a), where=denom > 1e-9)   # (6, n)

        win = (t[-1] - self.plot_window_sec, t[-1])

        # Deadzone window.
        self.line_pos_err.set_data(t, pos_err[:n])
        self.line_ang_err.set_data(t, ang_err[:n])
        self.ax_dz.set_xlim(*win)
        self.ax_dz.relim()
        self.ax_dz.autoscale_view(scalex=False, scaley=True)

        # Share windows (force = axes 0..2, torque = axes 3..5).
        for i in range(3):
            self.l_shF[i].set_data(t, share[i])
            self.l_shT[i].set_data(t, share[i + 3])
        for ax in self.axs_sh:
            ax.set_xlim(*win)

        self.fig_dz.canvas.draw_idle()
        self.fig_sh.canvas.draw_idle()
        self.fig_dz.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = HapticForceManagerJFB()

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
