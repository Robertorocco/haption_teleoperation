#!/usr/bin/env python3
"""JOYSTICK full guidance (F=1, B=1): centering spring + guidance bias calibrated to the deadzone-exit force."""

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

# Cross-package condition selector + joystick home pose, spring gains and deadbands.
import triago_control.qp_controller.config as cfg


class HapticForceManagerJFB(Node):
    # Full-guidance joystick manager: F_home + feed-forward F_guide, blending handled robot-side.
    def __init__(self):
        super().__init__('haptic_force_manager_jfb')

        # Hard-error at startup unless config.py selects the JOYSTICK full-guidance cell.
        cfg.validate_condition('haptic_force_manager_JFB',
                               control_mode=cfg.JOYSTICK, feedback=True, blending=True)

        # Home pose (Haption base frame), neutral until the teleop broadcasts the live home.
        self.home_pos = np.array(cfg.JOYSTICK_NEUTRAL_POSITION_M, dtype=float)
        self.home_rot = R.from_quat(cfg.JOYSTICK_NEUTRAL_ORIENTATION_XYZW)  # xyzw

        self.handle_pos = None
        self.handle_rot = None
        self.handle_vel = np.zeros(6)

        # Spring gains, unified across all joystick cells.
        self.KP_LIN = cfg.JOYSTICK_SPRING_KP_LIN
        self.KD_LIN = cfg.JOYSTICK_SPRING_KD_LIN
        self.KP_ANG = cfg.JOYSTICK_SPRING_KP_ANG
        self.KD_ANG = cfg.JOYSTICK_SPRING_KD_ANG

        # Device safety clip and unified authority cap (currently equal).
        self.MAX_FORCE = 10.0
        self.MAX_TORQUE = 1.0
        self.MAX_TOTAL_FORCE = 10.0
        self.MAX_TOTAL_TORQUE = 1.0

        # Guidance inference inputs from shared autonomy.
        self.goal_names = []
        self.goal_probs = []
        self.user_policies = []
        self.fix_goal_pos = None      # active goal position (base frame)
        self.fix_confidence = 0.0     # b_max: active-goal belief (0 during grasp exec)
        self.pos_target = None        # the (blended) reference the QP tracks
        self.active_arm = 'right'

        # Guidance saturation = GUIDE_K x deadzone-exit force; GUIDE_K < 1 makes it a pure
        # bias that can never clear the deadband alone -- the operator always initiates motion.
        self.GUIDE_K = 0.55
        self.MAX_GUIDE_FORCE  = self.GUIDE_K * self.KP_LIN * cfg.JOYSTICK_DEADBAND_LIN
        self.MAX_GUIDE_TORQUE = self.GUIDE_K * self.KP_ANG * cfg.JOYSTICK_DEADBAND_ANG
        # Feed-forward magnitude-shaping gains (policy speed -> force), not velocity feedback:
        # high values push the tanh into saturation at any meaningful policy speed.
        self.D_guide_lin = 400.0   # N per (m/s)
        self.D_guide_ang = 30.0    # Nm per (rad/s)
        self.alpha_guide = 0.15    # LPF on the guidance wrench (C0 continuity)
        self.f_guide_filtered = np.zeros(6)
        # gain = confidence(b_max) x proximity(ref->goal), unified across all guidance cells.
        self.GUIDE_CONF_LO = 0.30   # below: guidance dead
        self.GUIDE_CONF_HI = 0.90   # at/above: full confidence gate
        self.GUIDE_PROX_NEAR = 0.10  # m: full proximity gate at/below
        self.GUIDE_PROX_FAR  = 0.60  # m: guidance dead at/beyond

        # Out-of-deadzone cue: zero-mean buzz whenever a non-zero twist is being commanded.
        self.VIB_AMP = 0.05           # Nm
        self.vib_toggle = 1.0         # sign flip every frame -> ~75 Hz square wave

        # Autonomous-grasp cue, unified across all 8 cells.
        self.grasp_active = False
        self.GRASP_VIB_AMP = 0.07    # Nm
        self.grasp_vib_toggle = 1.0

        # virtuose/pose is geometry_msgs/Pose (not PoseStamped).
        self.create_subscription(Pose, 'virtuose/pose', self.handle_pose_cb, 10)
        self.create_subscription(Twist, 'virtuose/velocity', self.vel_cb, 10)
        self.create_subscription(
            Float64MultiArray, cfg.JOYSTICK_HOME_POSE_TOPIC, self.home_pose_cb, 10)
        # Guidance inference state (belief-weighted policy field + active goal).
        self.create_subscription(String, '/shared_autonomy/goal_names', self.goal_names_cb, 10)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/goal_probabilities', self.goal_probs_cb, 10)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/user_policy', self.user_policy_cb, 10)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/active_goal_pose', self.goal_pose_cb, 10)
        # The (blended) reference the QP tracks feeds the ref->goal proximity gate.
        self.create_subscription(Float64MultiArray, '/arm_right/cartesian_reference', self.target_cb, 10)
        self.create_subscription(Float64MultiArray, '/arm_left/cartesian_reference', self.target_cb_left, 10)
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)
        # Blend telemetry: [alpha, v_user(6), v_policy(6), v_blend(6)] = 19 floats.
        self.create_subscription(
            Float64MultiArray, '/shared_autonomy/blend_debug', self.blend_debug_cb, 10)
        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_cb, 10)

        self.force_pub = self.create_publisher(Wrench, 'virtuose/force_cmd', 10)

        # Plot buffers (10 s window at 150 Hz), guarded by a lock shared with the UI thread.
        self.plot_lock = threading.Lock()
        self.plot_window_sec = 10.0
        self.buffer_size = int(150 * self.plot_window_sec)
        self.start_time = time.time()
        self.t_data = deque(maxlen=self.buffer_size)
        self.fh_data = [deque(maxlen=self.buffer_size) for _ in range(6)]  # home wrench
        self.fg_data = [deque(maxlen=self.buffer_size) for _ in range(6)]  # guide wrench
        self.pos_err_data = deque(maxlen=self.buffer_size)  # ||home_pos - handle_pos||
        self.ang_err_data = deque(maxlen=self.buffer_size)  # geodesic home<->handle

        # Blend telemetry buffers (alpha + user/policy share), shared design with every B=1 cell.
        self.alpha_data = deque(maxlen=self.buffer_size)
        self.user_pct_data = deque(maxlen=self.buffer_size)
        self.policy_pct_data = deque(maxlen=self.buffer_size)
        self._last_blend_alpha = 0.0
        self._last_blend_user_pct = 0.0
        self._last_blend_policy_pct = 0.0

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
        """Stores the latest handle pose (Haption base frame)."""
        p = msg.position
        q = msg.orientation
        self.handle_pos = np.array([p.x, p.y, p.z])
        self.handle_rot = R.from_quat([q.x, q.y, q.z, q.w])

    def vel_cb(self, msg):
        """Stores the latest handle 6-DOF spatial velocity."""
        self.handle_vel = np.array([
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z])

    def home_pose_cb(self, msg):
        """Updates the live home pose from the teleop node: [pos(3), quat_xyzw(4)]."""
        if len(msg.data) >= 7:
            self.home_pos = np.array(msg.data[0:3])
            self.home_rot = R.from_quat(np.array(msg.data[3:7]))

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
        """Updates the active goal pose + belief b_max (0 during autonomous grasp execution)."""
        if len(msg.data) >= 7:
            self.fix_goal_pos = np.array(msg.data[0:3])
            self.fix_confidence = float(msg.data[6])

    def target_cb(self, msg):
        """Updates the tracked reference position (right arm)."""
        if self.active_arm != 'right':
            return
        if len(msg.data) >= 3:
            self.pos_target = np.array(msg.data[0:3])

    def target_cb_left(self, msg):
        """Updates the tracked reference position (left arm)."""
        if self.active_arm != 'left':
            return
        if len(msg.data) >= 3:
            self.pos_target = np.array(msg.data[0:3])

    def active_arm_cb(self, msg):
        """Switches which arm's reference is used for the proximity gate."""
        if msg.data in ('right', 'left') and msg.data != self.active_arm:
            self.active_arm = msg.data
            self.get_logger().info(f"[HFM-JFB] Active arm switched to {msg.data.upper()}")

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

    # ------------------------------------------------------------------ helpers
    def _smoothstep(self, p, lo, hi):
        """C1-continuous ramp from 0 at p=lo to 1 at p=hi."""
        if p <= lo:
            return 0.0
        x = min((p - lo) / (hi - lo), 1.0)
        return 3.0 * x ** 2 - 2.0 * x ** 3

    # ------------------------------------------------------------------ forces
    def compute_spring(self):
        """Spring-damper wrench (Haption base frame) pulling the handle to the home pose."""
        f = np.zeros(6)
        if self.handle_pos is None or self.handle_rot is None:
            return f
        f[0:3] = self.KP_LIN * (self.home_pos - self.handle_pos) - self.KD_LIN * self.handle_vel[0:3]
        err_rotvec = (self.home_rot * self.handle_rot.inv()).as_rotvec()
        f[3:6] = self.KP_ANG * err_rotvec - self.KD_ANG * self.handle_vel[3:6]
        return f

    def compute_F_guide(self):
        """Feed-forward velocity-field guidance saturated at GUIDE_K x the deadzone-exit force."""
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
        conf_gate = self._smoothstep(self.fix_confidence,
                                     lo=self.GUIDE_CONF_LO, hi=self.GUIDE_CONF_HI)

        # Proximity gate: reference-to-goal distance, silencing guidance while the goal still swings.
        if self.fix_goal_pos is not None and self.pos_target is not None:
            d_goal = float(np.linalg.norm(self.fix_goal_pos - self.pos_target))
            prox = np.clip(
                (self.GUIDE_PROX_FAR - d_goal)
                / max(self.GUIDE_PROX_FAR - self.GUIDE_PROX_NEAR, 1e-6), 0.0, 1.0)
            prox_gate = 3.0 * prox ** 2 - 2.0 * prox ** 3   # smoothstep
        else:
            prox_gate = 0.0   # no goal/reference info -> no guidance

        gain = conf_gate * prox_gate

        # Map the policy twist into the Haption frame (180-deg Z-flip, matching the joystick teleop).
        v_field = np.array([
            -pi_blend[0], -pi_blend[1],  pi_blend[2],
            -pi_blend[3], -pi_blend[4],  pi_blend[5],
        ])

        # Feed-forward: F = MAX*tanh(D*v_field/MAX), gain applied AFTER the tanh so it reads as
        # a linear fraction of the exit force; no handle-velocity feedback (a virtual damper at
        # this gain would exceed the 150 Hz passivity limit). Vanishes at the goal (v_field -> 0).
        F_dir = np.zeros(6)
        F_dir[0:3] = self.MAX_GUIDE_FORCE * np.tanh(self.D_guide_lin * v_field[0:3] / self.MAX_GUIDE_FORCE)
        F_dir[3:6] = self.MAX_GUIDE_TORQUE * np.tanh(self.D_guide_ang * v_field[3:6] / self.MAX_GUIDE_TORQUE)
        F_guide_raw = gain * F_dir

        self.f_guide_filtered = (self.alpha_guide * F_guide_raw
                                 + (1.0 - self.alpha_guide) * self.f_guide_filtered)
        return self.f_guide_filtered.copy()

    # ------------------------------------------------------------------ loop
    def control_loop(self):
        """150 Hz: renders F_home + F_guide, adds the cues, caps, clips, publishes, buffers."""
        f_home = self.compute_spring()
        f_guide = self.compute_F_guide()
        f_total = f_home + f_guide

        # Authority cap: proportional rescale before the vibration cues.
        fn = np.linalg.norm(f_total[0:3])
        if fn > self.MAX_TOTAL_FORCE:
            f_total[0:3] *= self.MAX_TOTAL_FORCE / fn
        tn = np.linalg.norm(f_total[3:6])
        if tn > self.MAX_TOTAL_TORQUE:
            f_total[3:6] *= self.MAX_TOTAL_TORQUE / tn

        # Out-of-deadzone cue: buzz exactly while a command (user push or guidance bias) is impressed.
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

        # Autonomous-grasp cue.
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

        # Buffer telemetry.
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
            self.alpha_data.append(self._last_blend_alpha)
            self.user_pct_data.append(self._last_blend_user_pct)
            self.policy_pct_data.append(self._last_blend_policy_pct)

    # ------------------------------------------------------------------ plotting
    def setup_plot(self):
        """Initializes the live plots: deadzone condition, guidance share, blend telemetry."""
        plt.ion()
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']

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

        # Guidance share: per-axis |F_guide| fraction of the wrench (home share = 100% - this).
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

        # Blend-telemetry window: authority alpha + user/policy share.
        self.fig_bl, self.axs_bl = plt.subplots(2, 1, figsize=(10, 5))
        self.fig_bl.canvas.manager.set_window_title('JFB: Blending telemetry')
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
            t = list(self.t_data)
            fh = [list(self.fh_data[i]) for i in range(6)]
            fg = [list(self.fg_data[i]) for i in range(6)]
            pos_err = list(self.pos_err_data)
            ang_err = list(self.ang_err_data)
            alpha_list = list(self.alpha_data)
            upct_list = list(self.user_pct_data)
            ppct_list = list(self.policy_pct_data)

        n = min([len(t)] + [len(x) for x in fh] + [len(x) for x in fg]
                + [len(pos_err), len(ang_err)])
        if n == 0:
            return
        t = t[:n]
        fh_a = np.abs(np.array([x[:n] for x in fh]))   # (6, n): |home wrench|
        fg_a = np.abs(np.array([x[:n] for x in fg]))   # (6, n): |guide wrench|
        denom = fh_a + fg_a
        # Masked divide: evaluated only where denom>0, so the 0/0 case stays 0 without warnings.
        share = np.divide(100.0 * fg_a, denom,
                          out=np.zeros_like(fg_a), where=denom > 1e-9)   # (6, n)

        win = (t[-1] - self.plot_window_sec, t[-1])

        self.line_pos_err.set_data(t, pos_err[:n])
        self.line_ang_err.set_data(t, ang_err[:n])
        self.ax_dz.set_xlim(*win)
        self.ax_dz.relim()
        self.ax_dz.autoscale_view(scalex=False, scaley=True)

        for i in range(3):
            self.l_shF[i].set_data(t, share[i])
            self.l_shT[i].set_data(t, share[i + 3])
        for ax in self.axs_sh:
            ax.set_xlim(*win)

        na = min(len(t), len(alpha_list))
        self.line_alpha.set_data(t[:na], alpha_list[:na])
        self.axs_bl[0].set_xlim(*win)
        ns = min(len(t), len(upct_list), len(ppct_list))
        self.line_user_pct.set_data(t[:ns], upct_list[:ns])
        self.line_policy_pct.set_data(t[:ns], ppct_list[:ns])
        self.axs_bl[1].set_xlim(*win)

        self.fig_dz.canvas.draw_idle()
        self.fig_sh.canvas.draw_idle()
        self.fig_bl.canvas.draw_idle()
        self.fig_dz.canvas.flush_events()


def main(args=None):
    """Spins ROS on a daemon thread and drives Matplotlib on the main thread."""
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
