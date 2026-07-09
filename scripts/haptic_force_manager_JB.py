#!/usr/bin/env python3
"""Haptic Force Manager -- JOYSTICK MODE (BLENDING).

In Joystick Mode the handle is a spring-centered joystick: teleop_triago_joystick.py
reads its DISPLACEMENT from a home pose and maps that to a Cartesian twist. This
node renders the ONE and ONLY haptic force of that mode -- the restorative
centering spring that constantly pulls the handle back to its home configuration.

There is deliberately NO F_sync / F_guide / F_fixture / F_cbf, no clutch-align, no
joint-limit vibration, and no grasp-follow force. Coupling any robot-state-derived
force back onto the handle is exactly what made the previous BLENDING design
unstable (the force moved the handle, the handle motion was read as user intent,
and the loop diverged). The spring targets ONLY the home pose, which is a function
of the (fixed) neutral position and the gripper's orientation -- never of tracking
error -- so it cannot form that loop.

Home pose:
  - Position: fixed, cfg.JOYSTICK_NEUTRAL_POSITION_M = [0.5, -0.03, -0.03].
  - Orientation: dynamic, tracks the gripper (see teleop_triago_joystick.py), starting
    from cfg.JOYSTICK_NEUTRAL_ORIENTATION_XYZW = [0.0, 0.7071068, 0.0, 0.7071068].
  - The live home pose is OWNED and published by teleop_triago_joystick.py on
    cfg.JOYSTICK_HOME_POSE_TOPIC (single source of truth). This node subscribes to
    it and falls back to the neutral constants above until the first message.

Force (Haption base frame, spring-damper toward home):
  F_lin = KP_LIN * (home_pos - handle_pos) - KD_LIN * handle_vel_lin
  Tau   = KP_ANG * rotvec(home_rot * handle_rot^-1) - KD_ANG * handle_vel_ang
clipped to +/- MAX_FORCE / MAX_TORQUE and published on virtuose/force_cmd.
"""

import threading
import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench, Twist, Pose
from std_msgs.msg import Float64MultiArray, Bool

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Single source of truth for the joystick home pose + spring gains.
import triago_control.qp_controller.config as cfg


class HapticForceManagerBlending(Node):
    def __init__(self):
        super().__init__('haptic_force_manager_blending')

        # Fail loudly if launched under the wrong study condition. This is the
        # JOYSTICK "Guided blending" manager: NO assistive feedback (channel F off),
        # reference blending ON (channel B). The handle renders only the restorative
        # centering spring; assistance comes from main_shared_autonomy's alpha blend.
        cfg.validate_condition('haptic_force_manager_JB',
                               control_mode=cfg.JOYSTICK, feedback=False, blending=True)

        # --- Home pose (Haption base frame) -- neutral until teleop publishes ---
        self.home_pos = np.array(cfg.JOYSTICK_NEUTRAL_POSITION_M, dtype=float)
        self.home_rot = R.from_quat(cfg.JOYSTICK_NEUTRAL_ORIENTATION_XYZW)  # xyzw

        # --- Live handle state (Haption base frame) ---
        self.handle_pos = None
        self.handle_rot = None
        self.handle_vel = np.zeros(6)

        # --- Spring gains ---
        self.KP_LIN = cfg.JOYSTICK_SPRING_KP_LIN
        self.KD_LIN = cfg.JOYSTICK_SPRING_KD_LIN
        self.KP_ANG = cfg.JOYSTICK_SPRING_KP_ANG
        self.KD_ANG = cfg.JOYSTICK_SPRING_KD_ANG

        # --- Global force limits ---
        self.MAX_FORCE = 10.0
        self.MAX_TORQUE = 1.0

        # --- Out-of-deadzone vibration cue ---
        # A constant, low-amplitude buzz on the three torque axes rendered the
        # WHOLE time the handle sits outside the joystick deadband (i.e. exactly
        # while the teleop is commanding a non-zero twist). It tells the operator
        # "you are actively driving". Zero-mean + high-frequency, so it does not
        # bias the displacement the teleop reads back.
        self.VIB_AMP = 0.05           # Nm  constant torque amplitude while outside the deadband
        self.vib_toggle = 1.0         # per-frame sign toggle (~75 Hz square wave)

        # --- Grasp-execution vibration cue (UNIFIED with clutch cells) ---
        self.grasp_active = False
        self.GRASP_VIB_AMP = 0.07    # Nm  constant buzz during autonomous grasp
        self.grasp_vib_toggle = 1.0

        # --- Subscribers ---
        # NOTE: virtuose_server_node publishes virtuose/pose as geometry_msgs/Pose
        # (NOT PoseStamped) -- subscribing with the wrong type silently receives
        # nothing, which zeroes the spring (handle_pos stays None).
        self.create_subscription(Pose, 'virtuose/pose', self.handle_pose_cb, 10)
        self.create_subscription(Twist, 'virtuose/velocity', self.vel_cb, 10)
        self.create_subscription(
            Float64MultiArray, cfg.JOYSTICK_HOME_POSE_TOPIC, self.home_pose_cb, 10)
        # Blending telemetry: [alpha, v_user(6), v_policy(6), v_blend(6)] = 19 floats.
        self.create_subscription(
            Float64MultiArray, '/shared_autonomy/blend_debug', self.blend_debug_cb, 10)
        # Grasp-execution flag: vibrate during autonomous grasp.
        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_cb, 10)

        # --- Publisher ---
        self.force_pub = self.create_publisher(Wrench, 'virtuose/force_cmd', 10)

        # --- Plot buffers ---
        self.plot_lock = threading.Lock()
        self.plot_window_sec = 10.0
        self.buffer_size = int(150 * self.plot_window_sec)
        self.start_time = time.time()
        self.t_data = deque(maxlen=self.buffer_size)
        self.force_data = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.torque_data = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.pos_err_data = deque(maxlen=self.buffer_size)   # ||home_pos - handle_pos||
        self.ang_err_data = deque(maxlen=self.buffer_size)   # geodesic home<->handle
        self._freq_data = deque(maxlen=self.buffer_size)
        self._last_time = None
        self._freq_lpf = 0.0

        # --- Blending telemetry (from /shared_autonomy/blend_debug), identical to
        #     the CB manager so every B=1 cell reports the same alpha + share. ---
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
            f"[HFM-JOYSTICK] Restorative-spring haptic manager started. "
            f"Home position (Haption) = {self.home_pos.tolist()}, "
            f"KP_LIN={self.KP_LIN}, KP_ANG={self.KP_ANG}. "
            f"Listening for the live home pose on '{cfg.JOYSTICK_HOME_POSE_TOPIC}'.")

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

    def grasp_active_cb(self, msg):
        """Tracks whether the shared-autonomy node is autonomously driving a grasp."""
        self.grasp_active = bool(msg.data)

    def blend_debug_cb(self, msg):
        """Process /shared_autonomy/blend_debug: [alpha, v_user(6), v_policy(6), v_blend(6)]."""
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

    # ------------------------------------------------------------------ force
    def compute_spring(self):
        """Spring-damper wrench (Haption base frame) pulling the handle to home."""
        f = np.zeros(6)
        if self.handle_pos is None or self.handle_rot is None:
            return f

        # Linear spring toward the home position.
        f[0:3] = self.KP_LIN * (self.home_pos - self.handle_pos) - self.KD_LIN * self.handle_vel[0:3]

        # Angular spring toward the home orientation (both already in the Haption
        # frame, so no frame mapping is needed here).
        err_rotvec = (self.home_rot * self.handle_rot.inv()).as_rotvec()
        f[3:6] = self.KP_ANG * err_rotvec - self.KD_ANG * self.handle_vel[3:6]
        return f

    def control_loop(self):
        f = self.compute_spring()

        # --- Out-of-deadzone vibration cue ---
        # Buzz whenever the handle displacement from home exceeds EITHER the
        # linear OR the angular deadband (matches the teleop's radial deadband on
        # each channel, so the buzz starts exactly when a non-zero twist is sent).
        if self.handle_pos is not None and self.handle_rot is not None:
            lin_disp = float(np.linalg.norm(self.home_pos - self.handle_pos))
            ang_disp = float(np.linalg.norm((self.home_rot * self.handle_rot.inv()).as_rotvec()))
            if (lin_disp > cfg.JOYSTICK_DEADBAND_LIN
                    or ang_disp > cfg.JOYSTICK_DEADBAND_ANG):
                self.vib_toggle *= -1.0
                buzz = self.VIB_AMP * self.vib_toggle
                f[3] += buzz
                f[4] += buzz
                f[5] += buzz

        # --- Grasp vibration cue (0.07 Nm buzz during autonomous grasp) ---
        if self.grasp_active:
            self.grasp_vib_toggle *= -1.0
            gb = self.GRASP_VIB_AMP * self.grasp_vib_toggle
            f[3] += gb
            f[4] += gb
            f[5] += gb

        f[0:3] = np.clip(f[0:3], -self.MAX_FORCE, self.MAX_FORCE)
        f[3:6] = np.clip(f[3:6], -self.MAX_TORQUE, self.MAX_TORQUE)

        msg = Wrench()
        msg.force.x, msg.force.y, msg.force.z = float(f[0]), float(f[1]), float(f[2])
        msg.torque.x, msg.torque.y, msg.torque.z = float(f[3]), float(f[4]), float(f[5])
        self.force_pub.publish(msg)

        # --- Buffer telemetry ---
        t = time.time() - self.start_time
        pos_err = (float(np.linalg.norm(self.home_pos - self.handle_pos))
                   if self.handle_pos is not None else 0.0)
        ang_err = (float(np.linalg.norm((self.home_rot * self.handle_rot.inv()).as_rotvec()))
                   if self.handle_rot is not None else 0.0)
        with self.plot_lock:
            self.t_data.append(t)
            for i in range(3):
                self.force_data[i].append(f[i])
                self.torque_data[i].append(f[i + 3])
            self.pos_err_data.append(pos_err)
            self.ang_err_data.append(ang_err)
            now = time.time()
            if self._last_time is not None:
                d = now - self._last_time
                if d > 1e-6:
                    self._freq_lpf = 0.9 * self._freq_lpf + 0.1 * (1.0 / d)
            self._last_time = now
            self._freq_data.append(self._freq_lpf)
            self.alpha_data.append(self._last_blend_alpha)
            self.user_pct_data.append(self._last_blend_user_pct)
            self.policy_pct_data.append(self._last_blend_policy_pct)

    # ------------------------------------------------------------------ plotting
    def setup_plot(self):
        plt.ion()
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']

        # Window 1: spring force + torque.
        self.fig1, self.axs1 = plt.subplots(2, 1, figsize=(10, 5))
        self.fig1.canvas.manager.set_window_title('Joystick Restorative Spring')
        ax = self.axs1[0]
        ax.set_title("Spring FORCE (N)", fontsize=10, fontweight='bold')
        ax.set_ylabel("N")
        ax.grid(True, linestyle='--', alpha=0.6)
        self.lines_F = [ax.plot([], [], color=colors[i], label=f"F{labels[i]}")[0] for i in range(3)]
        ax.legend(loc='upper left', fontsize=8, ncol=3)
        ax = self.axs1[1]
        ax.set_title("Spring TORQUE (Nm)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Nm")
        ax.set_xlabel("Time (s)")
        ax.grid(True, linestyle='--', alpha=0.6)
        self.lines_T = [ax.plot([], [], color=colors[i], label=f"T{labels[i]}")[0] for i in range(3)]
        ax.legend(loc='upper left', fontsize=8, ncol=3)
        self.fig1.tight_layout()

        # Window 2: displacement from home + node frequency.
        self.fig2, self.axs2 = plt.subplots(2, 1, figsize=(10, 5))
        self.fig2.canvas.manager.set_window_title('Handle Displacement From Home')
        ax = self.axs2[0]
        ax.set_title("Displacement from home", fontsize=10, fontweight='bold')
        ax.set_ylabel("m  /  rad")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(cfg.JOYSTICK_DEADBAND_LIN, color='r', linestyle=':', linewidth=1.0,
                   alpha=0.7, label=f'lin deadband={cfg.JOYSTICK_DEADBAND_LIN} m')
        ax.axhline(cfg.JOYSTICK_DEADBAND_ANG, color='b', linestyle=':', linewidth=1.0,
                   alpha=0.7, label=f'ang deadband={cfg.JOYSTICK_DEADBAND_ANG:.3f} rad')
        self.line_pos_err, = ax.plot([], [], color='#e67e22', linewidth=1.6, label='||pos - home|| (m)')
        self.line_ang_err, = ax.plot([], [], color='#9b59b6', linewidth=1.6, label='ang gap (rad)')
        ax.legend(loc='upper left', fontsize=7, ncol=2)
        ax = self.axs2[1]
        ax.set_title("Node Frequency (Hz)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Hz")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 180)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(150, color='g', linestyle='--', linewidth=1.0, alpha=0.7, label='target 150Hz')
        self.line_freq, = ax.plot([], [], color='#1f77b4', linewidth=1.5, label='HFM freq')
        ax.legend(loc='upper left', fontsize=8)
        self.fig2.tight_layout()

        # Window 3: BLENDING telemetry (identical to the CB manager). alpha =
        # reference-level user<->policy authority; share = how much of the blended
        # twist magnitude came from the user vs the policy.
        self.fig3, self.axs3 = plt.subplots(2, 1, figsize=(10, 5))
        self.fig3.canvas.manager.set_window_title('Blending Telemetry')
        ax = self.axs3[0]
        ax.set_title("Blending authority \u03b1 (0=user, 1=policy)", fontsize=10, fontweight='bold')
        ax.set_ylabel("\u03b1"); ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color='#888', linestyle=':', linewidth=0.8)
        ax.grid(True, linestyle='--', alpha=0.6)
        self.line_alpha, = ax.plot([], [], color='#ff7f0e', linewidth=1.5, label='\u03b1')
        ax.legend(loc='upper left', fontsize=8)
        ax = self.axs3[1]
        ax.set_title("Blend share: (1-\u03b1)\u00b7v_user  vs  \u03b1\u00b7v_policy", fontsize=10, fontweight='bold')
        ax.set_ylabel("%"); ax.set_xlabel("Time (s)"); ax.set_ylim(-5, 105)
        ax.grid(True, linestyle='--', alpha=0.6)
        self.line_user_pct, = ax.plot([], [], color='#1f77b4', linewidth=1.4, label='user %')
        self.line_policy_pct, = ax.plot([], [], color='#ff7f0e', linewidth=1.4, label='policy %')
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        self.fig3.tight_layout()

        plt.show(block=False)

    def update_plot(self):
        with self.plot_lock:
            if len(self.t_data) == 0:
                return
            t_list = list(self.t_data)
            force = [list(self.force_data[i]) for i in range(3)]
            torque = [list(self.torque_data[i]) for i in range(3)]
            pos_err = list(self.pos_err_data)
            ang_err = list(self.ang_err_data)
            freq = list(self._freq_data)
            alpha_list = list(self.alpha_data)
            upct_list = list(self.user_pct_data)
            ppct_list = list(self.policy_pct_data)

        win = (t_list[-1] - self.plot_window_sec, t_list[-1])
        for i in range(3):
            self.lines_F[i].set_data(t_list, force[i])
            self.lines_T[i].set_data(t_list, torque[i])
        for ax in self.axs1:
            ax.set_xlim(*win)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        n = min(len(t_list), len(pos_err), len(ang_err))
        self.line_pos_err.set_data(t_list[:n], pos_err[:n])
        self.line_ang_err.set_data(t_list[:n], ang_err[:n])
        self.axs2[0].set_xlim(*win)
        self.axs2[0].relim()
        self.axs2[0].autoscale_view(scalex=False, scaley=True)
        nf = min(len(t_list), len(freq))
        self.line_freq.set_data(t_list[:nf], freq[:nf])
        self.axs2[1].set_xlim(*win)

        na = min(len(t_list), len(alpha_list))
        self.line_alpha.set_data(t_list[:na], alpha_list[:na])
        self.axs3[0].set_xlim(*win)
        ns = min(len(t_list), len(upct_list), len(ppct_list))
        self.line_user_pct.set_data(t_list[:ns], upct_list[:ns])
        self.line_policy_pct.set_data(t_list[:ns], ppct_list[:ns])
        self.axs3[1].set_xlim(*win)

        self.fig1.canvas.draw_idle()
        self.fig2.canvas.draw_idle()
        self.fig3.canvas.draw_idle()
        self.fig1.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = HapticForceManagerBlending()

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
