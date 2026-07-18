#!/usr/bin/env python3
"""JOYSTICK sync-only baseline (F=0, B=0): centering spring toward home + vibration cues, no assistance."""

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

# Cross-package condition selector + joystick home pose and spring gains.
import triago_control.qp_controller.config as cfg


class HapticForceManagerJoystickSync(Node):
    # Baseline joystick force manager: only the homing spring (orientation-sync) + cues are rendered.
    def __init__(self):
        super().__init__('haptic_force_manager_joystick_sync')

        # Hard-error at startup unless config.py selects the JOYSTICK sync-only cell.
        cfg.validate_condition('haptic_force_manager_J',
                               control_mode=cfg.JOYSTICK, feedback=False, blending=False)

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

        # Out-of-deadzone cue: zero-mean buzz whenever a non-zero twist is being commanded.
        self.VIB_AMP = 0.05           # Nm
        self.vib_toggle = 1.0         # sign flip every frame -> ~75 Hz square wave

        # Autonomous-grasp cue, unified across all 8 cells.
        self.grasp_active = False
        self.GRASP_VIB_AMP = 0.07    # Nm
        self.grasp_vib_toggle = 1.0

        # virtuose/pose is geometry_msgs/Pose (not PoseStamped) -- the wrong type silently receives nothing.
        self.create_subscription(Pose, 'virtuose/pose', self.handle_pose_cb, 10)
        self.create_subscription(Twist, 'virtuose/velocity', self.vel_cb, 10)
        self.create_subscription(
            Float64MultiArray, cfg.JOYSTICK_HOME_POSE_TOPIC, self.home_pose_cb, 10)
        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_cb, 10)

        self.force_pub = self.create_publisher(Wrench, 'virtuose/force_cmd', 10)

        # Plot buffers (10 s window at 150 Hz), guarded by a lock shared with the UI thread.
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

        self.dt = 1.0 / 150.0
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.setup_plot()
        self.get_logger().info(
            f"[HFM-JOYSTICK-SYNC] Sync-only baseline (no assist) started. "
            f"Home position (Haption) = {self.home_pos.tolist()}, "
            f"KP_LIN={self.KP_LIN}, KP_ANG={self.KP_ANG}. "
            f"Listening for the live home pose on '{cfg.JOYSTICK_HOME_POSE_TOPIC}'.")

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

    def grasp_active_cb(self, msg):
        """Tracks whether shared autonomy is autonomously driving a grasp."""
        self.grasp_active = bool(msg.data)

    # ------------------------------------------------------------------ force
    def compute_spring(self):
        """Spring-damper wrench (Haption base frame) pulling the handle to the home pose."""
        f = np.zeros(6)
        if self.handle_pos is None or self.handle_rot is None:
            return f

        f[0:3] = self.KP_LIN * (self.home_pos - self.handle_pos) - self.KD_LIN * self.handle_vel[0:3]

        # Home and handle are both in the Haption frame, so no frame mapping is needed.
        err_rotvec = (self.home_rot * self.handle_rot.inv()).as_rotvec()
        f[3:6] = self.KP_ANG * err_rotvec - self.KD_ANG * self.handle_vel[3:6]
        return f

    def control_loop(self):
        """150 Hz: renders the homing spring, adds the cues, clips, publishes, buffers."""
        f = self.compute_spring()

        # Authority cap: proportional rescale before the vibration cues.
        fn = np.linalg.norm(f[0:3])
        if fn > self.MAX_TOTAL_FORCE:
            f[0:3] *= self.MAX_TOTAL_FORCE / fn
        tn = np.linalg.norm(f[3:6])
        if tn > self.MAX_TOTAL_TORQUE:
            f[3:6] *= self.MAX_TOTAL_TORQUE / tn

        # Out-of-deadzone cue: buzz exactly while a non-zero twist is being commanded.
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

        # Autonomous-grasp cue.
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

        # Buffer telemetry.
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

    # ------------------------------------------------------------------ plotting
    def setup_plot(self):
        """Initializes the live plots: spring wrench + displacement-from-home / loop rate."""
        plt.ion()
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']

        self.fig1, self.axs1 = plt.subplots(2, 1, figsize=(10, 5))
        self.fig1.canvas.manager.set_window_title('Joystick Sync-Only Restorative Spring')
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

        plt.show(block=False)

    def update_plot(self):
        """Snapshots buffers under the lock and refreshes the Matplotlib UI."""
        with self.plot_lock:
            if len(self.t_data) == 0:
                return
            t_list = list(self.t_data)
            force = [list(self.force_data[i]) for i in range(3)]
            torque = [list(self.torque_data[i]) for i in range(3)]
            pos_err = list(self.pos_err_data)
            ang_err = list(self.ang_err_data)
            freq = list(self._freq_data)

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

        self.fig1.canvas.draw_idle()
        self.fig2.canvas.draw_idle()
        self.fig1.canvas.flush_events()


def main(args=None):
    """Spins ROS on a daemon thread and drives Matplotlib on the main thread."""
    rclpy.init(args=args)
    node = HapticForceManagerJoystickSync()

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
