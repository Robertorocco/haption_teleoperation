#!/usr/bin/env python3
"""Live telemetry plotter for the Haption device: handle pose, velocity, and commanded force."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist, Wrench

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import time

class VirtuosePlotterNode(Node):
    def __init__(self):
        super().__init__('virtuose_realtime_plotter')

        self.max_length = 1500  # 10 s at 150 Hz
        self.start_time = time.time()
        self._lock = threading.Lock()

        self.pose_data  = deque(maxlen=self.max_length)
        self.vel_data   = deque(maxlen=self.max_length)
        self.force_data = deque(maxlen=self.max_length)

        self.create_subscription(Pose,   'virtuose/pose',      self.pose_callback,  10)
        self.create_subscription(Twist,  'virtuose/velocity',  self.vel_callback,   10)
        self.create_subscription(Wrench, 'virtuose/force_cmd', self.force_callback, 10)

    def get_current_time(self):
        return time.time() - self.start_time

    def pose_callback(self, msg):
        with self._lock:
            self.pose_data.append((self.get_current_time(),
                                   msg.position.x, msg.position.y, msg.position.z))

    def vel_callback(self, msg):
        with self._lock:
            self.vel_data.append((self.get_current_time(),
                                  msg.linear.x, msg.linear.y, msg.linear.z))

    def force_callback(self, msg):
        with self._lock:
            self.force_data.append((self.get_current_time(),
                                    msg.force.x, msg.force.y, msg.force.z))

    def get_snapshot(self):
        """Atomically snapshot all three deques in one lock acquisition."""
        with self._lock:
            return list(self.pose_data), list(self.vel_data), list(self.force_data)


# =========================================
# Helpers
# =========================================
def _set_ylim_from_data(ax, *data_lists):
    """Sets y-limits directly from the data (relim() can ValueError on transiently unsynced lines)."""
    all_vals = [v for dl in data_lists for v in dl]
    if not all_vals:
        return
    vmin, vmax = min(all_vals), max(all_vals)
    margin = max((vmax - vmin) * 0.1, 1e-3)
    ax.set_ylim(vmin - margin, vmax + margin)


# =========================================
# Matplotlib Animation
# =========================================
def update_plots(frame, node, lines, axes):
    current_time = node.get_current_time()
    window_start = max(0.0, current_time - 10.0)

    for ax in axes:
        ax.set_xlim(window_start, window_start + 10.0)

    pose_list, vel_list, force_list = node.get_snapshot()

    # --- POSE ---
    if pose_list:
        t  = [p[0] for p in pose_list]
        px = [p[1] for p in pose_list]
        py = [p[2] for p in pose_list]
        pz = [p[3] for p in pose_list]
        lines['p_x'].set_data(t, px)
        lines['p_y'].set_data(t, py)
        lines['p_z'].set_data(t, pz)
        _set_ylim_from_data(axes[0], px, py, pz)

    # --- VELOCITY ---
    if vel_list:
        t  = [v[0] for v in vel_list]
        vx = [v[1] for v in vel_list]
        vy = [v[2] for v in vel_list]
        vz = [v[3] for v in vel_list]
        lines['v_x'].set_data(t, vx)
        lines['v_y'].set_data(t, vy)
        lines['v_z'].set_data(t, vz)
        _set_ylim_from_data(axes[1], vx, vy, vz)

    # --- FORCE (only populated when an impedance controller publishes virtuose/force_cmd) ---
    if force_list:
        t  = [f[0] for f in force_list]
        fx = [f[1] for f in force_list]
        fy = [f[2] for f in force_list]
        fz = [f[3] for f in force_list]
        lines['f_x'].set_data(t, fx)
        lines['f_y'].set_data(t, fy)
        lines['f_z'].set_data(t, fz)
        _set_ylim_from_data(axes[2], fx, fy, fz)

    return lines.values()


def main(args=None):
    rclpy.init(args=args)
    node = VirtuosePlotterNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.canvas.manager.set_window_title('Virtuose Real-Time Telemetry')

    lines = {}

    axes[0].set_title('Cartesian Position')
    axes[0].set_ylabel('Position (m)')
    lines['p_x'], = axes[0].plot([], [], label='X', color='r')
    lines['p_y'], = axes[0].plot([], [], label='Y', color='g')
    lines['p_z'], = axes[0].plot([], [], label='Z', color='b')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    axes[1].set_title('Cartesian Velocity')
    axes[1].set_ylabel('Velocity (m/s)')
    lines['v_x'], = axes[1].plot([], [], label='v_x', color='r')
    lines['v_y'], = axes[1].plot([], [], label='v_y', color='g')
    lines['v_z'], = axes[1].plot([], [], label='v_z', color='b')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    axes[2].set_title('Impedance Force Command  (requires publisher on virtuose/force_cmd)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Force (N)')
    lines['f_x'], = axes[2].plot([], [], label='f_x', color='r')
    lines['f_y'], = axes[2].plot([], [], label='f_y', color='g')
    lines['f_z'], = axes[2].plot([], [], label='f_z', color='b')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)

    plt.tight_layout()

    ani = animation.FuncAnimation(
        fig, update_plots, fargs=(node, lines, axes),
        interval=50, blit=False, cache_frame_data=False
    )

    plt.show()

    node.destroy_node()
    rclpy.shutdown()
    spin_thread.join()


if __name__ == '__main__':
    main()