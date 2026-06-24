#!/usr/bin/env python3
"""
Workspace Debug Visualizer — 4 independent windows
=======================================================

Window 1 – Haption device in its native frame
    Origin  : Haption base
    X axis  : → toward the operator
    Z axis  : ↑ up

Window 2 – Haption device in Rotated frame (180° around Z)
    Origin  : Haption base
    X axis  : → away from operator (negative space)
    Z axis  : ↑ up

Window 3 – TIAGo end-effector in TIAGo base frame
    Origin  : TIAGo footprint centre
    X axis  : → robot forward
    Z axis  : ↑ up

Window 4 – Unified workspace expressed in TIAGo base frame
    Both workspaces are drawn; Haption is mapped into TIAGo frame.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation as R

# ─────────────────────────────────────────────────────────────────────────────
# Workspace constants  (edit here if limits change)
# ─────────────────────────────────────────────────────────────────────────────
TIAGO_MIN = np.array([0.100, -0.996,  0.200])
TIAGO_MAX = np.array([0.987,  0.567,  1.772])

# Measured Haption limits  (Haption base frame, X → user, Z ↑)
HAPTION_MIN    = np.array([0.192, -0.562, -0.352])
HAPTION_MAX    = np.array([0.738,  0.490,  0.424])
HAPTION_CENTER = np.array([0.46475608, -0.03577431,  0.03573696])


# ─────────────────────────────────────────────────────────────────────────────
class WorkspaceVisualizer(Node):

    def __init__(self):
        super().__init__('workspace_visualizer')

        # --- live state ---
        self.tiago_pos   = (TIAGO_MIN + TIAGO_MAX) / 2.0
        self.tiago_rot   = np.eye(3)
        self.haption_pos = HAPTION_CENTER.copy()
        self.haption_rot = np.eye(3)

        # Bridge Constants
        self.center_tiago_bridge = np.array([0.544, -0.215, 0.986])
        self.center_haption_bridge = np.array([0.4647, -0.0357, 0.0357])
        self.K = 1.0  # Isotropic scale factor

        
        # ------------------------------
        # --- subscribers ---
        self.create_subscription(
            Float64MultiArray, '/qp_debug/ee_real', self._tiago_cb, 10)
        self.create_subscription(
            Pose, 'virtuose/pose', self._haption_cb, 10)   # raw hardware pose

        # --- figures ---
        plt.ion()

        self.fig1 = plt.figure(figsize=(7, 6))
        self.ax1  = self.fig1.add_subplot(111, projection='3d')
        self.fig1.suptitle(
            "Window 1 – Haption Native  (Haption base frame)\n"
            "X → user   |   Y → operator-right   |   Z ↑", fontsize=9)

        self.fig2 = plt.figure(figsize=(7, 6))
        self.ax2  = self.fig2.add_subplot(111, projection='3d')
        self.fig2.suptitle(
            "Window 2 – Haption Rotated  (180° around Z)\n"
            "X → away from user   |   Y → operator-left   |   Z ↑", fontsize=9)

        self.fig3 = plt.figure(figsize=(7, 6))
        self.ax3  = self.fig3.add_subplot(111, projection='3d')
        self.fig3.suptitle(
            "Window 3 – TIAGo EE  (TIAGo base frame)\n"
            "X → forward   |   Y → robot-left   |   Z ↑", fontsize=9)

        self.fig4 = plt.figure(figsize=(7, 6))
        self.ax4  = self.fig4.add_subplot(111, projection='3d')
        self.fig4.suptitle(
            "Window 4 – Unified workspace  (expressed in TIAGo base frame)\n"
            "Haption WS mapped: flip X, flip Y, scale Z", fontsize=9)
        
        self.fig5 = plt.figure(figsize=(7, 6))
        self.ax5  = self.fig5.add_subplot(111, projection='3d')
        self.fig5.suptitle(
            f"Window 5 – Bridge Workspace (Isotropic, K={self.K})\n"
            "Preserves geometry (circles stay circles)", fontsize=9)
        
        self.fig6 = plt.figure(figsize=(7, 6))
        self.ax6  = self.fig6.add_subplot(111, projection='3d')
        self.fig6.suptitle(
            "Window 6 – Pure Orientation Alignment\n"
            "(Solid = TIAGo Gripper, Dashed = Mapped Haption Handle)", fontsize=9)

        
        # Tool Frame Alignment: +90 degrees around Y to align Haption handle with TIAGo gripper
        self.R_haption_to_tiago_tool = R.from_euler('y', np.pi/2).as_matrix()

        # 10 Hz redraw
        self.create_timer(0.10, self._update)

    # ── ROS callbacks ──────────────────────────────────────────────────────
    def _tiago_cb(self, msg):
        try:
            self.tiago_pos = np.array(msg.data[0:3])
            rpy = np.array(msg.data[12:15])
            self.tiago_rot = R.from_euler('xyz', rpy, degrees=False).as_matrix()
        except IndexError:
            self.get_logger().warn('Malformed /qp_debug/ee_real')

    def _haption_cb(self, msg):
        self.haption_pos = np.array([
            msg.position.x, msg.position.y, msg.position.z])
        q = [msg.orientation.x, msg.orientation.y,
             msg.orientation.z, msg.orientation.w]
        self.haption_rot = R.from_quat(q).as_matrix()

    # ── Coordinate mapping ─────────────────────────────────────────────────
    @staticmethod
    def get_rotated_pose(pos, rot_mat):
        """Applies a 180-degree rotation around the Z-axis."""
        R_z_180 = np.array([[-1.0,  0.0, 0.0], 
                            [ 0.0, -1.0, 0.0], 
                            [ 0.0,  0.0, 1.0]])
        rotated_pos = R_z_180 @ pos
        rotated_rot = R_z_180 @ rot_mat
        return rotated_pos, rotated_rot

    @staticmethod
    def haption_to_tiago(h_pos,
                         h_min=HAPTION_MIN, h_max=HAPTION_MAX,
                         t_min=TIAGO_MIN,   t_max=TIAGO_MAX):
        """Map one point from Native Haption frame → TIAGo frame for unified display."""
        h_range = h_max - h_min
        t_range = t_max - t_min

        norm = (h_pos - h_min) / h_range          

        t_x = t_max[0] - norm[0] * t_range[0]    # flip X
        t_y = t_max[1] - norm[1] * t_range[1]    # flip Y
        t_z = t_min[2] + norm[2] * t_range[2]    # scale Z only

        return np.array([t_x, t_y, t_z])

    def map_haption_to_tiago_bridge(self, h_pos):
        """
        Applies the exact math from haption_bridge_VIRTMECH.py 
        (Forward mapping: Haption -> TIAGo)
        """
        # Haption displacement from its center, scaled by K
        delta_h = (h_pos - self.center_haption_bridge) * self.K
        
        # Apply to TIAGo center, flipping X and Y (180 deg rot around Z)
        t_x = self.center_tiago_bridge[0] - delta_h[0]
        t_y = self.center_tiago_bridge[1] - delta_h[1]
        t_z = self.center_tiago_bridge[2] + delta_h[2]
        
        return np.array([t_x, t_y, t_z])
    # -----------------------------------------

    # ── Drawing helpers ────────────────────────────────────────────────────
    @staticmethod
    def _aabb_faces(mn, mx):
        v = np.array([
            [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
        ])
        return [
            [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
            [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
            [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]],
        ]

    def _draw_box(self, ax, mn, mx, color, alpha=0.08, lw=0.8):
        col = Poly3DCollection(
            self._aabb_faces(mn, mx),
            alpha=alpha, facecolors=color, edgecolors=color, linewidths=lw)
        ax.add_collection3d(col)

    @staticmethod
    def _draw_frame(ax, origin, rot, scale=0.05):
        for i, c in enumerate(('r', 'g', 'b')):
            v = rot[:, i] * scale
            ax.quiver(*origin, *v, color=c, arrow_length_ratio=0.25,
                      linewidth=1.5)

    # ── Main update ────────────────────────────────────────────────────────
    def _update(self):

        # ── Window 1 : Haption Native ─────────────────────────────────── #
        ax = self.ax1
        ax.cla()

        self._draw_box(ax, HAPTION_MIN, HAPTION_MAX, 'magenta', alpha=0.10)
        ax.scatter(*HAPTION_CENTER, c='k', s=60, marker='+', zorder=4, label='WS centre')
        ax.scatter(*self.haption_pos, c='magenta', s=120, marker='o', zorder=6, label='Haption EE', edgecolors='k', linewidths=0.6)
        self._draw_frame(ax, self.haption_pos, self.haption_rot, scale=0.04)

        ax.set_xlim([0.05,  0.85])
        ax.set_ylim([-0.65,  0.60])
        ax.set_zlim([-0.45,  0.55])
        ax.set_xlabel('X  (→ user) [m]',  labelpad=4)
        ax.set_ylabel('Y  (→ right) [m]',           labelpad=4)
        ax.set_zlabel('Z  (↑) [m]',       labelpad=4)
        ax.legend(loc='upper left', fontsize=8)

        # ── Window 2 : Haption Rotated ────────────────────────────────── #
        ax = self.ax2
        ax.cla()
        
        # Calculate rotated bounds and ensure min/max alignment
        rot_min, _ = self.get_rotated_pose(HAPTION_MIN, np.eye(3))
        rot_max, _ = self.get_rotated_pose(HAPTION_MAX, np.eye(3))
        r_min = np.minimum(rot_min, rot_max)
        r_max = np.maximum(rot_min, rot_max)
        
        r_center, _ = self.get_rotated_pose(HAPTION_CENTER, np.eye(3))
        r_pos, r_rot = self.get_rotated_pose(self.haption_pos, self.haption_rot)

        self._draw_box(ax, r_min, r_max, 'orange', alpha=0.10)
        ax.scatter(*r_center, c='k', s=60, marker='+', zorder=4, label='Rotated WS centre')
        ax.scatter(*r_pos, c='orange', s=120, marker='o', zorder=6, label='Rotated EE', edgecolors='k', linewidths=0.6)
        self._draw_frame(ax, r_pos, r_rot, scale=0.04)

        ax.set_xlim([-0.85, -0.05])
        ax.set_ylim([-0.60,  0.65])
        ax.set_zlim([-0.45,  0.55])
        ax.set_xlabel('X  (→ away from user) [m]', labelpad=4)
        ax.set_ylabel('Y  (→ left) [m]', labelpad=4)
        ax.set_zlabel('Z  (↑) [m]', labelpad=4)
        ax.legend(loc='upper right', fontsize=8)

        # ── Window 3 : TIAGo EE ───────────────────────────────────────── #
        ax = self.ax3
        ax.cla()

        self._draw_box(ax, TIAGO_MIN, TIAGO_MAX, 'cyan', alpha=0.10)
        t_center = (TIAGO_MIN + TIAGO_MAX) / 2
        ax.scatter(*t_center, c='k', s=60, marker='+', zorder=4, label='WS centre')
        ax.scatter(*self.tiago_pos, c='cyan', s=120, marker='o', zorder=6, label='TIAGo EE', edgecolors='k', linewidths=0.6)
        self._draw_frame(ax, self.tiago_pos, self.tiago_rot, scale=0.08)

        ax.set_xlim([-0.05, 1.10])
        ax.set_ylim([-1.10, 0.70])
        ax.set_zlim([ 0.00, 2.00])
        ax.set_xlabel('X  (→ forward) [m]', labelpad=4)
        ax.set_ylabel('Y  (→ left) [m]',    labelpad=4)
        ax.set_zlabel('Z  (↑) [m]',         labelpad=4)
        ax.legend(loc='upper left', fontsize=8)

        # ── Window 4 : Unified workspace ──────────────────────────────── #
        ax = self.ax4
        ax.cla()

        self._draw_box(ax, TIAGO_MIN, TIAGO_MAX, 'cyan', alpha=0.07, lw=0.8)

        h_corner_a = self.haption_to_tiago(HAPTION_MIN)
        h_corner_b = self.haption_to_tiago(HAPTION_MAX)
        mn_h = np.minimum(h_corner_a, h_corner_b)
        mx_h = np.maximum(h_corner_a, h_corner_b)
        self._draw_box(ax, mn_h, mx_h, 'magenta', alpha=0.07, lw=1.5)

        ax.scatter(*self.tiago_pos, c='cyan', s=140, marker='o', zorder=6, label='TIAGo EE', edgecolors='k', linewidths=0.6)
        self._draw_frame(ax, self.tiago_pos, self.tiago_rot, scale=0.07)

        h_mapped = self.haption_to_tiago(self.haption_pos)
        ax.scatter(*h_mapped, c='magenta', s=140, marker='X', zorder=6, label='Haption EE  (mapped)', edgecolors='k', linewidths=0.6)

        dist = np.linalg.norm(self.tiago_pos - h_mapped)
        mid  = (self.tiago_pos + h_mapped) / 2
        ax.text(*mid, f' Δ={dist:.3f} m', fontsize=7, color='gray')

        ax.set_xlim([-0.05, 1.10])
        ax.set_ylim([-1.10, 0.70])
        ax.set_zlim([ 0.00, 2.00])
        ax.set_xlabel('X  (TIAGo forward) [m]', labelpad=4)
        ax.set_ylabel('Y  (TIAGo left) [m]',    labelpad=4)
        ax.set_zlabel('Z  (↑) [m]',             labelpad=4)
        ax.legend(loc='upper left', fontsize=8)

        # ── Window 5 : Isotropic Bridge workspace ─────────────────────── #
        ax = self.ax5
        ax.cla()

        # 1. Draw TIAGo actual WS (Cyan)
        self._draw_box(ax, TIAGO_MIN, TIAGO_MAX, 'cyan', alpha=0.07, lw=0.8)

        # 2. Draw Haption WS mapped via Bridge Logic (Magenta)
        # We map the min and max points, then find the new bounding box
        p1 = self.map_haption_to_tiago_bridge(HAPTION_MIN)
        p2 = self.map_haption_to_tiago_bridge(HAPTION_MAX)
        mn_hb = np.minimum(p1, p2)
        mx_hb = np.maximum(p1, p2)
        
        self._draw_box(ax, mn_hb, mx_hb, 'magenta', alpha=0.07, lw=1.5)
        ax.scatter(*self.center_tiago_bridge, c='k', s=60, marker='+', zorder=4, label='Bridge Center')

        # 3. Draw actual TIAGo EE (Cyan)
        ax.scatter(*self.tiago_pos, c='cyan', s=140, marker='o', zorder=6, label='TIAGo EE', edgecolors='k', linewidths=0.6)
        
        # 4. Draw mapped Haption EE (Magenta X)
        h_mapped_bridge = self.map_haption_to_tiago_bridge(self.haption_pos)
        ax.scatter(*h_mapped_bridge, c='magenta', s=140, marker='X', zorder=6, label='Haption EE (Bridge)', edgecolors='k', linewidths=0.6)

        # Distance indicator
        dist_b = np.linalg.norm(self.tiago_pos - h_mapped_bridge)
        mid_b  = (self.tiago_pos + h_mapped_bridge) / 2
        ax.text(*mid_b, f' Δ={dist_b:.3f} m', fontsize=7, color='gray')

        ax.set_xlim([-0.05, 1.10])
        ax.set_ylim([-1.10, 0.70])
        ax.set_zlim([ 0.00, 2.00])
        ax.set_xlabel('X  (TIAGo forward) [m]', labelpad=4)
        ax.set_ylabel('Y  (TIAGo left) [m]',    labelpad=4)
        ax.set_zlabel('Z  (↑) [m]',             labelpad=4)
        ax.legend(loc='upper left', fontsize=8)

        # ── Window 6 : Pure Orientation (The Tool Fix) ────────────────── #
        ax = self.ax6
        ax.cla()

        # 1. Draw actual TIAGo orientation (Solid Lines)
        self._draw_pure_rotation(ax, self.tiago_rot, scale=1.0, linestyle='-')

        # 2. Apply Base Flip AND Tool Alignment to Haption
        # First: Flip the base frame 180 around Z (like in the bridge)
        R_base_flip = R.from_euler('z', np.pi).as_matrix()
        
        # Math: BaseFlip * HaptionRaw * ToolAlignment
        haption_fixed_rot = R_base_flip @ self.haption_rot @ self.R_haption_to_tiago_tool

        # 3. Draw mapped Haption orientation (Dashed Lines)
        self._draw_pure_rotation(ax, haption_fixed_rot, scale=0.8, linestyle='--')

        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel('X (Forward)')
        ax.set_ylabel('Y (Right)')
        ax.set_zlabel('Z (Down)')
        ax.legend(['TIAGo Gripper', 'Mapped Haption Handle'], loc='upper left', fontsize=8)

        # flush all four canvases
        for fig in (self.fig1, self.fig2, self.fig3, self.fig4, self.fig5, self.fig6):
            fig.canvas.draw_idle()
        plt.pause(0.01)
    
    def _draw_pure_rotation(self, ax, rot_matrix, scale=1.0, linestyle='-'):
        """Draws a coordinate triad at the origin given a Rotation matrix."""
        origin = np.zeros(3)
        axes = rot_matrix @ (np.eye(3) * scale)
        
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']
        
        for i in range(3):
            ax.plot([origin[0], axes[0, i]], 
                    [origin[1], axes[1, i]], 
                    [origin[2], axes[2, i]], 
                    color=colors[i], linestyle=linestyle, linewidth=2.5)
            ax.text(axes[0, i]*1.1, axes[1, i]*1.1, axes[2, i]*1.1, labels[i], color=colors[i], fontsize=10)


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = WorkspaceVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()