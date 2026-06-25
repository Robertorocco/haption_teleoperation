#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench, Twist
from std_msgs.msg import Bool, Float64MultiArray, Float64, String
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from collections import deque
from geometry_msgs.msg import PoseStamped #
import matplotlib.pyplot as plt
import matplotlib

# Set backend to avoid blocking the ROS spin loop
matplotlib.use('TkAgg') 

class HapticForceManager(Node):
    def __init__(self):
        """Initializes the Haptic Force Manager, publishers, subscribers, thread locks, and plot buffers."""
        super().__init__('haptic_force_manager')

        # --- State Variables ---
        self.pos_target = None
        self.rot_target = None
        self.pos_real = None
        self.rot_real = None
        self.vel_haption = np.zeros(6)  

        # --- CBF State Variables ---
        self.grad_cbf_right = np.zeros(6)
        self.lambda_cbf = 0.0

        # --- CBF Smoothing Parameters ---
        self.f_cbf_filtered = np.zeros(6)
        #update current force by taking only 15% of the new raw data and keeping 85% of old value.
        self.alpha_cbf = 0.15          # LPF cutoff (lower = smoother but slightly delayed)

        self.MAX_CBF_FORCE = 15.0       # N (Maximum comfortable repulsion force)
        self.MAX_CBF_TORQUE = 1.0      # Nm (Maximum comfortable repulsion torque)

         # --- NEW: Inference State Variables ---
        self.goal_names = []
        self.goal_probs = []
        self.user_policies = []

        # --- Tunable Force Parameters ---
        # Represents the viscous drag pulling the user toward the optimal policy
        self.B_guide_lin = 90.0   # N/(m/s) (Translational damping)
        self.B_guide_ang = 0.5    # Nm/(rad/s) (Rotational damping)

        # --- Continuous Policy-Merging (Belief-Weighted Blend) Parameters ---
        # Instead of a winner-take-all gate that snaps pi_ref from one goal's
        # policy to another (and creates force discontinuities), we blend ALL
        # leaf policies by their joint hierarchical probability. The blended
        # reference twist is a smooth function of the beliefs, so it never jumps.
        #
        # confidence gain : continuous fade-in of the whole guidance wrench based
        #                   on how "peaked" the blended belief is (1 - normalised
        #                   entropy), shaped by a smoothstep with a soft floor/cap.
        self.GUIDE_CONF_LO   = 0.15   # below this confidence -> transparent (no force)
        self.GUIDE_CONF_HI   = 0.85   # at/above this confidence -> full guidance gain
        # Temporal smoothing of the final guidance wrench. Guarantees C0 continuity
        # even if a probability sample arrives noisy; removes any residual stepping.
        self.alpha_guide     = 0.15   # LPF coefficient (lower = smoother, more lag)
        self.f_guide_filtered = np.zeros(6)

        # --- User-led guidance gating ("listen more, lock less") ---
        # The guidance fades out (a) when the user is essentially still — so the
        # autonomy never drags a passive hand and the user always initiates — and
        # (b) when the user moves AGAINST the suggested direction — so a change of
        # mind is easy and never fought. Full guidance only when the user is
        # actively moving roughly along the suggestion (the behaviour we like).
        self.LEAD_V_STILL = 0.005   # m/s  below -> "still" (guidance ~0)
        self.LEAD_V_MOVE  = 0.030   # m/s  above -> fully engaged
        self.LEAD_COS_LO  = -0.10   # cos(user_vel, guide) below -> disagreeing (gain 0)
        self.LEAD_COS_HI  =  0.50   # above -> agreeing (gain 1)

        # --- Position Virtual Fixture (strong "funnel" near a confident goal) ---
        # The viscous F_guide vanishes near the goal (policy velocity -> 0), so it
        # cannot hold the user precisely AT the grasp pose against the CBF that
        # pushes the gripper off the cylinder. This POSITION spring pulls the user
        # toward the exact active-goal pose and does NOT vanish near it, giving a
        # strong, stable "lock-in" once a goal is confidently identified. Gated by
        # confidence so it stays silent in free space / when intent is ambiguous.
        self.fix_goal_pos = None
        self.fix_goal_rot = None
        self.fix_confidence = 0.0
        self.K_fix_force  = 20.0      # N/m   position spring toward the grasp pose (gentle nudge)
        self.K_fix_torque = 0.15      # Nm/rad orientation spring toward grasp orientation
        self.MAX_FIX_FORCE  = 3.0     # N     saturation (just a slight end-of-approach help)
        self.MAX_FIX_TORQUE = 0.25    # Nm    saturation
        self.FIX_CONF_LO = 0.55       # below this belief -> fixture OFF
        self.FIX_CONF_HI = 0.85       # at/above this belief -> full fixture
        self.alpha_fix = 0.15         # LPF on the fixture wrench (C0 continuity)
        self.f_fix_filtered = np.zeros(6)

        # --- NEW: Clutching Architecture Variables ---
        self.is_clutching = False
        self.was_clutching_last_frame = False
        self.f_clutch_frozen = np.zeros(6)
        self.K_align = 10.0  # Nm/rad (Rotational stiffness for alignment guidance)
        self.rot_haption = None


        # --- Tunable Force Parameters ---
        # F_guide (Virtual Fixture Guidance Gains)
        self.K_guide_force = 90.0   # N/m (Translational stiffness)
        self.K_guide_torque = 0.3   # Nm/rad (Rotational stiffness)        
        #Depends on the reference generated

        # --- Articular Limit Variables ---
        self.joint_pos = np.zeros(6)
        self.joint_min = np.array([-0.804283, -1.65038, 0.728283, -3.02431, -1.28196, -2.05398])
        self.joint_max = np.array([0.781944, -0.0654231, 2.49752, 2.82038, 1.04722, 2.09453])
        
        #  Articular Limit Vibration Tuning Parameters
        self.LIMIT_OUTER = 0.25       # Radians where vibration starts
        self.LIMIT_INNER = 0.15       # Radians where vibration hits maximum
        self.AMP_MIN = 0.05           # Nm torque at the outer boundary
        self.AMP_MAX = 0.07           # Nm torque at the inner boundary
        self.vib_toggle = 1.0         # Toggles between 1 and -1 every frame for 75Hz square wave

        # --- Tunable Force Parameters ---
        self.Kp_sync = 10.0#15.0  
        self.Kd_sync = 0.0  #if global damping is added, set this to 0 to avoid overdamping
        self.K_cbf_force = 2.0   
        self.K_cbf_torque = 0.1  
        self.MAX_FORCE = 10.0 
        self.MAX_TORQUE = 1.0 

        # --- Assistive-wrench authority cap ---
        # Hard ceiling on the MAGNITUDE of the assistive wrench (f_total_normal =
        # sync + cbf + guide + fix). When the sum exceeds this, the WHOLE vector
        # is scaled down proportionally — so the relative contribution proportions
        # are preserved, but the autonomy can never overpower the operator. Tune
        # these to set "how much the assistance is allowed to push".
        self.MAX_TOTAL_FORCE  = 5.0   # N
        self.MAX_TOTAL_TORQUE = 0.4   # Nm

        # --- Data Buffers & Synchronization ---
        self.plot_lock = threading.Lock()
        self.plot_window_sec = 10.0
        self.buffer_size = int(150 * self.plot_window_sec)
        self.t_data = deque(maxlen=self.buffer_size)
        self.start_time = time.time()
        self.v_lin_data = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.v_ang_data = [deque(maxlen=self.buffer_size) for _ in range(3)]

        # Per-component wrench history (Superposition window: no Total here).
        self.f_data = {
            'Sync':  {'F': [deque(maxlen=self.buffer_size) for _ in range(3)], 'T': [deque(maxlen=self.buffer_size) for _ in range(3)]},
            'CBF':   {'F': [deque(maxlen=self.buffer_size) for _ in range(3)], 'T': [deque(maxlen=self.buffer_size) for _ in range(3)]},
            'Guide': {'F': [deque(maxlen=self.buffer_size) for _ in range(3)], 'T': [deque(maxlen=self.buffer_size) for _ in range(3)]},
            'Limit': {'F': [deque(maxlen=self.buffer_size) for _ in range(3)], 'T': [deque(maxlen=self.buffer_size) for _ in range(3)]}
        }

        # Dedicated f_total window: final published wrench + per-source % breakdown.
        self.ftot_data = {'F': [deque(maxlen=self.buffer_size) for _ in range(3)],
                          'T': [deque(maxlen=self.buffer_size) for _ in range(3)]}
        self.pct_force  = {'Sync': deque(maxlen=self.buffer_size),
                           'CBF':  deque(maxlen=self.buffer_size),
                           'Guide': deque(maxlen=self.buffer_size)}
        self.pct_torque = {'Sync': deque(maxlen=self.buffer_size),
                           'CBF':  deque(maxlen=self.buffer_size),
                           'Guide': deque(maxlen=self.buffer_size)}

        # --- ROS 2 Interfaces ---
        self.create_subscription(Float64MultiArray, '/arm_right/cartesian_reference', self.target_cb, 10)
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self.real_cb, 10)
        self.create_subscription(Twist, 'virtuose/velocity', self.vel_cb, 10)
        self.create_subscription(Float64MultiArray, '/collision_constraints', self.cbf_gradient_cb, 10)
        self.create_subscription(Float64, '/qp_debug/lambda_cbf', self.lambda_cb, 10)
        self.create_subscription(Float64MultiArray, 'virtuose/articular_position', self.joint_cb, 10)
        #self.create_subscription(Float64MultiArray, '/shared_autonomy/assistive_reference', self.assist_cb, 10)
        self.create_subscription(Bool, 'virtuose/button_right', self.button_cb, 10)
        self.create_subscription(PoseStamped, 'virtuose/pose', self.haption_pose_cb, 10)
        #Unified Inference State Subscribers
        self.create_subscription(String, '/shared_autonomy/goal_names', self.goal_names_cb, 10)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/goal_probabilities', self.goal_probs_cb, 10)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/user_policy', self.user_policy_cb, 10)
        # Active goal pose + confidence for the position virtual fixture
        self.create_subscription(Float64MultiArray, '/shared_autonomy/active_goal_pose', self.goal_pose_cb, 10)
        
        self.force_pub = self.create_publisher(Wrench, 'virtuose/force_cmd', 10)

        # --- Timers ---
        self.dt = 1.0 / 150.0
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.setup_plot()
        self.get_logger().info("Haptic Force Manager (tutorial) started.")

    # =========================
    # PLOT SETUP & UPDATE
    # =========================
    def setup_plot(self):
        """Initializes the live Matplotlib windows."""
        plt.ion()
        self.fig, self.axs = plt.subplots(4, 2, figsize=(12, 9))
        self.fig.canvas.manager.set_window_title('Haptic Force Superposition')
        
        self.lines = {}
        categories = ['Sync', 'CBF', 'Guide', 'Limit']
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']
        
        for row, cat in enumerate(categories):
            self.lines[cat] = {'F': [], 'T': []}
            
            # Left Column (Forces)
            ax_f = self.axs[row, 0]
            ax_f.set_title(f"{cat} Wrench - FORCE (N)", fontsize=10, pad=3)
            ax_f.set_ylabel("Force (N)")
            ax_f.grid(True, linestyle='--', alpha=0.6)
            for i in range(3):
                line, = ax_f.plot([], [], color=colors[i], label=f"F{labels[i]}")
                self.lines[cat]['F'].append(line)
            ax_f.legend(loc='upper left', fontsize=8)

            # Right Column (Torques)
            ax_t = self.axs[row, 1]
            ax_t.set_title(f"{cat} Wrench - TORQUE (Nm)", fontsize=10, pad=3)
            ax_t.set_ylabel("Torque (Nm)")
            ax_t.grid(True, linestyle='--', alpha=0.6)
            for i in range(3):
                line, = ax_t.plot([], [], color=colors[i], label=f"T{labels[i]}")
                self.lines[cat]['T'].append(line)
            ax_t.legend(loc='upper left', fontsize=8)

        # Format X-axis for the bottom row only
        for col in range(2):
            self.axs[3, col].set_xlabel("Time (s)")
            
        self.fig.tight_layout()

        # ========================================================
        # --- F_total Window (final published wrench + % breakdown) ---
        # ========================================================
        self.fig_tot, self.axs_tot = plt.subplots(4, 1, figsize=(9, 10))
        self.fig_tot.canvas.manager.set_window_title('Total Wrench (published to device)')
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']
        src_colors = {'Sync': '#1f77b4', 'CBF': '#d62728', 'Guide': '#2ca02c'}

        # Subplot 0: f_total FORCE components + ±MAX_TOTAL_FORCE dashed
        ax = self.axs_tot[0]
        ax.set_title("f_total — FORCE components (N)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Force (N)")
        ax.grid(True, linestyle='--', alpha=0.6)
        self.lines_ftot_F = [ax.plot([], [], color=colors[i], label=f"F{labels[i]}")[0] for i in range(3)]
        ax.axhline( self.MAX_TOTAL_FORCE, color='k', linestyle='--', linewidth=1.0, alpha=0.7, label='±max')
        ax.axhline(-self.MAX_TOTAL_FORCE, color='k', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.legend(loc='upper left', fontsize=8, ncol=4)

        # Subplot 1: f_total TORQUE components + ±MAX_TOTAL_TORQUE dashed
        ax = self.axs_tot[1]
        ax.set_title("f_total — TORQUE components (Nm)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Torque (Nm)")
        ax.grid(True, linestyle='--', alpha=0.6)
        self.lines_ftot_T = [ax.plot([], [], color=colors[i], label=f"T{labels[i]}")[0] for i in range(3)]
        ax.axhline( self.MAX_TOTAL_TORQUE, color='k', linestyle='--', linewidth=1.0, alpha=0.7, label='±max')
        ax.axhline(-self.MAX_TOTAL_TORQUE, color='k', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.legend(loc='upper left', fontsize=8, ncol=4)

        # Subplot 2: % of FORCE magnitude from each source
        ax = self.axs_tot[2]
        ax.set_title("Force contribution share (%)", fontsize=10, fontweight='bold')
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)
        self.lines_pctF = {k: ax.plot([], [], color=src_colors[k], label=k)[0]
                           for k in ['Sync', 'CBF', 'Guide']}
        ax.legend(loc='upper left', fontsize=8, ncol=3)

        # Subplot 3: % of TORQUE magnitude from each source
        ax = self.axs_tot[3]
        ax.set_title("Torque contribution share (%)", fontsize=10, fontweight='bold')
        ax.set_ylabel("%")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)
        self.lines_pctT = {k: ax.plot([], [], color=src_colors[k], label=k)[0]
                           for k in ['Sync', 'CBF', 'Guide']}
        ax.legend(loc='upper left', fontsize=8, ncol=3)

        self.fig_tot.tight_layout()

        # ========================================================
        # --- Twist Analyzer Window (2 Stacked Subplots) ---
        # ========================================================
        self.fig_v, (self.ax_v_lin, self.ax_v_ang) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig_v.canvas.manager.set_window_title('Haption 6D Twist Analyzer')
        
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']
        
        # --- TOP SUBPLOT: Linear Velocity ---
        self.ax_v_lin.set_title("Linear Velocity Components", fontsize=11, fontweight='bold')
        self.ax_v_lin.set_ylabel("Velocity (m/s)")
        self.ax_v_lin.grid(True, linestyle='--', alpha=0.6)
        
        self.lines_v_lin = []
        for i in range(3):
            line, = self.ax_v_lin.plot([], [], color=colors[i], linewidth=1.5, label=f"v_{labels[i]}")
            self.lines_v_lin.append(line)
        self.ax_v_lin.legend(loc='upper right')
        
        # --- BOTTOM SUBPLOT: Angular Velocity ---
        self.ax_v_ang.set_title("Angular Velocity Components", fontsize=11, fontweight='bold')
        self.ax_v_ang.set_ylabel("Velocity (rad/s)")
        self.ax_v_ang.set_xlabel("Time (s)")
        self.ax_v_ang.grid(True, linestyle='--', alpha=0.6)
        
        self.lines_v_ang = []
        for i in range(3):
            line, = self.ax_v_ang.plot([], [], color=colors[i], linewidth=1.5, label=f"w_{labels[i]}")
            self.lines_v_ang.append(line)
        self.ax_v_ang.legend(loc='upper right')
        
        self.fig_v.tight_layout()
        plt.show(block=False)


    def update_plot(self):
        """Safely captures a synchronized data snapshot via thread lock and updates the Matplotlib UI."""
        # 1. Snapshot the data inside the lock
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
            v_lin_lists = [list(self.v_lin_data[i]) for i in range(3)]
            v_ang_lists = [list(self.v_ang_data[i]) for i in range(3)]

        # 2. Update Matplotlib outside the lock to prevent stalling the ROS 2 loop
        current_t = t_list[-1]
        win = (current_t - self.plot_window_sec, current_t)

        # --- Superposition window (Sync / CBF / Guide / Limit) ---
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

        # --- F_total window ---
        for i in range(3):
            self.lines_ftot_F[i].set_data(t_list, ftot_F[i])
            self.lines_ftot_T[i].set_data(t_list, ftot_T[i])
        for k in ['Sync', 'CBF', 'Guide']:
            self.lines_pctF[k].set_data(t_list, pctF[k])
            self.lines_pctT[k].set_data(t_list, pctT[k])

        # Force components: y-range pinned a little beyond the cap for context
        self.axs_tot[0].set_xlim(*win)
        self.axs_tot[0].set_ylim(-self.MAX_TOTAL_FORCE * 1.4, self.MAX_TOTAL_FORCE * 1.4)
        self.axs_tot[1].set_xlim(*win)
        self.axs_tot[1].set_ylim(-self.MAX_TOTAL_TORQUE * 1.4, self.MAX_TOTAL_TORQUE * 1.4)
        self.axs_tot[2].set_xlim(*win)
        self.axs_tot[3].set_xlim(*win)

        self.fig_tot.canvas.draw_idle()

        # --- Twist window ---
        for i in range(3):
            self.lines_v_lin[i].set_data(t_list, v_lin_lists[i])
        self.ax_v_lin.set_xlim(*win)
        self.ax_v_lin.relim()
        self.ax_v_lin.autoscale_view(scalex=False, scaley=True)

        for i in range(3):
            self.lines_v_ang[i].set_data(t_list, v_ang_lists[i])
        self.ax_v_ang.set_xlim(*win)
        self.ax_v_ang.relim()
        self.ax_v_ang.autoscale_view(scalex=False, scaley=True)

        self.fig_v.canvas.draw_idle()

        # Flush events once at the very end to update all windows simultaneously
        self.fig.canvas.flush_events()

    # =========================
    # CALLBACKS
    # =========================
    def haption_pose_cb(self, msg):
        """Updates the real Cartesian orientation of the Virtuose handle."""
        # Assuming the orientation is a quaternion [x, y, z, w]
        q = msg.pose.orientation
        self.rot_haption = R.from_quat([q.x, q.y, q.z, q.w])
        
    def button_cb(self, msg):
        """Updates the clutching state from the Virtuose button."""
        self.is_clutching = msg.data

    def goal_names_cb(self, msg):
        """Updates the list of active goal names from the shared autonomy inference engine."""
        self.goal_names = msg.data.split(',')

    def goal_probs_cb(self, msg):
        """Updates the array of goal probabilities perfectly synchronized with the goal names."""
        self.goal_probs = list(msg.data)

    def user_policy_cb(self, msg):
        """Updates the flattened array of optimal spatial twists evaluated from the user's reference frame."""
        self.user_policies = list(msg.data)

    def goal_pose_cb(self, msg):
        """Updates the active goal pose + confidence for the position virtual fixture.

        Layout: [x, y, z, roll, pitch, yaw, confidence] in base_footprint.
        confidence is the belief of the active goal (0 during grasp execution).
        """
        if len(msg.data) >= 7:
            self.fix_goal_pos = np.array(msg.data[0:3])
            self.fix_goal_rot = R.from_euler('xyz', np.array(msg.data[3:6]), degrees=False)
            self.fix_confidence = float(msg.data[6])

    def target_cb(self, msg):
        """Updates the target Cartesian position and orientation of the TRIAGo arm."""
        if len(msg.data) >= 6:
            self.pos_target = np.array(msg.data[0:3])
            rpy = np.array(msg.data[3:6])
            self.rot_target = R.from_euler('xyz', rpy, degrees=False)

    def real_cb(self, msg):
        """Updates the real Cartesian position and orientation of the TRIAGo arm."""
        if len(msg.data) >= 15:
            self.pos_real = np.array(msg.data[0:3])
            rpy = np.array(msg.data[12:15])
            self.rot_real = R.from_euler('xyz', rpy, degrees=False)

    def vel_cb(self, msg):
        """Updates the current 6D spatial velocity (raw, unfiltered)."""
        # NOTE: velocity-measurement LPF removed on purpose. The previous
        # first-order filter added phase lag to every velocity-dependent force
        # (guidance damping, sync, etc.), which is itself a destabiliser in a
        # sampled-data haptic loop. We now use the raw device velocity directly.
        self.vel_haption = np.array([
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z
        ])

    def cbf_gradient_cb(self, msg):
        """Updates the control barrier function gradients mapped from the QP controller."""
        if len(msg.data) >= 13:
            self.grad_cbf_right = np.array(msg.data[1:7])

    def lambda_cb(self, msg):
        """Updates the active CBF slack variable representing obstacle proximity."""
        self.lambda_cbf = msg.data

    def joint_cb(self, msg):
        """Updates the current 6-DoF joint positions directly from the Haption encoders."""
        if len(msg.data) >= 6:
            self.joint_pos = np.array(msg.data[0:6])

    # =========================
    # FORCE COMPONENTS
    # =========================
    
    def compute_F_sync(self):
        """Calculates a 3D spring-damper tether force to keep the human operator synced with the real robot."""
        F_sync = np.zeros(6) 
        if self.pos_target is None or self.pos_real is None:
            return F_sync

        error_pos_tiago = self.pos_real - self.pos_target
        F_spring_tiago = self.Kp_sync * error_pos_tiago

        F_spring_haption = np.zeros(3)
        F_spring_haption[0] = -F_spring_tiago[0]
        F_spring_haption[1] = -F_spring_tiago[1]
        F_spring_haption[2] =  F_spring_tiago[2]

        # FIXED: Slice vel_haption to [0:3] to match the 3D spring array
        F_damped_haption = F_spring_haption - (self.Kd_sync * self.vel_haption[0:3])
        F_sync[0:3] = F_damped_haption
        
        return F_sync

    def compute_F_cbf(self):
        """Calculates, spatially saturates (tanh), and temporally filters (LPF) the repulsive 6D CBF wrench."""
        # 1. Handle free-space decay
        if self.lambda_cbf <= 0.0:
            # If we leave the obstacle, smoothly decay the residual force to zero instead of snapping
            self.f_cbf_filtered = (1.0 - self.alpha_cbf) * self.f_cbf_filtered
            return self.f_cbf_filtered

        # 2. Raw Force Calculation (TRIAGo Frame)
        F_cbf_triago = self.grad_cbf_right * self.lambda_cbf
        F_cbf_triago[0:3] *= self.K_cbf_force   
        F_cbf_triago[3:6] *= self.K_cbf_torque  

        # 3. Spatial Shaping: Tanh Soft-Saturation
        # This bends the infinite CBF spike into a smooth, bounded curve for human comfort
        F_cbf_triago[0:3] = self.MAX_CBF_FORCE * np.tanh(F_cbf_triago[0:3] / self.MAX_CBF_FORCE)
        F_cbf_triago[3:6] = self.MAX_CBF_TORQUE * np.tanh(F_cbf_triago[3:6] / self.MAX_CBF_TORQUE)

        # 4. Kinematic Mapping (TRIAGo to Haption Frame)
        F_cbf_raw_haption = np.zeros(6)
        F_cbf_raw_haption[0] = -F_cbf_triago[0]
        F_cbf_raw_haption[1] = -F_cbf_triago[1]
        F_cbf_raw_haption[2] =  F_cbf_triago[2]
        F_cbf_raw_haption[3] = -F_cbf_triago[3]
        F_cbf_raw_haption[4] = -F_cbf_triago[4]
        F_cbf_raw_haption[5] =  F_cbf_triago[5]
        
        # 5. Temporal Smoothing: First-Order Low-Pass Filter
        self.f_cbf_filtered = (self.alpha_cbf * F_cbf_raw_haption) + ((1.0 - self.alpha_cbf) * self.f_cbf_filtered)
        
        return self.f_cbf_filtered

    def _smoothstep(self, p, lo=0.70, hi=1.0):
        """C1-continuous ramp from 0 at p=lo to 1 at p=hi."""
        if p <= lo:
            return 0.0
        x = min((p - lo) / (hi - lo), 1.0)
        return 3.0 * x**2 - 2.0 * x**3
    
    # Expected layout of /shared_autonomy/goal_names (published by triago_control):
    #   "Red_Top,Red_Side,Blue_Top,Blue_Side,Platform_Place"
    #
    # /shared_autonomy/goal_probabilities : 5 floats aligned to the same order
    #   These form a flat probability simplex (sum to 1). Some goals may be
    #   excluded (probability = 0) by the belief estimator depending on the
    #   grasp state (e.g. grasped cylinder's goals = 0, Platform = 0 when empty).
    #
    # /shared_autonomy/user_policy : 30 floats (5 goals × 6 DOF)

    def compute_F_guide(self):
        """
        Continuous policy-merging guidance wrench (flat 5-goal simplex).

        The shared_autonomy node in triago_control publishes a FLAT belief
        distribution over 5 goals: Red_Top, Red_Side, Blue_Top, Blue_Side,
        Platform_Place. No hierarchy — just a proper simplex where excluded
        goals sit at probability 0.

        The blended reference twist is:
            pi_blend = Σ_k  P(k) · pi_k

        A continuous confidence gain (1 − normalised entropy, smoothstepped)
        fades the guidance in/out, and a final first-order LPF guarantees C0
        continuity even on noisy probability samples.
        """
        n_goals = len(self.goal_names) if self.goal_names else 0
        n_policies = len(self.user_policies)

        # Guard: need all inference data to have arrived with consistent sizes.
        if (n_goals == 0
                or len(self.goal_probs) != n_goals
                or n_policies != n_goals * 6):
            self.f_guide_filtered = (1.0 - self.alpha_guide) * self.f_guide_filtered
            return self.f_guide_filtered.copy()

        # ------------------------------------------------------------------ #
        # 1.  Parse flat arrays into per-goal probabilities and policies      #
        # ------------------------------------------------------------------ #
        probs = np.array(self.goal_probs)
        policies = np.array(self.user_policies).reshape(n_goals, 6)

        # ------------------------------------------------------------------ #
        # 2.  Belief-weighted blended policy (convex combination)             #
        # ------------------------------------------------------------------ #
        # Only blend over goals with non-zero probability (excluded goals
        # contribute nothing). This is mathematically equivalent to summing all
        # since excluded goals have P=0, but avoids numerical noise.
        pi_blend = probs @ policies  # (n_goals,) @ (n_goals, 6) -> (6,)

        # ------------------------------------------------------------------ #
        # 3.  Continuous confidence gain = 1 − normalised entropy             #
        # ------------------------------------------------------------------ #
        # Count only the ACTIVE goals (prob > 0) for entropy normalisation,
        # so that excluded goals (which are always 0) don't artificially inflate
        # entropy and suppress guidance when only 2-3 goals remain.
        active_mask = probs > 1e-12
        n_active = int(np.sum(active_mask))
        if n_active <= 1:
            # Single goal dominates (or nothing active): full confidence.
            confidence = 1.0
        else:
            H = -np.sum(probs[active_mask] * np.log(probs[active_mask]))
            H_norm = H / np.log(n_active)   # ∈ [0, 1]
            confidence = 1.0 - H_norm       # ∈ [0, 1], peaked → 1

        alpha = self._smoothstep(confidence,
                                 lo=self.GUIDE_CONF_LO,
                                 hi=self.GUIDE_CONF_HI)

        # ------------------------------------------------------------------ #
        # 4.  Velocity error in a consistent frame                            #
        # ------------------------------------------------------------------ #
        # pi_blend is in the robot/world frame (policies evaluated from
        # current_T_user in shared_autonomy). vel_haption arrives in the
        # Haption device frame → apply the 180° Z-flip to get robot frame.
        vel_h = self.vel_haption.copy()
        vel_robot = np.array([-vel_h[0], -vel_h[1],  vel_h[2],
                              -vel_h[3], -vel_h[4],  vel_h[5]])

        error_v_lin = pi_blend[0:3] - vel_robot[0:3]
        error_v_ang = pi_blend[3:6] - vel_robot[3:6]

        # ------------------------------------------------------------------ #
        # 4b. User-led gate: engagement × agreement                           #
        # ------------------------------------------------------------------ #
        # engagement -> 0 when the user is still (let them initiate, never drag a
        # passive hand); agreement -> 0 when the user moves against the suggested
        # direction (let them override a wrong guess without a fight). The product
        # multiplies the whole guidance wrench, so the system "listens" to the
        # user's twist instead of locking a goal and pulling.
        speed = float(np.linalg.norm(vel_robot[0:3]))
        pi_n = float(np.linalg.norm(pi_blend[0:3]))
        if speed > 1e-6 and pi_n > 1e-6:
            cos_align = float(np.dot(vel_robot[0:3], pi_blend[0:3]) / (speed * pi_n))
        else:
            cos_align = 0.0
        engage = self._smoothstep(speed, lo=self.LEAD_V_STILL, hi=self.LEAD_V_MOVE)
        agree = self._smoothstep(cos_align, lo=self.LEAD_COS_LO, hi=self.LEAD_COS_HI)
        lead_gain = engage * agree

        # ------------------------------------------------------------------ #
        # 5.  Viscous guidance wrench (robot frame), scaled by confidence     #
        #     AND the user-led gate.                                          #
        # ------------------------------------------------------------------ #
        gain = alpha * lead_gain
        F_guide_robot   = self.B_guide_lin * error_v_lin * gain
        Tau_guide_robot = self.B_guide_ang * error_v_ang * gain

        # Map robot → Haption frame (180° Z-flip)
        F_guide_raw = np.array([
            -F_guide_robot[0],   -F_guide_robot[1],    F_guide_robot[2],
            -Tau_guide_robot[0], -Tau_guide_robot[1],  Tau_guide_robot[2],
        ])

        # ------------------------------------------------------------------ #
        # 6.  Temporal smoothing (LPF) — final guarantee of C0 continuity     #
        # ------------------------------------------------------------------ #
        self.f_guide_filtered = (self.alpha_guide * F_guide_raw
                                 + (1.0 - self.alpha_guide) * self.f_guide_filtered)
        return self.f_guide_filtered.copy()

    def compute_F_fixture(self):
        """Position+orientation virtual fixture pulling the user toward the active goal.

        Unlike F_guide (viscous, velocity-based, vanishes at the goal), this is a
        POSITION spring: F = K·(goal_pose − real_pose), saturated and gated by the
        goal confidence. It does not weaken near the goal, so it lets the operator
        settle precisely at the grasp standoff against the CBF that pushes the
        gripper off the cylinder. Confidence gating (smoothstep over belief) keeps
        it silent in free space and when intent is ambiguous.
        """
        if (self.fix_goal_pos is None or self.pos_real is None
                or self.rot_real is None):
            self.f_fix_filtered = (1.0 - self.alpha_fix) * self.f_fix_filtered
            return self.f_fix_filtered.copy()

        gate = self._smoothstep(self.fix_confidence,
                                lo=self.FIX_CONF_LO, hi=self.FIX_CONF_HI)
        if gate <= 0.0:
            self.f_fix_filtered = (1.0 - self.alpha_fix) * self.f_fix_filtered
            return self.f_fix_filtered.copy()

        # Position spring in robot frame: points from the real EE toward the goal,
        # i.e. the direction the user must move the handle to drive the arm in.
        err_pos = self.fix_goal_pos - self.pos_real
        F_fix_robot = self.K_fix_force * err_pos
        F_fix_robot = self.MAX_FIX_FORCE * np.tanh(F_fix_robot / self.MAX_FIX_FORCE)

        # Orientation spring: R_err = R_goal · R_real^T (robot frame)
        err_rot_vec = R.from_matrix(
            self.fix_goal_rot.as_matrix() @ self.rot_real.as_matrix().T).as_rotvec()
        Tau_fix_robot = self.K_fix_torque * err_rot_vec
        Tau_fix_robot = self.MAX_FIX_TORQUE * np.tanh(Tau_fix_robot / self.MAX_FIX_TORQUE)

        # Map robot -> Haption frame (180° Z-flip), scaled by the confidence gate.
        F_fix_raw = np.array([
            -F_fix_robot[0],   -F_fix_robot[1],    F_fix_robot[2],
            -Tau_fix_robot[0], -Tau_fix_robot[1],  Tau_fix_robot[2],
        ]) * gate

        self.f_fix_filtered = (self.alpha_fix * F_fix_raw
                               + (1.0 - self.alpha_fix) * self.f_fix_filtered)
        return self.f_fix_filtered.copy()

    def compute_F_limit_warning(self):
        """Joint-limit vibration warning — DISABLED for now (returns zero wrench).

        The 75 Hz square-wave rumble is commented out pending a redesign of the
        vibration pattern. Kept as a no-op so the force-superposition pipeline and
        the 'Limit' plot trace stay structurally intact.
        """
        F_vib = np.zeros(6)
        return F_vib

        # --- VIBRATION SIGNAL (commented out) ---
        # dist_to_min = self.joint_pos - self.joint_min
        # dist_to_max = self.joint_max - self.joint_pos
        # min_margin = np.min(np.concatenate([dist_to_min, dist_to_max]))
        #
        # if min_margin <= self.LIMIT_OUTER:
        #     if min_margin <= self.LIMIT_INNER:
        #         amplitude = self.AMP_MAX
        #     else:
        #         ratio = (self.LIMIT_OUTER - min_margin) / (self.LIMIT_OUTER - self.LIMIT_INNER)
        #         amplitude = self.AMP_MIN + ratio * (self.AMP_MAX - self.AMP_MIN)
        #     self.vib_toggle *= -1.0
        #     F_vib[3] = amplitude * self.vib_toggle
        #     F_vib[4] = amplitude * self.vib_toggle
        #     F_vib[5] = amplitude * self.vib_toggle
        # return F_vib

    # =========================
    # MAIN LOOP
    # =========================
    def control_loop(self):
        """Aggregates forces, tracks/enforces passivity, applies safety clippings, publishes, and buffers data."""
        f_sync = self.compute_F_sync()
        f_cbf = self.compute_F_cbf()
        f_guide = self.compute_F_guide()
        f_fix = self.compute_F_fixture()
        f_vib = self.compute_F_limit_warning()

        # Calculate the normal running force
        f_total_normal = f_sync + f_cbf + f_guide + f_fix + f_vib

        # --- AUTHORITY CAP -------------------------------------------------- #
        # Bound the MAGNITUDE of the assistive wrench so the autonomy can never
        # overpower the operator. Scaling the whole vector preserves the relative
        # proportions of sync/cbf/guide/fix (the "feel" we tuned) while limiting
        # the total push to MAX_TOTAL_FORCE / MAX_TOTAL_TORQUE.
        f_norm = np.linalg.norm(f_total_normal[0:3])
        if f_norm > self.MAX_TOTAL_FORCE:
            f_total_normal[0:3] *= self.MAX_TOTAL_FORCE / f_norm
        t_norm = np.linalg.norm(f_total_normal[3:6])
        if t_norm > self.MAX_TOTAL_TORQUE:
            f_total_normal[3:6] *= self.MAX_TOTAL_TORQUE / t_norm

        # ========================================================
        # CLUTCHING ARCHITECTURE & ALIGNMENT GUIDANCE
        # ========================================================
        if self.is_clutching:
            # 1. Edge Detection: The exact millisecond the clutch is pressed
            if not self.was_clutching_last_frame:
                # Save the total force and immediately halve it for cognitive grounding
                self.f_clutch_frozen = f_total_normal / 2.0
                self.was_clutching_last_frame = True
            
            # 2. Apply the frozen 50% wrench
            f_total = self.f_clutch_frozen.copy()
            
            # 3. Haptic Alignment Guidance (Orientation Only)
            if self.rot_haption is not None and self.rot_target is not None:
                
                # Calculate the rotation error pulling the HAPTION HANDLE toward the FROZEN TARGET
                # Math: R_error = R_target * R_haption^T
                error_rot_matrix = self.rot_target.as_matrix() @ self.rot_haption.as_matrix().T
                error_rot_vec = R.from_matrix(error_rot_matrix).as_rotvec()
                
                # Calculate torque in the base frame
                tau_align_base = self.K_align * error_rot_vec
                
                # Map to Haption frame (180 deg Z-flip if required by your kinematic setup)
                tau_align_haption = np.zeros(3)
                tau_align_haption[0] = -tau_align_base[0]
                tau_align_haption[1] = -tau_align_base[1]
                tau_align_haption[2] =  tau_align_base[2]
                
                # 4. Joint Limit Compromise (Proximity Fade)
                dist_to_min = self.joint_pos - self.joint_min
                dist_to_max = self.joint_max - self.joint_pos
                min_margin = np.min(np.concatenate([dist_to_min, dist_to_max]))
                
                # Fade torque to zero if within 0.35 rad of a physical limit
                fade_margin = 0.35 
                if min_margin < fade_margin:
                    scale = max(0.0, min_margin / fade_margin)
                    tau_align_haption *= scale 
                
                # Add the saturated alignment torque to the frozen wrench
                f_total[3:6] += tau_align_haption

        else:
            # 5. Normal operation when unclutched
            f_total = f_total_normal
            self.was_clutching_last_frame = False

        # GLOBAL DAMPING. IF PLUG HIGH VALUES, YOU GET INSTABILITY DUE TO ZOH. NEEDED TO AVOID LOW FREQ OSCILLATION.
        Kd_global_lin = 0.7  
        Kd_global_ang = 0.1
        
        f_total[0:3] -= Kd_global_lin * self.vel_haption[0:3]
        f_total[3:6] -= Kd_global_ang * self.vel_haption[3:6]

        # ========================================================
        # CLIPPING & PUBLISHING
        # ========================================================
        f_total[0:3] = np.clip(f_total[0:3], -self.MAX_FORCE, self.MAX_FORCE)
        f_total[3:6] = np.clip(f_total[3:6], -self.MAX_TORQUE, self.MAX_TORQUE)

        msg = Wrench()
        msg.force.x, msg.force.y, msg.force.z = float(f_total[0]), float(f_total[1]), float(f_total[2])
        msg.torque.x, msg.torque.y, msg.torque.z = float(f_total[3]), float(f_total[4]), float(f_total[5])
        self.force_pub.publish(msg)

        # Buffer Data for Plotting
        t = time.time() - self.start_time
        guide_comb = f_guide + f_fix   # guidance share = viscous guide + position fixture
        components = {'Sync': f_sync, 'CBF': f_cbf, 'Guide': guide_comb, 'Limit': f_vib}

        # Per-source contribution share (% of summed component magnitudes).
        nF = {'Sync': np.linalg.norm(f_sync[0:3]),
              'CBF':  np.linalg.norm(f_cbf[0:3]),
              'Guide': np.linalg.norm(guide_comb[0:3])}
        nT = {'Sync': np.linalg.norm(f_sync[3:6]),
              'CBF':  np.linalg.norm(f_cbf[3:6]),
              'Guide': np.linalg.norm(guide_comb[3:6])}
        sF = sum(nF.values())
        sT = sum(nT.values())

        # Lock the buffer modification to prevent matplotlib from reading a partially updated structure
        with self.plot_lock:
            self.t_data.append(t)
            # Per-component wrench (Superposition window)
            for cat, force_vec in components.items():
                for i in range(3):
                    self.f_data[cat]['F'][i].append(force_vec[i])
                    self.f_data[cat]['T'][i].append(force_vec[i + 3])
            # Final published total wrench (F_total window)
            for i in range(3):
                self.ftot_data['F'][i].append(f_total[i])
                self.ftot_data['T'][i].append(f_total[i + 3])
            # Contribution shares (%)
            for k in ['Sync', 'CBF', 'Guide']:
                self.pct_force[k].append(100.0 * nF[k] / sF if sF > 1e-9 else 0.0)
                self.pct_torque[k].append(100.0 * nT[k] / sT if sT > 1e-9 else 0.0)
            # 6D Velocity components (Twist window)
            for i in range(3):
                self.v_lin_data[i].append(self.vel_haption[i])
                self.v_ang_data[i].append(self.vel_haption[i + 3])

def main(args=None):
    """Initializes ROS, spins the node on a daemon thread, and drives Matplotlib updates safely on the main thread."""
    rclpy.init(args=args)
    node = HapticForceManager()

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