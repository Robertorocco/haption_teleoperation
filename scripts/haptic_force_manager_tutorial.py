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

        # --- NEW: Passivity Smoothing Parameters ---
        # (velocity-measurement LPF removed: vel_haption is now used raw)
        self.MAX_PC_FORCE = 5.0        # N (Max allowed PC damping force)
        self.MAX_PC_TORQUE = 0.5       # Nm (Max allowed PC damping torque)

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

        # --- NEW: Passivity Observer Variables ---
        self.energy_observer = 0.0
        self.ENABLE_PASSIVITY_CONTROL = False  # Toggle to apply or ignore the PC damping force

        # --- Tunable Force Parameters ---
        self.Kp_sync = 10.0#15.0  
        self.Kd_sync = 0.0  #if global damping is added, set this to 0 to avoid overdamping
        self.K_cbf_force = 2.0   
        self.K_cbf_torque = 0.1  
        self.MAX_FORCE = 10.0 
        self.MAX_TORQUE = 1.0 

        # --- Data Buffers & Synchronization ---
        self.plot_lock = threading.Lock()
        self.plot_window_sec = 10.0
        self.buffer_size = int(150 * self.plot_window_sec)
        self.t_data = deque(maxlen=self.buffer_size)
        self.start_time = time.time()
        self.e_data = deque(maxlen=self.buffer_size) # NEW: Energy buffer
        self.f_pc_data = deque(maxlen=self.buffer_size) # NEW: Tracks PC Force magnitude
        self.t_pc_data = deque(maxlen=self.buffer_size) # NEW: Tracks PC Torque magnitude (Nm)
        self.v_lin_data = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.v_ang_data = [deque(maxlen=self.buffer_size) for _ in range(3)]
        
        self.f_data = {
            'Total': {'F': [deque(maxlen=self.buffer_size) for _ in range(3)], 'T': [deque(maxlen=self.buffer_size) for _ in range(3)]},
            'Sync':  {'F': [deque(maxlen=self.buffer_size) for _ in range(3)], 'T': [deque(maxlen=self.buffer_size) for _ in range(3)]},
            'CBF':   {'F': [deque(maxlen=self.buffer_size) for _ in range(3)], 'T': [deque(maxlen=self.buffer_size) for _ in range(3)]},
            'Guide': {'F': [deque(maxlen=self.buffer_size) for _ in range(3)], 'T': [deque(maxlen=self.buffer_size) for _ in range(3)]},
            'Limit': {'F': [deque(maxlen=self.buffer_size) for _ in range(3)], 'T': [deque(maxlen=self.buffer_size) for _ in range(3)]}
        }

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
        
        self.force_pub = self.create_publisher(Wrench, 'virtuose/force_cmd', 10)

        # --- Timers ---
        self.dt = 1.0 / 150.0
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.setup_plot()
        self.get_logger().info("Haptic Force Manager started. Max Freq (75Hz) Square Wave Vibration enabled.")

    # =========================
    # PLOT SETUP & UPDATE
    # =========================
    def setup_plot(self):
        """Initializes the 5x2 grid of Matplotlib subplots for live drawing."""
        plt.ion()
        self.fig, self.axs = plt.subplots(5, 2, figsize=(12, 11))
        self.fig.canvas.manager.set_window_title('Haptic Force Superposition')
        
        self.lines = {}
        categories = ['Total', 'Sync', 'CBF', 'Guide', 'Limit']
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
            self.axs[4, col].set_xlabel("Time (s)")
            
        self.fig.tight_layout()
       # ========================================================
        # --- Passivity Architecture Window (3 Stacked Subplots) ---
        # ========================================================
        # Create a figure with 3 subplots. Top gets more vertical space.
        self.fig_e, (self.ax_e, self.ax_fpc, self.ax_tpc) = plt.subplots(
            3, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1, 1]}
        )
        self.fig_e.canvas.manager.set_window_title('Passivity Architecture')
        
        # --- TOP SUBPLOT: Energy Observer ---
        self.ax_e.set_title("Real-Time Energy Flow (PO)", fontsize=12, fontweight='bold')
        self.ax_e.set_ylabel("Energy (Joules)")
        
        self.ax_e.axhspan(0, 1000, color='green', alpha=0.1, label='Passive Region (Safe)')
        self.ax_e.axhspan(-1000, 0, color='red', alpha=0.1, label='Active Region')
        self.ax_e.axhline(0, color='black', linestyle='--', linewidth=1.5)
        
        self.line_e, = self.ax_e.plot([], [], color='purple', linewidth=2.5, label='Observed Energy')
        self.ax_e.legend(loc='upper right')
        
        # --- MIDDLE SUBPLOT: PC Force ---
        self.ax_fpc.set_ylabel("PC Force (N)")
        self.ax_fpc.grid(True, linestyle='--', alpha=0.6)
        
        # Continuous blue line
        self.line_fpc, = self.ax_fpc.plot([], [], color='blue', linewidth=2.0, linestyle='-', label='||F_pc||')
        self.ax_fpc.legend(loc='upper right')
        
        # --- BOTTOM SUBPLOT: PC Torque ---
        self.ax_tpc.set_ylabel("PC Torque (Nm)")
        self.ax_tpc.set_xlabel("Time (s)")
        self.ax_tpc.grid(True, linestyle='--', alpha=0.6)
        
        # Continuous blue line
        self.line_tpc, = self.ax_tpc.plot([], [], color='blue', linewidth=2.0, linestyle='-', label='||T_pc||')
        self.ax_tpc.legend(loc='upper right')
        self.fig_e.tight_layout()

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
            e_list = list(self.e_data) # NEW: Extract energy data
            fpc_list = list(self.f_pc_data) # NEW: Extract PC Force
            tpc_list = list(self.t_pc_data) # NEW: Extract PC Torque
            f_lists = {
                cat: {
                    'F': [list(self.f_data[cat]['F'][i]) for i in range(3)],
                    'T': [list(self.f_data[cat]['T'][i]) for i in range(3)]
                } for cat in ['Total', 'Sync', 'CBF', 'Guide', 'Limit']
            }

            v_lin_lists = [list(self.v_lin_data[i]) for i in range(3)]
            v_ang_lists = [list(self.v_ang_data[i]) for i in range(3)]

        # 2. Update Matplotlib outside the lock to prevent stalling the ROS 2 loop
        current_t = t_list[-1]
        
        for row, cat in enumerate(['Total', 'Sync', 'CBF', 'Guide', 'Limit']):
            # Update Forces
            for i in range(3):
                self.lines[cat]['F'][i].set_data(t_list, f_lists[cat]['F'][i])
            self.axs[row, 0].set_xlim(current_t - self.plot_window_sec, current_t)
            self.axs[row, 0].relim()
            self.axs[row, 0].autoscale_view(scalex=False, scaley=True)

            # Update Torques
            for i in range(3):
                self.lines[cat]['T'][i].set_data(t_list, f_lists[cat]['T'][i])
            self.axs[row, 1].set_xlim(current_t - self.plot_window_sec, current_t)
            self.axs[row, 1].relim()
            self.axs[row, 1].autoscale_view(scalex=False, scaley=True)

        ## --- Update Passivity Windows ---
        self.line_e.set_data(t_list, e_list)
        self.line_fpc.set_data(t_list, fpc_list) 
        self.line_tpc.set_data(t_list, tpc_list) 
        
        current_t = t_list[-1]
        
        # 1. Update Top Subplot (Energy)
        self.ax_e.set_xlim(current_t - self.plot_window_sec, current_t)
        if len(e_list) > 0:
            min_e, max_e = min(e_list), max(e_list)
            pad = max(abs(max_e - min_e) * 0.1, 0.1)
            self.ax_e.set_ylim(min(min_e - pad, -0.2), max(max_e + pad, 0.2))
            
        # 2. Update Middle Subplot (Force)
        self.ax_fpc.set_xlim(current_t - self.plot_window_sec, current_t)
        if len(fpc_list) > 0:
            max_f = max(fpc_list)
            self.ax_fpc.set_ylim(-0.1, max(max_f * 1.2, 1.0))

        # 3. Update Bottom Subplot (Torque)
        self.ax_tpc.set_xlim(current_t - self.plot_window_sec, current_t)
        if len(tpc_list) > 0:
            max_t = max(tpc_list)
            self.ax_tpc.set_ylim(-0.01, max(max_t * 1.2, 0.1))
            
        # ========================================================
        # --- Update Twist Window ---
        # ========================================================
        current_t = t_list[-1]
        
        # 1. Update Linear Velocity Plot
        for i in range(3):
            self.lines_v_lin[i].set_data(t_list, v_lin_lists[i])
            
        self.ax_v_lin.set_xlim(current_t - self.plot_window_sec, current_t)
        self.ax_v_lin.relim()
        self.ax_v_lin.autoscale_view(scalex=False, scaley=True)

        # 2. Update Angular Velocity Plot
        for i in range(3):
            self.lines_v_ang[i].set_data(t_list, v_ang_lists[i])
            
        self.ax_v_ang.set_xlim(current_t - self.plot_window_sec, current_t)
        self.ax_v_ang.relim()
        self.ax_v_ang.autoscale_view(scalex=False, scaley=True)

        self.fig_v.canvas.draw_idle()
        
        # Flush events once at the very end to update all 3 windows simultaneously
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
        # 5.  Viscous guidance wrench (robot frame), scaled by confidence     #
        # ------------------------------------------------------------------ #
        F_guide_robot   = self.B_guide_lin * error_v_lin * alpha
        Tau_guide_robot = self.B_guide_ang * error_v_ang * alpha

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

    def compute_F_limit_warning(self):
        """Calculates a 75Hz square wave rumble with variable intensity inside a specific boundary zone."""
        F_vib = np.zeros(6)
        
        dist_to_min = self.joint_pos - self.joint_min
        dist_to_max = self.joint_max - self.joint_pos
        min_margin = np.min(np.concatenate([dist_to_min, dist_to_max]))

        if min_margin <= self.LIMIT_OUTER:
            
            if min_margin <= self.LIMIT_INNER:
                amplitude = self.AMP_MAX
            else:
                ratio = (self.LIMIT_OUTER - min_margin) / (self.LIMIT_OUTER - self.LIMIT_INNER)
                amplitude = self.AMP_MIN + ratio * (self.AMP_MAX - self.AMP_MIN)
            
            self.vib_toggle *= -1.0
            
            F_vib[3] = amplitude * self.vib_toggle
            F_vib[4] = amplitude * self.vib_toggle
            F_vib[5] = amplitude * self.vib_toggle

        return F_vib

    # =========================
    # MAIN LOOP
    # =========================
    def control_loop(self):
        """Aggregates forces, tracks/enforces passivity, applies safety clippings, publishes, and buffers data."""
        f_sync = self.compute_F_sync()
        f_cbf = self.compute_F_cbf()
        f_guide = self.compute_F_guide()
        f_vib = self.compute_F_limit_warning()

        # Calculate the normal running force
        f_total_normal = f_sync + f_cbf + f_guide + f_vib

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
        # PASSIVITY OBSERVER (PO)
        # ========================================================
        # Power = - (Wrench dot Twist). 6D dot product.
        power = -np.dot(f_total, self.vel_haption)
        self.energy_observer += power * self.dt

        # ========================================================
        # FULL 6D PASSIVITY CONTROLLER (PC)
        # ========================================================
        f_pc = np.zeros(6) 

        if self.energy_observer < 0.0:
            v_squared = np.dot(self.vel_haption, self.vel_haption)
            
            # Increased threshold to prevent division by near-zero
            if v_squared > 1e-4: 
                beta = -self.energy_observer / (v_squared * self.dt)
                
                # 1. Calculate raw damping required
                f_pc_raw = -beta * self.vel_haption
                
                # 2. SATURATE THE BRAKE: Prevent massive hammer blows
                f_pc[0:3] = np.clip(f_pc_raw[0:3], -self.MAX_PC_FORCE, self.MAX_PC_FORCE)
                f_pc[3:6] = np.clip(f_pc_raw[3:6], -self.MAX_PC_TORQUE, self.MAX_PC_TORQUE)
                
                if self.ENABLE_PASSIVITY_CONTROL:
                    f_total += f_pc
            
            # 3. SMOOTH RESET: Decay the energy instead of snapping to 0.0
            # This prevents the PC from toggling on and off every other tick
            self.energy_observer *= 0.1 
            
        else:
            self.energy_observer *= 0.99

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
        components = {'Total': f_total, 'Sync': f_sync, 'CBF': f_cbf, 'Guide': f_guide, 'Limit': f_vib}
        
        # Lock the buffer modification to prevent matplotlib from reading a partially updated structure
        with self.plot_lock:
            self.t_data.append(t)
            self.e_data.append(self.energy_observer)
            # Calculate and store the magnitudes of the PC damping wrench
            self.f_pc_data.append(np.linalg.norm(f_pc[0:3])) # Linear Force (N)
            self.t_pc_data.append(np.linalg.norm(f_pc[3:6])) # Angular Torque (Nm)
            # 2. Append Force Dictionaries (5 force categories per frame)
            for cat, force_vec in components.items():
                for i in range(3):
                    self.f_data[cat]['F'][i].append(force_vec[i])     
                    self.f_data[cat]['T'][i].append(force_vec[i+3])

            # 3. Append 6D Velocity Components (1 loop per frame)
            # CAUTION: Ensure this is NOT indented inside the force loop above!
            for i in range(3):
                self.v_lin_data[i].append(self.vel_haption[i])     # Indices 0, 1, 2
                self.v_ang_data[i].append(self.vel_haption[i+3])   # Indices 3, 4, 5

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