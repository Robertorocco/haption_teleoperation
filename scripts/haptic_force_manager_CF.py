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
from geometry_msgs.msg import Pose  # server publishes geometry_msgs/Pose on virtuose/pose
import matplotlib.pyplot as plt
import matplotlib

# Single source of truth for the experiment-condition selector (2x3 study).
import triago_control.qp_controller.config as cfg

# Set backend to avoid blocking the ROS spin loop
matplotlib.use('TkAgg') 

class HapticForceManager(Node):
    def __init__(self):
        """Initializes the Haptic Force Manager, publishers, subscribers, thread locks, and plot buffers."""
        super().__init__('haptic_force_manager')

        # Fail loudly if launched under the wrong study condition. This is the
        # CLUTCH "Guided feedback" manager: assistive haptic FORCES on, reference
        # blending OFF (Virtual Fixture, feedback-only channel).
        cfg.validate_condition('haptic_force_manager_CF',
                               control_mode=cfg.CLUTCH, feedback=True, blending=False)

        # --- State Variables ---
        self.pos_target = None
        self.rot_target = None
        self.pos_real = None
        self.rot_real = None
        self.vel_real = np.zeros(3)   # right EE linear velocity (from ee_real[3:6])
        self.vel_haption = np.zeros(6)  

        # --- CBF State Variables ---
        self.grad_cbf_right = np.zeros(6)
        self.lambda_cbf = 0.0
        self.lambda_cbf_f = 0.0       # LPF'd lambda for smooth damping scaling
        self.CBF_LAMBDA_ALPHA = 0.05  # LPF coefficient on lambda (low = very smooth)
        self.CBF_GAIN_BOOST = 1.2     # CBF feedback 1.2x stronger

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
        # F_guide: VELOCITY-FIELD guidance (not a position spring).
        #
        # Rationale (bug fix): the previous approach turned the tanh-saturated
        # policy velocity into a position offset (pi_blend * lookahead, capped),
        # which made F_guide a NEAR-CONSTANT ~4.5 N push whenever the EE was more
        # than ~10-15 cm from the goal. On an impedance device with no damping
        # (DEBUG_ONLY_GUIDE bypasses the global damper) a constant force does not
        # "drive the handle to a target" — it just keeps accelerating a lightly
        # held handle (felt as "too strong / moves on its own / never settles"),
        # because the force depends only on the ROBOT state and never on the HAND.
        #
        # New model: render the blended policy as a velocity FIELD the handle
        # should follow:
        #       F = D_guide * ( v_field_haption - v_handle )
        # where v_field_haption is the policy twist pi_blend mapped into the
        # Haption frame. Properties that fix the bug:
        #   * produces a force when the handle is still (D * v_field), so it
        #     initiates motion toward the goal;
        #   * INTRINSICALLY DAMPED — as the handle accelerates up to v_field the
        #     force fades to zero, so it can never run away / fling the handle;
        #   * a passive operator lets the handle cruise at exactly v_field, so the
        #     teleop reference integrates the same path the POLICY_BELIEF_TEST=True
        #     mode commands directly -> the gripper is driven to the goal the same
        #     way, but through the user's hand;
        #   * vanishes at the goal (pi_blend -> 0), so it settles instead of
        #     pushing the handle past it.
        self.D_guide_lin = 33.6   # Ns/m  velocity-field tracking gain (translation) [1.2x]
        self.D_guide_ang = 0.54   # Nms/rad velocity-field tracking gain (rotation) [1.2x]
        self.MAX_GUIDE_FORCE  = 4.2   # N   saturation on the guidance force  [1.2x]
        self.MAX_GUIDE_TORQUE = 0.10  # Nm  saturation on the guidance torque [reduced further]
        # ^ scaled 1.2x vs the first pass: near the goal the policy twist (and thus
        #   v_field) is small, so the raw guidance struggled to move the handle.

        # --- Proximity gate on F_guide -------------------------------------- #
        # The velocity-field policy is LARGE far from the goal (tanh-saturated
        # v_geo) and small near it, so far away the guidance would drag the handle
        # toward a goal pose that is still ill-determined and swinging (orientation
        # candidate flips, azimuth changes) → erratic "exploding" guidance. Fade
        # the guidance to ZERO beyond GUIDE_PROX_FAR (device totally free) and ramp
        # to full only by GUIDE_PROX_NEAR, measured as the distance from the
        # REFERENCE (pos_target) to the active goal (fix_goal_pos). Assistance then
        # only engages once the user has steered the reference close enough that
        # the goal is committed and stable.
        self.GUIDE_PROX_FAR  = 0.60   # m — beyond this: device free (gate = 0); UNIFIED across all guidance cells
        self.GUIDE_PROX_NEAR = 0.10   # m — at/below this: full guidance (gate = 1)

        # DEBUG: set True to output ONLY F_guide (isolate guidance for testing).
        # Set to False to re-enable the full superposition path (F_sync active;
        # F_cbf still EVALUATED every tick for telemetry/plots but its
        # contribution to f_total_normal is commented out below — see
        # "F_CBF DISABLED" in control_loop).
        self.DEBUG_ONLY_GUIDE = False

        # --- Continuous Policy-Merging (Belief-Weighted Blend) Parameters ---
        # Instead of a winner-take-all gate that snaps pi_ref from one goal's
        # policy to another (and creates force discontinuities), we blend ALL
        # leaf policies by their joint hierarchical probability. The blended
        # reference twist is a smooth function of the beliefs, so it never jumps.
        #
        # confidence gain : continuous fade-in of the whole guidance wrench based
        #                   on the active-goal belief b_max (max posterior, from
        #                   active_goal_pose[6]) -- the SAME belief function JF/JFB
        #                   use, shaped by a smoothstep. UNIFIED across all cells.
        self.GUIDE_CONF_LO   = 0.30   # below this confidence -> transparent; UNIFIED across all guidance cells
        self.GUIDE_CONF_HI   = 0.90   # at/above this confidence -> full guidance gain
        # Temporal smoothing of the final guidance wrench. Guarantees C0 continuity
        # even if a probability sample arrives noisy; removes any residual stepping.
        self.alpha_guide     = 0.15   # LPF coefficient (lower = smoother, more lag)
        self.f_guide_filtered = np.zeros(6)

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
        self.K_fix_force  = 38.016    # N/m   position spring toward the grasp pose [+10% again]
        self.K_fix_torque = 0.2376    # Nm/rad orientation spring toward grasp orientation [+10% again]
        self.MAX_FIX_FORCE  = 5.7024  # N     saturation [+10% again]
        self.MAX_FIX_TORQUE = 0.396   # Nm    saturation [+10% again]
        self.FIX_CONF_LO = 0.55       # below this belief -> fixture OFF
        self.FIX_CONF_HI = 0.85       # at/above this belief -> full fixture
        self.alpha_fix = 0.15         # LPF on the fixture wrench (C0 continuity)
        self.f_fix_filtered = np.zeros(6)
        # --- Near-goal ORIENTATION assist (help the user rotate the gripper) ---
        # Rotating the handle to align the gripper is hard from the operator's POV,
        # so VERY CLOSE to the goal we strengthen the fixture torque (~20%) and add
        # angular damping so it is "strong but damped" — it drives small residual
        # orientation errors (~10 deg) to alignment without oscillating. Ramped by
        # the EE→goal distance (full within *_NEAR, off beyond *_FAR).
        self.FIX_TORQUE_NEAR = 0.05        # m — full orientation assist within 5 cm of the goal
        self.FIX_TORQUE_FAR  = 0.12        # m — no assist beyond 12 cm
        self.FIX_TORQUE_NEAR_BOOST = 0.20  # +20% torque gain/saturation at the goal
        self.K_FIX_TORQUE_DAMP = 0.06      # Nms/rad angular damping (kills oscillation)

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

        # --- Joint-limit "clutch advice" one-shot burst ---
        # When a joint enters LIMIT_OUTER, fire a SINGLE fixed-amplitude buzz that
        # lasts LIMIT_VIB_DURATION and then goes silent. It will NOT fire again
        # until the operator completes a full clutch cycle (press -> release), so
        # the burst is unambiguously interpreted as "you should clutch now".
        self.LIMIT_VIB_DURATION = 1.0   # s   burst length
        self.LIMIT_VIB_AMP = 0.07       # Nm  fixed torque amplitude of the burst
        self.limit_vib_armed = True     # ready to fire (re-armed by a clutch cycle)
        self.limit_vib_active = False   # a burst is currently playing
        self.limit_vib_start_time = 0.0 # wall-clock start of the active burst
        self._vib_clutch_prev = False   # previous clutch state (for cycle edge detect)

        # --- Tunable Force Parameters ---
        self.Kp_sync = 30.0      # N/m    [UNIFIED across all clutch cells: same sync spring everywhere]
        self.Kd_sync = 0.0  #if global damping is added, set this to 0 to avoid overdamping
        self.Kp_sync_ang = 0.9   # Nm/rad [UNIFIED: orientation sync spring (handle -> real EE orientation)]

        # --- Adaptive sync authority (anti reference-runaway) ---
        # When the cartesian REFERENCE drifts far from the REAL EE (the robot can't
        # follow the guidance/CBF push), the pushing forces are attenuated so the
        # sync spring takes over and reins the reference back in. The sync SHARE of
        # the wrench grows linearly to SYNC_SHARE_AT_FULL at the "full" error
        # (SYNC_FULL_POS_ERR / SYNC_FULL_ANG_ERR) and keeps growing up to
        # SYNC_SHARE_CAP beyond it. Sync itself is never attenuated, so free-space
        # tracking (small error -> share ~0) is unchanged.
        self.SYNC_SHARE_AT_FULL = 0.5      # sync = 50% of the wrench at the full error
        self.SYNC_FULL_POS_ERR  = 0.30     # m   "full" reference-real position error
        self.SYNC_FULL_ANG_ERR  = np.pi / 2  # rad "full" reference-real orientation error (90 deg)
        self.SYNC_SHARE_CAP     = 0.85     # max sync share beyond the full error

        # --- Grasp-execution coupling ---
        # During autonomous grasp execution (shared_autonomy drives the arm,
        # /shared_autonomy/grasp_active = True) the user input is ignored, but we
        # want the operator to FEEL what the arm is doing. We then output a strong,
        # pure F_sync (everything else disabled) so the handle is firmly pulled to
        # track the EE motion. GRASP_SYNC_BOOST scales the sync stiffness up; the
        # final MAX_FORCE/MAX_TORQUE clip still bounds it.
        self.grasp_active = False
        self._grasp_start_pos = None   # EE position at the start of the grasp (for velocity-following)
        self.GRASP_SYNC_BOOST = 6.0    # 2x stronger than before so user clearly feels the motion
        # Grasp-follow tether: a gentler POSITION spring (so the approach no longer
        # "advances too much") plus a VELOCITY-following term that renders the EE's
        # instantaneous motion — this is what makes the LIFT (pure +Z) clearly felt
        # even when the cumulative position error in Z is small (e.g. after a
        # top-grasp approach that first moved down then lifts back up).
        self.GRASP_FOLLOW_KP = 30.0    # N/m   position tether toward where the arm is now
        self.GRASP_FOLLOW_KD = 160.0   # Ns/m  velocity-following (feels direction/speed of travel)
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
        self.MAX_TOTAL_FORCE  = 10.0  # N  [UNIFIED cap = device clip, same everywhere]
        self.MAX_TOTAL_TORQUE = 1.0   # Nm [UNIFIED cap = device clip, same everywhere]

        # --- Data Buffers & Synchronization ---
        self.plot_lock = threading.Lock()
        self.plot_window_sec = 10.0
        self.buffer_size = int(150 * self.plot_window_sec)
        self.t_data = deque(maxlen=self.buffer_size)
        self.start_time = time.time()

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

        # Shared-autonomy inference frequency tracker (from goal_probs arrival rate)
        self._sa_freq_data = deque(maxlen=self.buffer_size)
        self._sa_last_time = None
        self._sa_freq_lpf = 0.0

        # Own node frequency tracker (from control_loop call rate)
        self._own_freq_data = deque(maxlen=self.buffer_size)
        self._own_last_time = None
        self._own_freq_lpf = 0.0
        self.create_subscription(Float64MultiArray, '/arm_right/cartesian_reference', self.target_cb, 10)
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self.real_cb, 10)
        self.create_subscription(Twist, 'virtuose/velocity', self.vel_cb, 10)
        self.create_subscription(Float64MultiArray, '/collision_constraints', self.cbf_gradient_cb, 10)
        self.create_subscription(Float64MultiArray, '/qp_debug/lambda_cbf', self.lambda_cb, 10)
        self.create_subscription(Float64MultiArray, 'virtuose/articular_position', self.joint_cb, 10)
        #self.create_subscription(Float64MultiArray, '/shared_autonomy/assistive_reference', self.assist_cb, 10)
        self.create_subscription(Bool, 'virtuose/button_right', self.button_cb, 10)
        self.create_subscription(Pose, 'virtuose/pose', self.haption_pose_cb, 10)
        #Unified Inference State Subscribers
        self.create_subscription(String, '/shared_autonomy/goal_names', self.goal_names_cb, 10)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/goal_probabilities', self.goal_probs_cb, 10)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/user_policy', self.user_policy_cb, 10)
        # Active goal pose + confidence for the position virtual fixture
        self.create_subscription(Float64MultiArray, '/shared_autonomy/active_goal_pose', self.goal_pose_cb, 10)
        # Grasp-execution flag: when True, output strong pure F_sync (track the EE)
        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_cb, 10)
        # Arm switch: follow the active arm for EE state slicing and reference topic
        self.active_arm = 'right'
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)
        self.create_subscription(Float64MultiArray, '/arm_left/cartesian_reference', self.target_cb_left, 10)
        
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
        self.fig_tot, self.axs_tot = plt.subplots(5, 1, figsize=(9, 12))
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

        # Subplot 4: Haptic Force Manager own frequency
        ax = self.axs_tot[4]
        ax.set_title("Haptic Force Manager Frequency (Hz)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Hz")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 180)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(150, color='g', linestyle='--', linewidth=1.0, alpha=0.7, label='target 150Hz')
        self.line_sa_freq, = ax.plot([], [], color='#9467bd', linewidth=1.5, label='HFM freq')
        ax.legend(loc='upper left', fontsize=8)

        self.fig_tot.tight_layout()
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

        # Freq subplot (own node frequency)
        with self.plot_lock:
            own_freq_list = list(self._own_freq_data)
        if own_freq_list:
            # Trim to the minimum length of both arrays to avoid numpy broadcast
            # mismatch (t_data and _own_freq_data are pushed in the same loop tick
            # but under a lock that update_plot may snapshot between the two appends).
            n = min(len(t_list), len(own_freq_list))
            self.line_sa_freq.set_data(t_list[:n], own_freq_list[:n])
        self.axs_tot[4].set_xlim(*win)

        self.fig_tot.canvas.draw_idle()

        # Flush events once at the very end to update all windows simultaneously
        self.fig.canvas.flush_events()

    # =========================
    # CALLBACKS
    # =========================
    def haption_pose_cb(self, msg):
        """Updates the real Cartesian orientation of the Virtuose handle."""
        # Server publishes geometry_msgs/Pose (NOT PoseStamped) -> read .orientation
        # directly. The former PoseStamped subscription silently never matched the
        # publisher, so rot_haption stayed None and the clutch alignment was dead.
        q = msg.orientation
        self.rot_haption = R.from_quat([q.x, q.y, q.z, q.w])
        
    def button_cb(self, msg):
        """Updates the clutching state from the Virtuose button."""
        self.is_clutching = msg.data

    def goal_names_cb(self, msg):
        """Updates the list of active goal names from the shared autonomy inference engine."""
        self.goal_names = msg.data.split(',')

    def goal_probs_cb(self, msg):
        """Updates the array of goal probabilities."""
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

    def grasp_active_cb(self, msg):
        """Tracks whether the shared-autonomy node is autonomously driving a grasp."""
        self.grasp_active = bool(msg.data)

    def active_arm_cb(self, msg):
        """Switches which arm's EE data is used for force computation."""
        if msg.data in ('right', 'left') and msg.data != self.active_arm:
            self.active_arm = msg.data
            self.get_logger().info(f"[FORCE MGR] Active arm switched to {msg.data.upper()}")

    def target_cb(self, msg):
        """Updates the target Cartesian position and orientation (right arm reference)."""
        if self.active_arm != 'right':
            return
        if len(msg.data) >= 6:
            self.pos_target = np.array(msg.data[0:3])
            rpy = np.array(msg.data[3:6])
            self.rot_target = R.from_euler('xyz', rpy, degrees=False)

    def target_cb_left(self, msg):
        """Updates the target Cartesian position and orientation (left arm reference)."""
        if self.active_arm != 'left':
            return
        if len(msg.data) >= 6:
            self.pos_target = np.array(msg.data[0:3])
            rpy = np.array(msg.data[3:6])
            self.rot_target = R.from_euler('xyz', rpy, degrees=False)

    def real_cb(self, msg):
        """Updates the real Cartesian position and orientation of the active arm."""
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
        """Updates the control barrier function gradients mapped from the QP controller.

        Layout (14 floats, per-arm CBF split): [b_col_r, b_col_l, J_c_cart_R(6),
        J_c_cart_L(6)] -- was 13 floats [b_col, J_c_cart_R(6), J_c_cart_L(6)]
        before the per-arm SoftMin split. This device always drives the RIGHT
        gripper's cartesian gradient (indices 2:8, was 1:7).
        """
        if len(msg.data) >= 14:
            self.grad_cbf_right = np.array(msg.data[2:8])

    def lambda_cb(self, msg):
        """Updates the active CBF shadow price representing obstacle proximity.

        msg.data = [lambda_cbf_R, lambda_cbf_L] (two independent per-arm shadow
        prices, replacing the old single combined scalar). This device always
        drives the right gripper, so we take lambda_cbf_R.
        """
        if len(msg.data) < 1:
            return
        lambda_r = float(msg.data[0])
        self.lambda_cbf = lambda_r
        self.lambda_cbf_f = ((1.0 - self.CBF_LAMBDA_ALPHA) * self.lambda_cbf_f
                             + self.CBF_LAMBDA_ALPHA * max(0.0, lambda_r))

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

        # Orientation sync spring: pull the handle orientation toward the REAL EE
        # orientation (not the reference). Critical for the anti-runaway balance —
        # when the reference orientation flies away but the robot can't follow,
        # this torque reins the handle back. R_err = R_real * R_target^T.
        if self.rot_real is not None and self.rot_target is not None:
            err_rot = R.from_matrix(
                self.rot_real.as_matrix() @ self.rot_target.as_matrix().T).as_rotvec()
            Tau_tiago = self.Kp_sync_ang * err_rot
            F_sync[3] = -Tau_tiago[0]
            F_sync[4] = -Tau_tiago[1]
            F_sync[5] =  Tau_tiago[2]

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

    @staticmethod
    def _sync_share_factor(sync_mag, push_mag, share):
        """Attenuation factor (<=1) for the pushing forces so the sync spring
        reaches at least `share` of the (sync + push) magnitude.

        Solve  sync / (sync + k*push) = share   ->   k = sync*(1-share)/(share*push).
        Returns 1.0 (no attenuation) when share is negligible or magnitudes are
        tiny; clamps to [0, 1] so it only ever weakens the pushers, never boosts.
        """
        if share <= 1e-3 or push_mag < 1e-6 or sync_mag < 1e-6:
            return 1.0
        max_push = sync_mag * (1.0 - share) / share
        if push_mag > max_push:
            return float(np.clip(max_push / push_mag, 0.0, 1.0))
        return 1.0
    
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
        Velocity-field guidance: render the belief-weighted policy twist as a
        velocity FIELD the handle should follow, and apply a damping-style force
        that drives the handle velocity toward that field.

        Architecture:
          1. pi_blend = Σ P(k) · pi_k          (belief-weighted policy twist, robot frame)
          2. v_field_haption = map(pi_blend)   (180° Z-flip into the Haption frame)
          3. F = D_guide · (v_field_haption − v_handle) · confidence   (saturated)

        Why this and not a position/offset spring (see __init__ for the full
        rationale): the policy velocity is tanh-saturated to a near-constant
        magnitude far from the goal, so an offset-spring becomes a constant push
        that — with no damping in DEBUG mode — just accelerates the handle and
        never settles. The velocity field instead:
          * pushes when the handle is still (D · v_field) → starts the motion,
          * fades to zero as the handle reaches v_field → no runaway / fling,
          * lets a passive hand cruise at exactly pi_blend, so the teleop
            reference traces the SAME path the POLICY_BELIEF_TEST=True mode
            commands directly (the gripper is driven to the goal through the
            user's hand),
          * vanishes at the goal (pi_blend → 0) → settles cleanly.
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

        # Confidence gate: the ACTIVE-goal belief b_max (max posterior), read from
        # active_goal_pose[6] into self.fix_confidence -- the SAME signal and
        # convention JF/JFB/JB use. Replaces the former 1-normalised-entropy metric
        # so ALL guidance cells gate on one belief function. b_max is forced to 0
        # during autonomous grasp execution, so the guidance releases cleanly while
        # the node drives the arm.
        alpha = self._smoothstep(self.fix_confidence, lo=self.GUIDE_CONF_LO, hi=self.GUIDE_CONF_HI)

        # Proximity gate: distance from the REFERENCE (pos_target) to the active
        # goal (fix_goal_pos). Fades guidance to zero far away (where the policy
        # twist is large but the goal pose is still swinging) and ramps to full
        # near the goal. See GUIDE_PROX_* in __init__ for the rationale.
        if self.fix_goal_pos is not None and self.pos_target is not None:
            d_goal = float(np.linalg.norm(self.fix_goal_pos - self.pos_target))
            prox = np.clip(
                (self.GUIDE_PROX_FAR - d_goal)
                / max(self.GUIDE_PROX_FAR - self.GUIDE_PROX_NEAR, 1e-6), 0.0, 1.0)
            prox_gate = 3.0 * prox ** 2 - 2.0 * prox ** 3   # smoothstep
        else:
            prox_gate = 0.0   # no goal/reference info → no guidance (safe)

        gain = alpha * prox_gate   # belief-confidence × proximity

        # Map the policy twist (robot frame) into the Haption frame: this is the
        # device velocity the handle must have for the teleop integrator to
        # reproduce pi_blend on the robot (180° Z-flip, matching teleop_triago_clutch).
        v_field = np.array([
            -pi_blend[0], -pi_blend[1],  pi_blend[2],
            -pi_blend[3], -pi_blend[4],  pi_blend[5],
        ])

        # Velocity-field tracking force: drive the handle velocity toward v_field.
        # Intrinsically damped via the −v_handle term, so it cannot run away.
        dv = v_field - self.vel_haption
        F_guide_raw = np.zeros(6)
        F_guide_raw[0:3] = self.D_guide_lin * dv[0:3] * gain
        F_guide_raw[3:6] = self.D_guide_ang * dv[3:6] * gain

        # Saturate (tanh soft-clip) for operator comfort and hard bounding.
        F_guide_raw[0:3] = self.MAX_GUIDE_FORCE * np.tanh(F_guide_raw[0:3] / self.MAX_GUIDE_FORCE)
        F_guide_raw[3:6] = self.MAX_GUIDE_TORQUE * np.tanh(F_guide_raw[3:6] / self.MAX_GUIDE_TORQUE)

        # Temporal smoothing (LPF)
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

        # Near-goal orientation ASSIST: strengthen the torque (up to +20%) as the
        # EE approaches the goal, so small residual rotation errors get actively
        # driven to alignment (the operator struggles to do this by hand). Ramped
        # by the EE→goal distance (full within FIX_TORQUE_NEAR, off past *_FAR).
        d_goal = float(np.linalg.norm(self.fix_goal_pos - self.pos_real))
        prox = np.clip(
            (self.FIX_TORQUE_FAR - d_goal)
            / max(self.FIX_TORQUE_FAR - self.FIX_TORQUE_NEAR, 1e-6), 0.0, 1.0)
        prox = 3.0 * prox ** 2 - 2.0 * prox ** 3     # smoothstep
        tau_scale = 1.0 + self.FIX_TORQUE_NEAR_BOOST * prox
        K_tau = self.K_fix_torque * tau_scale
        max_tau = self.MAX_FIX_TORQUE * tau_scale
        Tau_fix_robot = max_tau * np.tanh(K_tau * err_rot_vec / max_tau)

        # Map robot -> Haption frame (180° Z-flip), scaled by the confidence gate.
        F_fix_raw = np.array([
            -F_fix_robot[0],   -F_fix_robot[1],    F_fix_robot[2],
            -Tau_fix_robot[0], -Tau_fix_robot[1],  Tau_fix_robot[2],
        ])
        # "Damped but strong": near the goal, oppose the handle's angular velocity
        # so the strengthened torque settles the orientation instead of ringing.
        F_fix_raw[3:6] -= self.K_FIX_TORQUE_DAMP * prox * self.vel_haption[3:6]
        F_fix_raw *= gate

        self.f_fix_filtered = (self.alpha_fix * F_fix_raw
                               + (1.0 - self.alpha_fix) * self.f_fix_filtered)
        return self.f_fix_filtered.copy()

    def compute_F_limit_warning(self):
        """Joint-limit "clutch advice" vibration: a ONE-SHOT 1 s torque burst.

        Behaviour (replaces the old continuous ramp):
          * Trigger — the moment any Haption joint enters LIMIT_OUTER of a limit,
            AND the burst is currently armed, a single fixed-amplitude
            (LIMIT_VIB_AMP) 75 Hz square-wave buzz is started on the three torque
            axes.
          * Duration — the buzz always plays for its full LIMIT_VIB_DURATION
            (1 s), even if the operator starts clutching partway through.
          * Latch — once fired the burst is DISARMED and cannot fire again until
            the operator completes a full clutch cycle (button press -> release).
            The operator is meant to read the buzz as "clutch now to re-center".
        """
        F_vib = np.zeros(6)
        now = time.time()

        # Re-arm on a COMPLETED clutch cycle (press -> release = falling edge).
        if self._vib_clutch_prev and not self.is_clutching:
            self.limit_vib_armed = True
        self._vib_clutch_prev = self.is_clutching

        # Closest distance to any of the 12 joint bounds.
        dist_to_min = self.joint_pos - self.joint_min
        dist_to_max = self.joint_max - self.joint_pos
        min_margin = float(np.min(np.concatenate([dist_to_min, dist_to_max])))

        # Trigger a fresh one-shot burst (only if armed and not already playing).
        if (min_margin <= self.LIMIT_OUTER
                and self.limit_vib_armed
                and not self.limit_vib_active):
            self.limit_vib_active = True
            self.limit_vib_start_time = now
            self.limit_vib_armed = False   # stay silent until the next clutch cycle

        # Emit the burst for its fixed duration, then stop (let it finish even if
        # the user is already clutching).
        if self.limit_vib_active:
            if (now - self.limit_vib_start_time) <= self.LIMIT_VIB_DURATION:
                self.vib_toggle *= -1.0
                amp = self.LIMIT_VIB_AMP
                F_vib[3] = amp * self.vib_toggle
                F_vib[4] = amp * self.vib_toggle
                F_vib[5] = amp * self.vib_toggle
            else:
                self.limit_vib_active = False

        return F_vib

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

        # Grasp execution takes PRECEDENCE over every other mode (including DEBUG):
        # while the autonomy drives the arm the user input is ignored, so we drag
        # the handle to FOLLOW the EE motion — the operator physically feels what
        # the autonomous grasp / lift / abort-retreat is doing. (Previously this
        # was an `elif` after DEBUG_ONLY_GUIDE, so in the active DEBUG mode the
        # handle felt NOTHING during a grasp; now it always follows.)
        if self.grasp_active:
            if self.pos_real is not None:
                if self._grasp_start_pos is None:
                    self._grasp_start_pos = self.pos_real.copy()
                # Gentle position tether toward where the arm is now, PLUS a
                # velocity-following term so the operator feels the instantaneous
                # direction & speed of the autonomous motion. The velocity term is
                # what makes the LIFT clearly felt (the +Z EE velocity pulls the
                # handle up) even when the cumulative Z displacement is small.
                err = self.pos_real - self._grasp_start_pos
                F_follow = self.GRASP_FOLLOW_KP * err + self.GRASP_FOLLOW_KD * self.vel_real
                F_haption = np.zeros(6)
                F_haption[0] = -F_follow[0]
                F_haption[1] = -F_follow[1]
                F_haption[2] =  F_follow[2]
                f_total_normal = F_haption
            else:
                f_total_normal = np.zeros(6)
            f_cbf_s = np.zeros(6)
            f_guide_s = np.zeros(6)
            f_fix_s = np.zeros(6)
        elif self.DEBUG_ONLY_GUIDE:
            # Guidance-only mode: F_guide (velocity field, drives from far) +
            # F_fixture (position spring, holds precisely at the goal).
            self._grasp_start_pos = None  # reset so the next grasp re-anchors
            f_total_normal = f_guide + f_fix
            f_cbf_s = np.zeros(6)
            f_guide_s = f_guide.copy()
            f_fix_s = f_fix.copy()
        else:
            self._grasp_start_pos = None  # reset for next grasp
            pos_err = (np.linalg.norm(self.pos_real - self.pos_target)
                       if (self.pos_real is not None and self.pos_target is not None) else 0.0)
            ang_err = 0.0
            if self.rot_real is not None and self.rot_target is not None:
                ang_err = float(np.linalg.norm(R.from_matrix(
                    self.rot_real.as_matrix() @ self.rot_target.as_matrix().T).as_rotvec()))
            divergence = max(pos_err / self.SYNC_FULL_POS_ERR, ang_err / self.SYNC_FULL_ANG_ERR)
            sync_share = min(self.SYNC_SHARE_AT_FULL * divergence, self.SYNC_SHARE_CAP)

            # Guidance (guide + fix) yields as the sync demand grows (reference
            # drifting far).
            guide_gain = float(np.clip(1.0 - sync_share, 0.0, 1.0))
            # F_CBF DISABLED (per operator request): compute_F_cbf() above is still
            # called every tick — f_cbf keeps updating (LPF state, telemetry, plot
            # trace) — but its contribution to the total wrench is zeroed out here.
            # f_cbf_s = self.CBF_GAIN_BOOST * f_cbf
            f_cbf_s = np.zeros(6)
            f_guide_s = guide_gain * f_guide
            f_fix_s = guide_gain * f_fix

            # Calculate the normal running force (F_sync + guidance; F_cbf excluded).
            # NOTE: f_vib is intentionally NOT summed here — the one-shot limit
            # burst is injected AFTER the clutch/grasp branching (see below) so it
            # is felt live and never captured/frozen by the clutch snapshot.
            f_total_normal = f_sync + f_cbf_s + f_guide_s + f_fix_s

            # --- AUTHORITY CAP --------------------------------------------- #
            # Bound the MAGNITUDE only when it is exceeded (does NOT force a fixed
            # total): the autonomy can never overpower the operator.
            f_norm = np.linalg.norm(f_total_normal[0:3])
            if f_norm > self.MAX_TOTAL_FORCE:
                f_total_normal[0:3] *= self.MAX_TOTAL_FORCE / f_norm
            t_norm = np.linalg.norm(f_total_normal[3:6])
            if t_norm > self.MAX_TOTAL_TORQUE:
                f_total_normal[3:6] *= self.MAX_TOTAL_TORQUE / t_norm

        # ========================================================
        # CLUTCHING ARCHITECTURE & ALIGNMENT GUIDANCE
        # ========================================================
        if self.DEBUG_ONLY_GUIDE:
            # In debug mode: skip clutching, skip global damping — raw F_guide only.
            f_total = f_total_normal.copy()
        elif self.is_clutching:
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

        # GLOBAL VISCOUS DAMPING (impedance-device stability).
        # UNIFIED: CONSTANT 0.7 / 0.1 in all clutch cells (same as C). The former
        # CBF-aware damp_scale (1.0 -> 2.0 with lambda_cbf_f) was removed so the
        # damping felt by the operator is identical across every clutch condition.
        if not self.DEBUG_ONLY_GUIDE:
            Kd_global_lin = 0.7
            Kd_global_ang = 0.1
            f_total[0:3] -= Kd_global_lin * self.vel_haption[0:3]
            f_total[3:6] -= Kd_global_ang * self.vel_haption[3:6]

        # ========================================================
        # JOINT-LIMIT "CLUTCH ADVICE" VIBRATION (injected last, on top of
        # whatever wrench the branch produced — normal, clutch-frozen or grasp —
        # so the one-shot burst is always felt live and toggles every frame).
        # ========================================================
        f_total[3:6] += f_vib[3:6]

        # ========================================================
        # CLIPPING & PUBLISHING
        # ========================================================
        f_total[0:3] = np.clip(f_total[0:3], -self.MAX_FORCE, self.MAX_FORCE)
        f_total[3:6] = np.clip(f_total[3:6], -self.MAX_TORQUE, self.MAX_TORQUE)
        msg = Wrench()
        msg.force.x, msg.force.y, msg.force.z = float(f_total[0]), float(f_total[1]), float(f_total[2])
        msg.torque.x, msg.torque.y, msg.torque.z = float(f_total[3]), float(f_total[4]), float(f_total[5])
        self.force_pub.publish(msg)

        # Buffer Data for Plotting (use the SCALED pushers so the % shares reflect
        # what was actually sent after the adaptive-sync attenuation).
        t = time.time() - self.start_time
        guide_comb = f_guide_s + f_fix_s   # guidance share = viscous guide + position fixture
        components = {'Sync': f_sync, 'CBF': f_cbf_s, 'Guide': guide_comb, 'Limit': f_vib}

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
            # SA inference frequency (unchanged — still tracked from goal_probs)
            self._sa_freq_data.append(self._sa_freq_lpf)
            # Own node frequency
            now = time.time()
            if self._own_last_time is not None:
                dt_own = now - self._own_last_time
                if dt_own > 1e-6:
                    self._own_freq_lpf = 0.9 * self._own_freq_lpf + 0.1 * (1.0 / dt_own)
            self._own_last_time = now
            self._own_freq_data.append(self._own_freq_lpf)

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