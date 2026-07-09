#!/usr/bin/env python3
"""haptic_force_manager_C -- "no-guidance" baseline force feedback.

THIRD feedback strategy (baseline / control condition for the user study). It runs
alongside the EXISTING clutch teleop node (teleop_triago_clutch.py); no predictive
assistance is provided and main_shared_autonomy's guidance is NOT used.

The operator teleoperates the TRIAGo hand entirely by hand. The ONLY assistive
feedback rendered is F_sync -- the spring-damper tether that keeps the Haption
handle synced with the real EE pose -- computed EXACTLY as in
haptic_force_manager_CF.py but 30% STRONGER (sync gains x1.3). There is no
F_guide, no F_fixture, no F_cbf, and no clutch alignment guidance.

To stay CONSISTENT with haptic_force_manager_CF (same rules/features), this
node keeps everything that is not guidance:
  * grasp_active EE-FOLLOWING: while the grasp state machine drives the arm
    autonomously (/shared_autonomy/grasp_active = True) the user input is ignored
    and the handle is dragged to follow the EE motion (position tether + velocity
    following) so the operator feels the autonomous grasp/lift/abort -- identical
    to the tutorial. (If the grasp machine is not running, this branch is simply
    never entered.)
  * clutch handling: on clutch press the wrench is frozen at 50% (cognitive
    grounding) -- but WITHOUT the tutorial's added rotational alignment guidance.
  * joint-limit "clutch advice" vibration: IDENTICAL to the tutorial (Mode A) --
    a one-shot 1 s / 0.07 Nm torque buzz fired when a Haption joint enters
    LIMIT_OUTER of a limit, disarmed after firing and re-armed only by a full
    clutch cycle. Both are clutch teleoperation methods, so the cue is shared
    verbatim. (Replaces the earlier CBF-proximity buzz.)
  * global viscous damping, arm switching, and the 180-deg-Z Haption<->TRIAGo
    frame map are unchanged.
  * final MAX_FORCE / MAX_TORQUE device safety clip is unchanged.

Removed vs the tutorial (this is a NO-GUIDANCE baseline):
  * F_guide (velocity-field guidance), F_fixture (position funnel), F_cbf
    (collision repulsion) and all their subscriptions/state.
  * the adaptive sync-share and the MAX_TOTAL assistive authority cap (they only
    shaped/bounded the guidance pushers, which no longer exist).
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench, Twist, Pose
from std_msgs.msg import Bool, Float64MultiArray, String
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from collections import deque
import matplotlib.pyplot as plt
import matplotlib

# Single source of truth for the experiment-condition selector (2x3 study).
import triago_control.qp_controller.config as cfg

# Set backend to avoid blocking the ROS spin loop
matplotlib.use('TkAgg')


class HapticForceManagerNoGuidance(Node):
    def __init__(self):
        """Initializes the no-guidance haptic force manager (F_sync tether only)."""
        super().__init__('haptic_force_manager_noguidance')

        # Fail loudly if launched under the wrong study condition. This is the
        # CLUTCH "Sync only" baseline: NO assistive feedback, NO blending
        # (F_sync tether is the only rendered force).
        cfg.validate_condition('haptic_force_manager_C',
                               control_mode=cfg.CLUTCH, feedback=False, blending=False)

        # --- State Variables ---
        self.pos_target = None
        self.rot_target = None
        self.pos_real = None
        self.rot_real = None
        self.vel_real = np.zeros(3)     # active-arm EE linear velocity (from ee_real)
        self.vel_haption = np.zeros(6)  # handle 6D spatial velocity (Haption frame)

        # --- F_sync gains (DOUBLED for this baseline) ---
        # Tutorial: Kp_sync = 10.0, Kp_sync_ang = 0.3, Kd_sync = 0.0.
        # This baseline renders ONLY F_sync, so the tether is made much stronger:
        # 2x the previous setting of this node (13.0 / 0.39) = 2.6x the tutorial.
        # Kd stays 0 -> the global damping below supplies the viscous term.
        self.Kp_sync = 30.0        # N/m     translation sync spring  [×3 tutorial: sync is the only force]
        self.Kd_sync = 0.0         # Ns/m    (0: global damping supplies the viscous term)
        self.Kp_sync_ang = 0.9     # Nm/rad  orientation sync spring  [×3 tutorial]

        # --- Grasp-execution coupling (identical to the tutorial) ---
        # While /shared_autonomy/grasp_active = True the SM drives the arm and the
        # user input is ignored; we render a strong follow wrench so the operator
        # FEELS the autonomous motion. Position tether toward where the arm is now,
        # plus a velocity-following term (makes the LIFT clearly felt).
        self.grasp_active = False
        self._grasp_start_pos = None
        self.GRASP_FOLLOW_KP = 30.0     # N/m   position tether toward the current EE
        self.GRASP_FOLLOW_KD = 160.0    # Ns/m  velocity-following (direction/speed of travel)
        # Grasp vibration cue (REPLACES the follow force): a constant 0.07 Nm
        # square-wave buzz on the torque axes rendered for the WHOLE autonomous
        # grasp. No force is impressed during the grasp — only this cue.
        self.GRASP_VIB_AMP = 0.07
        self.grasp_vib_toggle = 1.0

        # --- Clutching (freeze wrench at 50% on press + orientation alignment) ---
        self.is_clutching = False
        self.was_clutching_last_frame = False
        self.f_clutch_frozen = np.zeros(6)
        # Clutch orientation-alignment torque: UNIFIED across all clutch cells and
        # treated as a SYNC effect (help the operator re-align the handle to the EE
        # reference while clutching), so it is present in the baseline too.
        self.rot_haption = None
        self.K_align = 10.0  # Nm/rad — clutch orientation-alignment stiffness
        # DISABLED (bug fix): the alignment error mixes frames — rot_target is in
        # the robot base frame, rot_haption in the Haption device frame — so the
        # torque is NON-restorative and drives an unstable limit cycle that
        # saturates the torque clip ("explosion") on clutch press. It was previously
        # inert (the PoseStamped subscription never matched the Pose publisher).
        # Keep OFF until a frame-correct alignment is derived and bench-tested.
        self.ENABLE_CLUTCH_ALIGN = False

        # --- Global viscous damping (impedance-device stability, same as tutorial) ---
        self.Kd_global_lin = 0.7   # Ns/m
        self.Kd_global_ang = 0.1   # Nms/rad

        # --- Device safety clip (final bound published to the handle) ---
        self.MAX_FORCE = 10.0      # N
        self.MAX_TORQUE = 1.0      # Nm
        # Authority cap (UNIFIED across all 8 cells; currently == the device clip):
        # one knob for the max assistance magnitude, applied proportionally.
        self.MAX_TOTAL_FORCE = 10.0
        self.MAX_TOTAL_TORQUE = 1.0

        # --- Arm the force is computed for (follows /shared_autonomy/active_arm) ---
        self.active_arm = 'right'

        # --- Articular Limit Variables (Haption joint positions + bounds) ---
        self.joint_pos = np.zeros(6)
        self.joint_min = np.array([-0.804283, -1.65038, 0.728283, -3.02431, -1.28196, -2.05398])
        self.joint_max = np.array([0.781944, -0.0654231, 2.49752, 2.82038, 1.04722, 2.09453])

        #  Articular Limit Vibration Tuning Parameters
        self.LIMIT_OUTER = 0.25       # Radians where vibration starts
        self.LIMIT_INNER = 0.15       # Radians where vibration hits maximum
        self.AMP_MIN = 0.05           # Nm torque at the outer boundary
        self.AMP_MAX = 0.07           # Nm torque at the inner boundary
        self.vib_toggle = 1.0         # Toggles between 1 and -1 every frame for 75Hz square wave

        # --- Joint-limit "clutch advice" one-shot burst (IDENTICAL to Mode A) ---
        # When a joint enters LIMIT_OUTER, fire a SINGLE fixed-amplitude buzz that
        # lasts LIMIT_VIB_DURATION and then goes silent. It will NOT fire again
        # until the operator completes a full clutch cycle (press -> release), so
        # the burst is unambiguously interpreted as "you should clutch now". Both
        # this baseline and Mode A are clutch teleoperation methods, so the cue is
        # shared verbatim.
        self.LIMIT_VIB_DURATION = 1.0   # s   burst length
        self.LIMIT_VIB_AMP = 0.07       # Nm  fixed torque amplitude of the burst
        self.limit_vib_armed = True     # ready to fire (re-armed by a clutch cycle)
        self.limit_vib_active = False   # a burst is currently playing
        self.limit_vib_start_time = 0.0 # wall-clock start of the active burst
        self._vib_clutch_prev = False   # previous clutch state (for cycle edge detect)

        # --- Data Buffers & Synchronization (live plot) ---
        self.plot_lock = threading.Lock()
        self.plot_window_sec = 10.0
        self.buffer_size = int(150 * self.plot_window_sec)
        self.t_data = deque(maxlen=self.buffer_size)
        self.start_time = time.time()
        self.sync_F = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.sync_T = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.tot_F = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.tot_T = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.freq_data = deque(maxlen=self.buffer_size)
        self._own_freq_lpf = 0.0
        self._own_last_time = None

        # --- Subscriptions (reference + real EE + handle velocity + clutch/arm/grasp) ---
        # NOTE: NO guidance topics (goal_names/probabilities/user_policy/
        # active_goal_pose) and NO CBF topics (collision_constraints/lambda_cbf):
        # this baseline renders only F_sync.
        self.create_subscription(Float64MultiArray, '/arm_right/cartesian_reference', self.target_cb, 10)
        self.create_subscription(Float64MultiArray, '/arm_left/cartesian_reference', self.target_cb_left, 10)
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self.real_cb, 10)
        self.create_subscription(Twist, 'virtuose/velocity', self.vel_cb, 10)
        self.create_subscription(Bool, 'virtuose/button_right', self.button_cb, 10)
        # Grasp-execution flag: follow the EE (same as the tutorial) while True.
        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_cb, 10)
        # Arm switch: follow the active arm for EE state slicing and reference topic.
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)
        # Haption joint encoders for the joint-limit clutch-advice vibration.
        self.create_subscription(Float64MultiArray, 'virtuose/articular_position', self.joint_cb, 10)
        # Handle orientation for the clutch alignment torque (same as CF/CB/CFB).
        self.create_subscription(Pose, 'virtuose/pose', self.haption_pose_cb, 10)

        self.force_pub = self.create_publisher(Wrench, 'virtuose/force_cmd', 10)

        # --- Timer (150 Hz control loop) ---
        self.dt = 1.0 / 150.0
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.setup_plot()
        self.get_logger().info(
            "Haptic Force Manager (NO-GUIDANCE baseline) started: F_sync only, "
            f"doubled tether (Kp_sync={self.Kp_sync}, Kp_sync_ang={self.Kp_sync_ang}).")

    # =========================
    # PLOT SETUP & UPDATE
    # =========================
    def setup_plot(self):
        """Initializes a compact live plot: F_sync, published total, and loop rate."""
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 2, figsize=(11, 8))
        self.fig.canvas.manager.set_window_title('Haptic Force (NO-GUIDANCE: F_sync only)')
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']

        # [0,0] Sync force, [0,1] Sync torque
        self.lines_sync_F = []
        ax = self.axs[0, 0]
        ax.set_title("F_sync - FORCE (N)", fontsize=10)
        ax.set_ylabel("Force (N)"); ax.grid(True, linestyle='--', alpha=0.6)
        for i in range(3):
            self.lines_sync_F.append(ax.plot([], [], color=colors[i], label=f"F{labels[i]}")[0])
        ax.legend(loc='upper left', fontsize=8, ncol=3)

        self.lines_sync_T = []
        ax = self.axs[0, 1]
        ax.set_title("F_sync - TORQUE (Nm)", fontsize=10)
        ax.set_ylabel("Torque (Nm)"); ax.grid(True, linestyle='--', alpha=0.6)
        for i in range(3):
            self.lines_sync_T.append(ax.plot([], [], color=colors[i], label=f"T{labels[i]}")[0])
        ax.legend(loc='upper left', fontsize=8, ncol=3)

        # [1,0] Total force, [1,1] Total torque
        self.lines_tot_F = []
        ax = self.axs[1, 0]
        ax.set_title("Published TOTAL - FORCE (N)", fontsize=10)
        ax.set_ylabel("Force (N)"); ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(self.MAX_FORCE, color='k', linestyle='--', linewidth=1.0, alpha=0.6, label='±max')
        ax.axhline(-self.MAX_FORCE, color='k', linestyle='--', linewidth=1.0, alpha=0.6)
        for i in range(3):
            self.lines_tot_F.append(ax.plot([], [], color=colors[i], label=f"F{labels[i]}")[0])
        ax.legend(loc='upper left', fontsize=8, ncol=4)

        self.lines_tot_T = []
        ax = self.axs[1, 1]
        ax.set_title("Published TOTAL - TORQUE (Nm)", fontsize=10)
        ax.set_ylabel("Torque (Nm)"); ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(self.MAX_TORQUE, color='k', linestyle='--', linewidth=1.0, alpha=0.6, label='±max')
        ax.axhline(-self.MAX_TORQUE, color='k', linestyle='--', linewidth=1.0, alpha=0.6)
        for i in range(3):
            self.lines_tot_T.append(ax.plot([], [], color=colors[i], label=f"T{labels[i]}")[0])
        ax.legend(loc='upper left', fontsize=8, ncol=4)

        # [2,0] loop frequency, [2,1] hidden
        ax = self.axs[2, 0]
        ax.set_title("Force Manager Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Hz"); ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 180); ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(150, color='g', linestyle='--', linewidth=1.0, alpha=0.7, label='target 150Hz')
        self.line_freq, = ax.plot([], [], color='#9467bd', linewidth=1.5, label='HFM freq')
        ax.legend(loc='upper left', fontsize=8)
        self.axs[2, 1].axis('off')

        self.fig.tight_layout()
        plt.show(block=False)

    def update_plot(self):
        """Snapshots buffers under the lock and refreshes the Matplotlib UI."""
        with self.plot_lock:
            if len(self.t_data) == 0:
                return
            t_list = list(self.t_data)
            sF = [list(self.sync_F[i]) for i in range(3)]
            sT = [list(self.sync_T[i]) for i in range(3)]
            tF = [list(self.tot_F[i]) for i in range(3)]
            tT = [list(self.tot_T[i]) for i in range(3)]
            freq_list = list(self.freq_data)

        current_t = t_list[-1]
        win = (current_t - self.plot_window_sec, current_t)

        for i in range(3):
            self.lines_sync_F[i].set_data(t_list, sF[i])
            self.lines_sync_T[i].set_data(t_list, sT[i])
            self.lines_tot_F[i].set_data(t_list, tF[i])
            self.lines_tot_T[i].set_data(t_list, tT[i])
        for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            self.axs[r, c].set_xlim(*win)
            self.axs[r, c].relim()
            self.axs[r, c].autoscale_view(scalex=False, scaley=True)

        n = min(len(t_list), len(freq_list))
        self.line_freq.set_data(t_list[:n], freq_list[:n])
        self.axs[2, 0].set_xlim(*win)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # =========================
    # CALLBACKS
    # =========================
    def haption_pose_cb(self, msg):
        """Handle orientation (geometry_msgs/Pose) for the clutch alignment torque."""
        q = msg.orientation
        self.rot_haption = R.from_quat([q.x, q.y, q.z, q.w])

    def button_cb(self, msg):
        """Updates the clutching state from the Virtuose button."""
        self.is_clutching = msg.data

    def grasp_active_cb(self, msg):
        """Tracks whether the shared-autonomy node is autonomously driving a grasp."""
        self.grasp_active = bool(msg.data)

    def active_arm_cb(self, msg):
        """Switches which arm's EE data is used for force computation."""
        if msg.data in ('right', 'left') and msg.data != self.active_arm:
            self.active_arm = msg.data
            self.get_logger().info(f"[FORCE MGR] Active arm switched to {msg.data.upper()}")

    def joint_cb(self, msg):
        """Updates the current 6-DoF joint positions directly from the Haption encoders."""
        if len(msg.data) >= 6:
            self.joint_pos = np.array(msg.data[0:6])

    def target_cb(self, msg):
        """Updates the target Cartesian pose (right arm reference)."""
        if self.active_arm != 'right':
            return
        if len(msg.data) >= 6:
            self.pos_target = np.array(msg.data[0:3])
            self.rot_target = R.from_euler('xyz', np.array(msg.data[3:6]), degrees=False)

    def target_cb_left(self, msg):
        """Updates the target Cartesian pose (left arm reference)."""
        if self.active_arm != 'left':
            return
        if len(msg.data) >= 6:
            self.pos_target = np.array(msg.data[0:3])
            self.rot_target = R.from_euler('xyz', np.array(msg.data[3:6]), degrees=False)

    def real_cb(self, msg):
        """Updates the real Cartesian pose + linear velocity of the active arm."""
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
        """Updates the current 6D spatial velocity of the handle (raw, unfiltered)."""
        self.vel_haption = np.array([
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z
        ])

    # =========================
    # FORCE COMPONENT (F_sync only)
    # =========================
    def compute_F_sync(self):
        """3D+3D spring tether keeping the handle synced with the real EE pose.

        Identical formulation to haptic_force_manager_CF.compute_F_sync (the
        180-deg-Z Haption<->TRIAGo frame map on both the position spring and the
        orientation spring), only with the 30%-boosted gains set in __init__.
        """
        F_sync = np.zeros(6)
        if self.pos_target is None or self.pos_real is None:
            return F_sync

        # Position spring (TRIAGo frame) -> Haption frame (negate X, negate Y, keep Z).
        error_pos_tiago = self.pos_real - self.pos_target
        F_spring_tiago = self.Kp_sync * error_pos_tiago
        F_spring_haption = np.array([-F_spring_tiago[0], -F_spring_tiago[1], F_spring_tiago[2]])
        F_sync[0:3] = F_spring_haption - (self.Kd_sync * self.vel_haption[0:3])

        # Orientation spring: R_err = R_real * R_target^T, mapped to the Haption frame.
        if self.rot_real is not None and self.rot_target is not None:
            err_rot = R.from_matrix(
                self.rot_real.as_matrix() @ self.rot_target.as_matrix().T).as_rotvec()
            Tau_tiago = self.Kp_sync_ang * err_rot
            F_sync[3] = -Tau_tiago[0]
            F_sync[4] = -Tau_tiago[1]
            F_sync[5] = Tau_tiago[2]

        return F_sync

    def compute_F_limit_warning(self):
        """Joint-limit "clutch advice" vibration: a ONE-SHOT 1 s torque burst.

        IDENTICAL to haptic_force_manager_CF.compute_F_limit_warning (Mode A):
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
        """Renders F_sync only (or the grasp-follow wrench during autonomous grasp),
        applies clutch-freeze + global damping, clips, publishes, and buffers."""
        f_sync = self.compute_F_sync()

        # Grasp execution takes precedence (same rule as the tutorial): while the
        # autonomy drives the arm, drag the handle to FOLLOW the EE motion so the
        # operator feels the grasp/lift/abort. If the grasp machine is not running,
        # grasp_active stays False and this branch is never entered.
        if self.grasp_active:
            # Grasp in progress: impress NO force (the old EE-follow pull is
            # removed). A constant 0.07 Nm vibration cue is added below instead.
            self._grasp_start_pos = None
            f_total_normal = np.zeros(6)
        else:
            self._grasp_start_pos = None   # reset so the next grasp re-anchors
            f_total_normal = f_sync.copy()

        # Authority cap (UNIFIED across all cells): proportionally bound the
        # assistive wrench magnitude to MAX_TOTAL_* (currently == the device clip).
        fn = np.linalg.norm(f_total_normal[0:3])
        if fn > self.MAX_TOTAL_FORCE:
            f_total_normal[0:3] *= self.MAX_TOTAL_FORCE / fn
        tn = np.linalg.norm(f_total_normal[3:6])
        if tn > self.MAX_TOTAL_TORQUE:
            f_total_normal[3:6] *= self.MAX_TOTAL_TORQUE / tn

        # Clutch handling: freeze the wrench at 50% on press (cognitive grounding).
        # No alignment guidance is added (this is the no-guidance baseline).
        if self.is_clutching:
            if not self.was_clutching_last_frame:
                self.f_clutch_frozen = f_total_normal / 2.0
                self.was_clutching_last_frame = True
            f_total = self.f_clutch_frozen.copy()

            # Clutch orientation-alignment torque (UNIFIED with CF/CB/CFB, treated
            # as a sync effect): pull the HANDLE orientation toward the target
            # orientation, faded to zero near a Haption joint limit.
            if self.ENABLE_CLUTCH_ALIGN and self.rot_haption is not None and self.rot_target is not None:
                error_rot_matrix = self.rot_target.as_matrix() @ self.rot_haption.as_matrix().T
                error_rot_vec = R.from_matrix(error_rot_matrix).as_rotvec()
                tau_align_base = self.K_align * error_rot_vec
                tau_align_haption = np.array([
                    -tau_align_base[0], -tau_align_base[1], tau_align_base[2]])
                dist_to_min = self.joint_pos - self.joint_min
                dist_to_max = self.joint_max - self.joint_pos
                min_margin = np.min(np.concatenate([dist_to_min, dist_to_max]))
                fade_margin = 0.35
                if min_margin < fade_margin:
                    scale = max(0.0, min_margin / fade_margin)
                    tau_align_haption *= scale
                f_total[3:6] += tau_align_haption
        else:
            f_total = f_total_normal
            self.was_clutching_last_frame = False

        # Global viscous damping (impedance-device stability). Skipped during a
        # grasp so the handle is free apart from the vibration cue below.
        if not self.grasp_active:
            f_total[0:3] -= self.Kd_global_lin * self.vel_haption[0:3]
            f_total[3:6] -= self.Kd_global_ang * self.vel_haption[3:6]

        # Joint-limit "clutch advice" vibration (IDENTICAL to Mode A): a ONE-SHOT
        # 1 s burst fired when a Haption joint nears a limit, re-armed only by a
        # full clutch cycle. Injected last so it rides on top of the possibly
        # clutch-frozen wrench and toggles every frame.
        f_vib = self.compute_F_limit_warning()
        if self.grasp_active:
            # Grasp cue: constant 0.07 Nm square-wave buzz on the torque axes for
            # the whole grasp (replaces the removed EE-follow force).
            self.grasp_vib_toggle *= -1.0
            gb = self.GRASP_VIB_AMP * self.grasp_vib_toggle
            f_total[3] += gb
            f_total[4] += gb
            f_total[5] += gb
        else:
            f_total[3:6] += f_vib[3:6]

        # Device safety clip.
        f_total[0:3] = np.clip(f_total[0:3], -self.MAX_FORCE, self.MAX_FORCE)
        f_total[3:6] = np.clip(f_total[3:6], -self.MAX_TORQUE, self.MAX_TORQUE)

        msg = Wrench()
        msg.force.x, msg.force.y, msg.force.z = float(f_total[0]), float(f_total[1]), float(f_total[2])
        msg.torque.x, msg.torque.y, msg.torque.z = float(f_total[3]), float(f_total[4]), float(f_total[5])
        self.force_pub.publish(msg)

        # Buffer for plotting.
        t = time.time() - self.start_time
        with self.plot_lock:
            self.t_data.append(t)
            for i in range(3):
                self.sync_F[i].append(f_sync[i])
                self.sync_T[i].append(f_sync[i + 3])
                self.tot_F[i].append(f_total[i])
                self.tot_T[i].append(f_total[i + 3])
            now = time.time()
            if self._own_last_time is not None:
                dt_own = now - self._own_last_time
                if dt_own > 1e-6:
                    self._own_freq_lpf = 0.9 * self._own_freq_lpf + 0.1 * (1.0 / dt_own)
            self._own_last_time = now
            self.freq_data.append(self._own_freq_lpf)


def main(args=None):
    """Initializes ROS, spins on a daemon thread, drives Matplotlib on the main thread."""
    rclpy.init(args=args)
    node = HapticForceManagerNoGuidance()

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
