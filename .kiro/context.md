# AI Agent Context — haption_teleoperation

> **This file is maintained by the AI agent. Do not edit manually.**
> Last updated: 2026-07-03 (§4.3 NEW: shared-autonomy TWIST BLENDING mode.
> `teleop_triago_clutch.py` and the new `haptic_force_manager_blending_
> tutorial.py` now both `import triago_control.qp_controller.config as cfg`
> and read `cfg.BLENDING` as the SAME single source of truth used by
> `triago_control/scripts/qp_arm_teleop/main_shared_autonomy.py`. This is the
> first time this package imports from `triago_control` — `package.xml` gained
> `<depend>triago_control</depend>`. See §4.3 for the full design; see
> `triago_control`'s own context.md §11.5 for the companion (robot-side) half.)
> Earlier: 2026-06-29 (per-arm bimanual, cost-decoupling, dual FSM/belief, arm-switch, grasp-fail retreat, force tuning)

---

## 1. Project Identity

- **Package name**: `haption_teleoperation`
- **ROS 2 distribution**: Humble (Ubuntu 22.04)
- **Robot**: PAL Robotics TRIAGo++ (bimanual, mobile base) — teleoperated via Haption Virtuose 6D
- **Maintainer**: Roberto Rocco (roberto.rocco@irisa.fr)
- **Repository**: https://github.com/Robertorocco/haption_teleoperation
- **Build system**: `ament_cmake` (hybrid C++/Python: C++ nodes for hardware API, Python scripts for teleoperation logic)
- **Runtime environment**: Dockerized ROS 2 workspace, shared via `~/exchange/` with host

---

## 2. Workspace Layout

```
~/exchange/ros2-ws/
└── src/
    ├── haption_teleoperation/   ← THIS REPO
    ├── triago_control/          ← QP controller + shared autonomy (sibling package)
    ├── haption_interface/       ← hardware driver (not maintained by user)
    └── pal-packages/            ← PAL vendor packages
```

---

## 3. Package Structure

```
haption_teleoperation/
├── CMakeLists.txt               (ament_cmake, links VirtuoseAPI + libtirpc)
├── package.xml                  (depends: rclcpp, geometry_msgs, sensor_msgs, rclpy)
├── .kiro/
│   └── context.md               ← THIS FILE
├── include/
│   └── VirtuoseAPI.h            (proprietary C header, Haption S.A.)
├── lib/
│   └── libVirtuoseAPI.so        (proprietary shared library — device driver)
├── src/                         ← C++ NODES (hardware API layer)
│   ├── virtuose_server_node.cpp     ★ primary: 150Hz impedance-mode device server
│   └── calibration_main.cpp         utility: manual joint-limit discovery tool
└── scripts/                     ← PYTHON NODES (teleoperation + force feedback)
    ├── teleop_triago_clutch.py                 ★ active: clutch-indexing teleop (mouse-mode).
    │                                              Topic-routes via cfg.BLENDING (§4.3).
    ├── haptic_force_manager_tutorial.py        ★ active (cfg.BLENDING=False): Virtual Fixture mode
    ├── haptic_force_manager_blending_tutorial.py ★ active (cfg.BLENDING=True): TWIST BLENDING mode (§4.3)
    ├── haptic_force_manager_battery.py         alternate/experimental variant
    ├── teleop_triago.py             forward teleop (no clutch, continuous integration)
    ├── teleop_demo_integrator.py    RViz-only demo (no robot, visualizes in "map" frame)
    ├── haption_plotter.py           live matplotlib: pose/vel/force from virtuose topics
    └── workspace_debug_visualizer.py  6-window 3D workspace alignment debugger
```

---

## 4. Teleoperation Strategy: Pure Haptic Guidance (Virtual Fixtures)

This package implements the **human-side** of the shared-autonomy architecture:

1. **Motion path** (user drives the robot):
   ```
   Haption device → teleop_triago_clutch.py → /arm_right/cartesian_reference → main_qp_controller.py (triago_control)
   ```
   The QP CLF-CBF safety controller tracks the Haption-generated reference. **No motion-level blending** is applied.

2. **Force feedback path** (assistive guidance):
   ```
   triago_control/main_shared_autonomy.py → /shared_autonomy/{goal_names, goal_probabilities, ee_policy, user_policy}
                                                        ↓
   haptic_force_manager.py → F_guide (belief-weighted viscous nudge toward most-probable goal)
                                                        ↓
   virtuose/force_cmd → Haption device → user FEELS the corrective force
   ```

3. **Exception — grasp execution**: during GRASP_APPROACH/GRASP_CLOSE/LIFT, the `triago_control` grasp state machine takes over motion control directly (precision + synchronization that the user cannot provide).

### 4.1 Authority Handover (teleop ↔ autonomous grasp)

In teleop mode (`POLICY_BELIEF_TEST=False`), `main_shared_autonomy.py` publishes
`/shared_autonomy/grasp_active` (Bool):
- **True** during GRASP_APPROACH/GRASP_CLOSE/LIFT → the node publishes the grasp reference
  directly to `/arm_right/cartesian_reference`, and `teleop_triago_clutch.py` **freezes** (stops
  publishing).
- **Falling edge** (grasp done → HOLDING) → the clutch **re-anchors** at the actual post-grasp
  robot pose (resets `initialized=False`) so teleop resumes with no jump.

> **Handle drag during autonomous phases (2026-06-29):** while `grasp_active=True`
> (grasp/align/approach/close/lift/abort-lift) the force manager applies `F_follow` — a position
> tether pulling the handle to follow the EE's displacement since grasp start — so the operator
> physically feels the autonomous motion. This now takes **precedence over `DEBUG_ONLY_GUIDE`**
> (previously the DEBUG branch short-circuited it, so the handle felt nothing during a grasp).

### 4.2 Position Virtual Fixture (haptic_force_manager_tutorial.py)

`main_shared_autonomy.py` publishes `/shared_autonomy/active_goal_pose`
(`[x,y,z,roll,pitch,yaw,confidence]`). The tutorial force manager applies `F_fixture`: a
position+orientation spring pulling the handle toward the exact grasp pose, gated by confidence
(`FIX_CONF_LO=0.55`→`HI=0.85`). Unlike the viscous `F_guide` (which vanishes at the goal), this
does NOT weaken near the goal, so it lets the operator settle precisely at the grasp standoff
against the CBF push-off. Confidence is forced to 0 during grasp execution (fixture releases).

### 4.3 Alternate mode: shared-autonomy TWIST BLENDING (2026-07-03)

A SECOND teleoperation strategy, selected via `cfg.BLENDING` in
`triago_control/qp_controller/config.py` (the single source of truth, also
read directly by `main_shared_autonomy.py` — see that repo's context.md
§11.5 for the full companion design). Unlike Virtual Fixture mode (§4, §4.2),
where the user's raw reference reaches the QP unmodified and ALL assistance
is rendered as haptic force, this mode blends the ACTUAL Cartesian reference
sent to the QP:

```
v_blend = (1 - alpha) * v_user + alpha * pi_policy      (twist-level blend)
alpha   = ALPHA_MAX * x**ALPHA_GAMMA                     (x = normalised belief)
```
persistently integrated by `main_shared_autonomy.py` every tick — this
package's two scripts only carry the pure USER reference and render the
resulting force feedback:

```
Haption → teleop_triago_clutch.py → /arm_right/user_cartesian_reference   [cfg.BLENDING=True]
                                              |
                                              v
                          triago_control/main_shared_autonomy.py
                            (belief inference, alpha=compute_alpha(b_max),
                             v_blend integration, SOLE publisher of the
                             real /arm_right/cartesian_reference,
                             publishes /shared_autonomy/blend_debug)
                                              |
                                              v
                            /arm_right/cartesian_reference
                                              |
                                              v
                          triago_control/main_qp_controller.py (QP CLF-CBF)
```

**Both `teleop_triago_clutch.py` and `haptic_force_manager_blending_
tutorial.py` `import triago_control.qp_controller.config as cfg`** and check
`cfg.BLENDING` at their own startup (no live-toggle — restart both nodes
after changing the flag) to decide which topic carries the pure user pose:
`/arm_*/cartesian_reference` (BLENDING=False) or `/arm_*/user_cartesian_
reference` (BLENDING=True). This is what lets `main_shared_autonomy.py`
become the sole, always-on publisher of the real `/arm_*/cartesian_
reference` without ever racing `teleop_triago_clutch.py` for the same topic.

**`haptic_force_manager_blending_tutorial.py`** (this mode's active force
node) renders ONLY `F_sync` — no `F_guide`/`F_fixture`/`F_cbf` at the haptic
level, since all assistance now happens at the reference level instead. It
tethers the handle toward the REAL robot EE using the PURE USER pose (read
from `/arm_*/user_cartesian_reference`, never the blended one — reading the
blended pose would hide the exact divergence the operator wants to feel).
Its 3rd plot window, "Authority Share", reads `/shared_autonomy/blend_debug`
(19 floats: `[alpha, v_user(6), v_policy(6), v_blend(6)]`) VERBATIM — the
exact numbers `main_shared_autonomy.py` used to command the robot — rather
than recomputing the blend independently, so the plot can never drift from
what actually happened.

**Cross-package dependency**: `package.xml` gained `<depend>triago_
control</depend>` (this is the first cross-package Python import in this
repo).

---

## 5. C++ Node: virtuose_server_node

- **Frequency**: 150 Hz (microsecond-precise wall timer)
- **Command mode**: `COMMAND_TYPE_IMPEDANCE` (force in, position out)
- **Indexing**: `INDEXING_NONE` (button must be held for the device to track)
- **IP**: `127.0.0.1#53210` (communicates via `libtirpc` with device controller)
- **Startup sequence**: open → configure → power on → 3s relay wait → loop
- **Force subscribe pattern**: asynchronous `ForceCallback` writes to `current_force[6]`; the 150 Hz timer reads and applies it with `virtSetForce` every tick

### Published topics:
| Topic | Type | Content |
|-------|------|---------|
| `virtuose/pose` | `Pose` | Device handle position + quaternion [x,y,z,w] |
| `virtuose/velocity` | `Twist` | 6-DOF spatial velocity (raw, device frame) |
| `virtuose/button` | `Bool` | Right button state (clutch) |
| `virtuose/articular_position` | `Float64MultiArray` | 6 joint positions (rad) |

### Subscribed topics:
| Topic | Type | Content |
|-------|------|---------|
| `virtuose/force_cmd` | `Wrench` | 6-DOF wrench applied to the device handle |

---

## 6. Key Script: teleop_triago_clutch.py

Implements **clutch-indexing** (mouse-mode) teleoperation:

- **Initialization**: waits for `/qp_debug/ee_real` to anchor integration at current robot EE pose
- **Frame mapping**: Haption→TRIAGo = 180° rotation around Z (negate X, negate Y, keep Z)
- **Clutch logic**: when button pressed → pose frozen, zero velocity published; when released → integration resumes from frozen pose
- **Output protocol**: 13-element `Float64MultiArray` on `/arm_right/cartesian_reference`:
  ```
  [pos(3), rpy(3), vel_lin(3), vel_ang(3), task_dim(1)]
  ```
- **task_dim** flag: `6.0` = full 6D control, `5.0` = free rotation around approach axis
- **Frequency**: 150 Hz (matches device server)

---

## 7. Key Script: haptic_force_manager.py

Multi-layer force-feedback superposition node (150 Hz). Computes and sums:

| Layer | Symbol | Description |
|-------|--------|-------------|
| Sync | F_sync | Spring-damper (Kp=10, Kd=0) tethering user to robot tracking error |
| CBF | F_cbf | Repulsive force from collision barrier gradient × λ_cbf, tanh-saturated, LPF'd (α=0.15) |
| Guide | F_guide | Belief-weighted blend of all leaf policies (continuous, entropy-gated confidence, viscous B=90 N/(m/s)) |
| Limit | F_limit | 75 Hz square-wave vibration when Haption joints approach mechanical limits |
| Clutch align | — | Rotational spring (K=10 Nm/rad) pulling handle toward target orientation during clutch |
| Global damping | — | Viscous Kd_lin=0.7, Kd_ang=0.1 for stability |

### F_guide (Virtual Fixtures) — Continuous Policy-Merging

The guidance wrench uses **continuous belief-weighted policy blending** (NOT winner-take-all):

```
pi_blend = Σ_k  w(k) · pi_k       (convex combination, smooth in beliefs)
error_v  = pi_blend − v_user       (velocity error in robot frame)
F_guide  = B · error_v · alpha     (viscous, scaled by confidence gain)
```

- **Confidence gain**: `alpha = smoothstep(1 − H_norm)` where H_norm is the normalised entropy of the belief distribution. Transparent at uniform belief, full guidance when one goal dominates.
- **Temporal LPF**: `alpha_guide = 0.15` guarantees C0 continuity even on noisy probability samples.

### Passivity architecture:
- **Observer (PO)**: integrates power = −(wrench · twist) to track energy balance
- **Controller (PC)**: when energy < 0 (active), injects dissipative damping β·v, saturated at MAX_PC_FORCE=5N / MAX_PC_TORQUE=0.5Nm
- **PC enable toggle**: `ENABLE_PASSIVITY_CONTROL` flag (currently `False` for tuning)

### Safety clipping:
Global MAX_FORCE=10N, MAX_TORQUE=1Nm after all layers summed.

### Subscribed topics (inference state from triago_control):
| Topic | Type | Content |
|-------|------|---------|
| `/shared_autonomy/goal_names` | `String` | Comma-separated goal keys |
| `/shared_autonomy/goal_probabilities` | `Float64MultiArray` | Per-goal belief probabilities |
| `/shared_autonomy/user_policy` | `Float64MultiArray` | Flattened optimal twists (6 DOF × N goals) |

### Live plotting:
3 matplotlib windows (force superposition 5×2 grid, passivity observer, twist analyzer) running on main thread with ROS spinning on daemon thread.

---

## 8. Frame Convention (Haption ↔ TRIAGo Mapping)

The Haption device base frame has **X pointing toward the user** and **Y to the right** (operator's perspective). The TRIAGo `base_footprint` has X forward and Y left. The relationship is a **pure 180° rotation around Z**:

```
TRIAGo_vel.x = -Haption_vel.x
TRIAGo_vel.y = -Haption_vel.y
TRIAGo_vel.z = +Haption_vel.z
(same for angular velocities)
```

For force feedback (Haption←TRIAGo), the **same** negation applies (transpose of rotation = same rotation for 180°):
```
Haption_force.x = -TRIAGo_force.x
Haption_force.y = -TRIAGo_force.y
Haption_force.z = +TRIAGo_force.z
```

---

## 9. Haption Device Joint Limits

From encoder calibration (`calibration_main.cpp`):

| Joint | Min (rad) | Max (rad) |
|-------|-----------|-----------|
| J1 | -0.804 | +0.782 |
| J2 | -1.650 | -0.065 |
| J3 | +0.728 | +2.498 |
| J4 | -3.024 | +2.820 |
| J5 | -1.282 | +1.047 |
| J6 | -2.054 | +2.095 |

Vibration warning triggers at `LIMIT_OUTER = 0.25 rad` from any limit; maximum at `LIMIT_INNER = 0.15 rad`.

---

## 10. Inter-Package Topic Interface

**Virtual Fixture mode (`cfg.BLENDING=False`, default):**

| Direction | Topic | Publisher | Subscriber |
|-----------|-------|-----------|------------|
| Haption → Robot | `/arm_right/cartesian_reference` | `teleop_triago_clutch.py` | `main_qp_controller.py` (triago_control) |
| Robot → Haption | `virtuose/force_cmd` | `haptic_force_manager_tutorial.py` | `virtuose_server_node` |
| Inference → Force | `/shared_autonomy/goal_names` | `main_shared_autonomy.py` | `haptic_force_manager_tutorial.py` |
| Inference → Force | `/shared_autonomy/goal_probabilities` | `main_shared_autonomy.py` | `haptic_force_manager_tutorial.py` |
| Inference → Force | `/shared_autonomy/user_policy` | `main_shared_autonomy.py` | `haptic_force_manager_tutorial.py` |
| Robot state | `/qp_debug/ee_real` | `main_qp_controller.py` | both teleop scripts |
| CBF gradient | `/collision_constraints` | `main_qp_controller.py` | `haptic_force_manager_tutorial.py` |
| CBF slack | `/qp_debug/lambda_cbf` | `main_qp_controller.py` | `haptic_force_manager_tutorial.py` |
| Clutch | `virtuose/button_right` | `virtuose_server_node` | teleop + force manager |
| Grasp trigger | `virtuose/button_left` | `virtuose_server_node` | `main_shared_autonomy.py` |
| Authority handover | `/shared_autonomy/grasp_active` | `main_shared_autonomy.py` | `teleop_triago_clutch.py` |
| Virtual fixture | `/shared_autonomy/active_goal_pose` | `main_shared_autonomy.py` | `haptic_force_manager_tutorial.py` |
| Device vel | `virtuose/velocity` | `virtuose_server_node` | teleop + force manager |

**TWIST BLENDING mode (`cfg.BLENDING=True`, §4.3) — topics that DIFFER:**

| Direction | Topic | Publisher | Subscriber |
|-----------|-------|-----------|------------|
| Haption → Robot (pure user intent) | `/arm_right/user_cartesian_reference` | `teleop_triago_clutch.py` | `main_shared_autonomy.py` |
| Robot (blended) → QP | `/arm_right/cartesian_reference` | `main_shared_autonomy.py` (SOLE publisher now) | `main_qp_controller.py` |
| Robot → Haption | `virtuose/force_cmd` | `haptic_force_manager_blending_tutorial.py` | `virtuose_server_node` |
| Authority-share telemetry | `/shared_autonomy/blend_debug` | `main_shared_autonomy.py` | `haptic_force_manager_blending_tutorial.py` |

All other topics (grasp trigger, clutch, device vel, `/qp_debug/ee_real`, etc.) are
unchanged between the two modes.

---

## 11. Build & Run

```bash
# Build
cd ~/exchange/ros2-ws
colcon build --packages-select haption_teleoperation
source install/setup.bash

# Run device server (requires hardware or simulator on 127.0.0.1#53210)
ros2 run haption_teleoperation virtuose_server_node

# Run clutch teleop (topic routing depends on cfg.BLENDING in triago_control's config.py)
ros2 run haption_teleoperation teleop_triago_clutch.py

# Run force feedback -- pick the script matching cfg.BLENDING:
ros2 run haption_teleoperation haptic_force_manager_tutorial.py            # cfg.BLENDING=False (Virtual Fixture)
ros2 run haption_teleoperation haptic_force_manager_blending_tutorial.py   # cfg.BLENDING=True  (Twist Blending, §4.3)

# Calibration utility (discover joint limits by manually moving device)
ros2 run haption_teleoperation virtuose_calibration

# Debug/visualization
ros2 run haption_teleoperation haption_plotter.py
ros2 run haption_teleoperation workspace_debug_visualizer.py
```

---

## 12. Current State & Known Issues

| Area | Status | Notes |
|------|--------|-------|
| virtuose_server_node (C++) | ✅ Working | 150 Hz impedance loop, stable |
| teleop_triago_clutch.py | ✅ Working | Clutch-indexing, 180° frame flip; follows `/shared_autonomy/active_arm` to switch between left/right EE slices; topic-routes via `cfg.BLENDING` (§4.3, unpublished/published topic pair) |
| haptic_force_manager_tutorial.py | ✅ Working (`cfg.BLENDING=False`) | `DEBUG_ONLY_GUIDE=True`: F_guide (velocity-field) + F_fixture (position spring near goal). `grasp_active` takes precedence → drag handle to follow EE motion during autonomous phases (KP=30 + KD=160 velocity-following). MAX_GUIDE_TORQUE=0.10 Nm. Follows active arm via `/shared_autonomy/active_arm` (reads correct arm's reference + EE slice). |
| haptic_force_manager_blending_tutorial.py | 🔧 New (2026-07-03), untested on real hardware (`cfg.BLENDING=True`) | Renders ONLY F_sync (Kp_sync=35.0, Kp_sync_ang=1.0 — boosted vs. the shared F_sync in the tutorial variant since it is now the SOLE force channel); no local blending math — reads `/shared_autonomy/blend_debug` verbatim for its "Authority Share" plot. See §4.3. |
| Passivity controller | ⚠️ Disabled | `ENABLE_PASSIVITY_CONTROL = False`; tuning pending |
| calibration_main.cpp | ✅ Working | Joint limits discovered and documented |
| Bimanual arm switch | ✅ Working | Both nodes (teleop + force mgr) subscribe to `/shared_autonomy/active_arm`; switch publishers/EE slices dynamically. Teleop re-anchors from the new arm's EE pose on switch. |

---

## 15. Known Issues / Next Steps for the Next Agent

| Issue | Description | Proposed Fix |
|-------|-------------|-------------|
| **Residual inactive-arm motion** | The inactive arm moves when the active arm moves fast. Root cause: the single shared scalar SoftMin CBF row `J_soft·q̇ ≥ b` mixes BOTH arms' Jacobian columns, so the global QP optimizer recruits the inactive arm's joints to cheaply satisfy the barrier. | **Per-arm SoftMin split**: instead of one scalar CBF row, emit two: right-involving pairs → one row touching only right joints; left-involving pairs → one row touching only left joints; inter-arm pairs contribute a shared row over both (keeps inter-arm safety). This makes the inactive arm's cost penalty (2× DAMP, MAX slack, GAMMA_MAX) actually effective since no barrier demand touches its joints unless the inter-arm pair itself is active. Contained change in `collision_manager.compute_softmin_jacobian` (return per-arm `J_soft`/`h_soft`) + `qp_formulator` (two CBF rows instead of one). |
| **Gazebo LinkAttacher dual-attach** | The IFRA_LinkAttacher plugin has a global `IsAttached` boolean that allows only ONE attachment in the whole world. A patched `gazebo_link_attacher.cpp` was provided to the user (per-pair gating, vector-based erase in Detach) but not yet confirmed working in simulation. | User must replace the plugin source, rebuild `ros2_linkattacher`, and verify two simultaneous attach/detach calls. |
| **Pure-user integrator drift (Twist Blending mode, §4.3)** | `teleop_triago_clutch.py`'s own `ref_pos`/`ref_rot` state keeps integrating from the raw Haption twist regardless of which topic it publishes to. At sustained high `alpha` the real robot EE (pulled by the blend) can drift away from this pure-user pose over time; only `F_sync`'s spring on the HANDLE reflects that divergence — the user-side integrator itself is never corrected. | Consider periodically re-anchoring `ref_pos`/`ref_rot` toward the real EE (via `/qp_debug/ee_real`) once divergence exceeds a threshold, similar to the existing re-anchor-on-arm-switch / re-anchor-on-grasp-done logic. Not implemented — flagged for whoever tests this mode first. |

---

## 13. Coding Conventions

- **Config**: tunable parameters are class-level constants in each script (no shared config file yet).
- **Naming**: snake_case for files and variables, PascalCase for classes.
- **Frame mapping**: always apply the 180° Z-flip explicitly (negate X, negate Y, keep Z) — never rely on implicit frame assumptions.
- **Console output**: no per-tick spam. Only startup banners, state transitions, and warnings.
- **Plots**: matplotlib on main thread, ROS spin on daemon thread (same pattern as triago_control).

---

## 14. Git Workflow

- **main** branch: stable, runnable code
- Direct push to main (same as triago_control — no PRs)
- Pull command: `cd ~/exchange/ros2-ws/src/haption_teleoperation && git pull origin main && cd ~/exchange/ros2-ws && colcon build --packages-select haption_teleoperation && source install/setup.bash`
