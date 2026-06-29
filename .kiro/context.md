# AI Agent Context — haption_teleoperation

> **This file is maintained by the AI agent. Do not edit manually.**
> Last updated: 2026-06-23 (initial context from package upload)

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
    ├── teleop_triago_clutch.py      ★ active: clutch-indexing teleop (mouse-mode)
    ├── haptic_force_manager.py      ★ active: multi-layer force-feedback superposition
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

### 4.2 Position Virtual Fixture (haptic_force_manager_tutorial.py)

`main_shared_autonomy.py` publishes `/shared_autonomy/active_goal_pose`
(`[x,y,z,roll,pitch,yaw,confidence]`). The tutorial force manager applies `F_fixture`: a
position+orientation spring pulling the handle toward the exact grasp pose, gated by confidence
(`FIX_CONF_LO=0.55`→`HI=0.85`). Unlike the viscous `F_guide` (which vanishes at the goal), this
does NOT weaken near the goal, so it lets the operator settle precisely at the grasp standoff
against the CBF push-off. Confidence is forced to 0 during grasp execution (fixture releases).

### 4.3 Force Smoothing (math-sound, anti-vibration) — 2026-06-29

Applying the policy velocity field directly produced buzz/vibration on the handle. Two
complementary, mathematically-principled smoothing stages were added:

1. **Guidance-velocity LPF**: the `F_guide` damping term `D·(v_field − v_handle)` uses a
   low-pass-filtered handle velocity (`vel_haption_f`, α=0.25) instead of the raw device
   velocity. The raw velocity is noisy and `D_guide≈42` amplifies that noise straight into the
   wrench. The raw velocity is still used for the sync / global-damping layers (their stiffer
   terms are lag-sensitive, per the original design note).
2. **Final output filter — 2nd-order critically-damped LPF (ζ=1, fc=12 Hz)** on the wrench
   actually injected into the device (`virtuose/force_cmd`). Critically damped ⇒ no resonant
   peak (cannot ring/amplify) and **C¹** (continuous force AND force-rate), so the handle feels
   smooth and jerk-free — a first-order LPF only gives C⁰. Discrete semi-implicit integration of
   `ÿ = ωn²(u − y) − 2ωn·ẏ`. The hard `MAX_FORCE/MAX_TORQUE` clip is applied AFTER the filter so
   the injected vector is always bounded.

Guidance gains were also bumped ~1.25× (operator wanted slightly stronger assistance):
`D_guide_lin 42`, `D_guide_ang 0.68`, `MAX_GUIDE_FORCE 5.0 N`, `MAX_GUIDE_TORQUE 0.38 Nm`.

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

| Direction | Topic | Publisher | Subscriber |
|-----------|-------|-----------|------------|
| Haption → Robot | `/arm_right/cartesian_reference` | `teleop_triago_clutch.py` | `main_qp_controller.py` (triago_control) |
| Robot → Haption | `virtuose/force_cmd` | `haptic_force_manager.py` | `virtuose_server_node` |
| Inference → Force | `/shared_autonomy/goal_names` | `main_shared_autonomy.py` | `haptic_force_manager.py` |
| Inference → Force | `/shared_autonomy/goal_probabilities` | `main_shared_autonomy.py` | `haptic_force_manager.py` |
| Inference → Force | `/shared_autonomy/user_policy` | `main_shared_autonomy.py` | `haptic_force_manager.py` |
| Robot state | `/qp_debug/ee_real` | `main_qp_controller.py` | both teleop scripts |
| CBF gradient | `/collision_constraints` | `main_qp_controller.py` | `haptic_force_manager.py` |
| CBF slack | `/qp_debug/lambda_cbf` | `main_qp_controller.py` | `haptic_force_manager.py` |
| Clutch | `virtuose/button_right` | `virtuose_server_node` | teleop + force manager |
| Grasp trigger | `virtuose/button_left` | `virtuose_server_node` | `main_shared_autonomy.py` |
| Authority handover | `/shared_autonomy/grasp_active` | `main_shared_autonomy.py` | `teleop_triago_clutch.py` |
| Virtual fixture | `/shared_autonomy/active_goal_pose` | `main_shared_autonomy.py` | `haptic_force_manager_tutorial.py` |
| Device vel | `virtuose/velocity` | `virtuose_server_node` | teleop + force manager |

---

## 11. Build & Run

```bash
# Build
cd ~/exchange/ros2-ws
colcon build --packages-select haption_teleoperation
source install/setup.bash

# Run device server (requires hardware or simulator on 127.0.0.1#53210)
ros2 run haption_teleoperation virtuose_server_node

# Run clutch teleop
ros2 run haption_teleoperation teleop_triago_clutch.py

# Run force feedback
ros2 run haption_teleoperation haptic_force_manager.py

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
| teleop_triago_clutch.py | ✅ Working | Clutch-indexing, 180° frame flip |
| haptic_force_manager.py | 🔧 Active dev | Force layers all functional; F_guide goal hierarchy hardcoded for a previous experiment (Battery/Hole layout) — needs updating for the Red/Blue/Platform goal set in triago_control |
| Passivity controller | ⚠️ Disabled | `ENABLE_PASSIVITY_CONTROL = False`; tuning pending |
| calibration_main.cpp | ✅ Working | Joint limits discovered and documented |

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
