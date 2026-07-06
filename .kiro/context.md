# AI Agent Context — haption_teleoperation

> **This file is maintained by the AI agent.**

## 0. Maintenance Rules

1. **Always share the pull/rebuild command** with the user immediately after pushing any change to this repo (see §9 for the exact sequence). Never wait to be asked.
2. **Keep this file clean.** It must contain only the math formulations and core architectural concepts an AI agent needs to work on this project — no dated changelogs, "Earlier:"/"Last updated:" narratives, or bugfix stories. When something changes, update the relevant section **in place**. Historical detail belongs in git commit messages, not here.

---

## 1. Project Identity

- **Package**: `haption_teleoperation` — ROS 2 Humble, `ament_cmake` (hybrid C++/Python)
- **Robot**: PAL Robotics TRIAGo++ (bimanual, mobile base) — teleoperated via Haption Virtuose 6D
- **Repository**: https://github.com/Robertorocco/haption_teleoperation
- **Sibling package**: `triago_control` (QP controller + shared autonomy). This package cross-imports `triago_control.qp_controller.config` (`package.xml` depends on `triago_control`) — `cfg.BLENDING` there is the single source of truth for which teleoperation mode is active (§3).

## 2. Package Structure

```
haption_teleoperation/
├── include/VirtuoseAPI.h        proprietary C header (Haption S.A.)
├── lib/libVirtuoseAPI.so        proprietary device driver
├── src/                         C++ nodes (hardware API layer)
│   ├── virtuose_server_node.cpp     150Hz impedance-mode device server
│   └── calibration_main.cpp         manual joint-limit discovery tool
└── scripts/                    Python nodes (teleop + force feedback)
    ├── teleop_triago_clutch.py                   clutch-indexing teleop, topic-routes via cfg.BLENDING
    ├── haptic_force_manager_tutorial.py          active when cfg.BLENDING=False (Virtual Fixture)
    ├── haptic_force_manager_blending_tutorial.py active when cfg.BLENDING=True (Twist Blending)
    ├── teleop_triago.py / teleop_demo_integrator.py   alternate/demo teleop variants
    └── haption_plotter.py / workspace_debug_visualizer.py   debug visualization
```

## 3. Architecture: Two Teleoperation Modes

Mode is selected by `cfg.BLENDING` (`triago_control/qp_controller/config.py`). Both nodes below read the SAME flag at their own startup — no live toggle, restart both after changing it.

### 3.1 Mode A — Virtual Fixture (`cfg.BLENDING = False`)

The user's raw reference reaches the QP unmodified; ALL assistance is rendered as haptic **force** only.

```
Haption device → teleop_triago_clutch.py → /arm_right/cartesian_reference → main_qp_controller.py (QP CLF-CBF)
triago_control/main_shared_autonomy.py → /shared_autonomy/{goal_names, goal_probabilities, ee_policy, user_policy, active_goal_pose}
                                                    ↓
                          haptic_force_manager_tutorial.py → F_guide + F_fixture
                                                    ↓
                                  virtuose/force_cmd → Haption device (user feels the force)
```

**Authority handover during grasp execution**: `main_shared_autonomy.py` publishes `/shared_autonomy/grasp_active` (Bool). While `True`, the grasp state machine drives `/arm_*/cartesian_reference` directly and `teleop_triago_clutch.py` freezes; on the falling edge the clutch re-anchors at the post-grasp EE pose.

### 3.2 Mode B — Twist Blending (`cfg.BLENDING = True`)

Assistance is applied at the **reference level** (the actual Cartesian command sent to the QP is a blend), not as force. See `triago_control`'s context.md §5 for the belief/alpha computation this depends on.

```
v_blend = (1 - alpha) * v_user + alpha * pi_policy        (twist-level blend, computed in main_shared_autonomy.py)
```

```
Haption → teleop_triago_clutch.py → /arm_*/user_cartesian_reference   (pure user intent, BLENDING=True)
                                              ↓
                     main_shared_autonomy.py: alpha = compute_alpha(belief);
                       v_blend integrated persistently every tick;
                       SOLE publisher of /arm_*/cartesian_reference;
                       publishes /shared_autonomy/blend_debug
                                              ↓
                          main_qp_controller.py (QP CLF-CBF)   [unchanged, topic-agnostic]
```

`haptic_force_manager_blending_tutorial.py` renders **only `F_sync`** (no `F_guide`/`F_fixture`/`F_cbf`, since assistance now happens at the reference level). It tethers the handle to the real EE using the **pure user pose** (`/arm_*/user_cartesian_reference`, never the blended one — reading the blended pose would hide the divergence the operator is meant to feel). Its telemetry plot reads `/shared_autonomy/blend_debug` (19 floats: `[alpha, v_user(6), v_policy(6), v_blend(6)]`) verbatim rather than recomputing the blend.

## 4. C++ Node: virtuose_server_node

- **Frequency**: 150 Hz. **Command mode**: `COMMAND_TYPE_IMPEDANCE` (force in, position out). **Indexing**: `INDEXING_NONE` (button held to track).
- **IP**: `127.0.0.1#53210` via `libtirpc`. Startup: open → configure → power on → 3s relay wait → loop.

| Published | Type | Content |
|---|---|---|
| `virtuose/pose` | Pose | handle position + quat [x,y,z,w] |
| `virtuose/velocity` | Twist | 6-DOF spatial velocity (device frame) |
| `virtuose/button` | Bool | right button (clutch) |
| `virtuose/articular_position` | Float64MultiArray | 6 joint positions (rad) |

| Subscribed | Type | Content |
|---|---|---|
| `virtuose/force_cmd` | Wrench | 6-DOF wrench applied to the handle |

## 5. Key Script: teleop_triago_clutch.py

Clutch-indexing (mouse-mode) teleoperation:
- Initializes by anchoring at `/qp_debug/ee_real`.
- Frame mapping: Haption→TRIAGo = 180° rotation about Z (§7).
- Clutch pressed → pose frozen, zero velocity published; released → integration resumes from frozen pose.
- Output: 13-float `Float64MultiArray` on `/arm_*/cartesian_reference` (or `/arm_*/user_cartesian_reference` when `cfg.BLENDING=True`): `[pos(3), rpy(3), vel_lin(3), vel_ang(3), task_dim(1)]`. `task_dim`: `6.0` = full 6D, `5.0` = free rotation about the approach axis.
- Runs at 150 Hz, matching the device server.

## 6. Force Feedback (haptic_force_manager_*)

Multi-layer force superposition, summed and clipped to `MAX_FORCE=10N`/`MAX_TORQUE=1Nm`:

| Layer | Symbol | Formula / description |
|---|---|---|
| Sync | F_sync | Spring-damper (Kp=10, Kd=0) tethering handle to tracking error |
| CBF | F_cbf | Repulsive force from collision-barrier gradient × λ_cbf, tanh-saturated, LPF (α=0.15) |
| Guide (Virtual Fixture only) | F_guide | Velocity-field: `F = D·(v_field − v_handle)·confidence`, `v_field = map(pi_blend)`; intrinsically damped (fades as handle reaches v_field, vanishes at goal) |
| Fixture (Virtual Fixture only) | F_fixture | Position+orientation spring toward `active_goal_pose`, gated by belief confidence (`FIX_CONF_LO=0.55 → HI=0.85`); does not weaken near goal (unlike F_guide) |
| Limit | F_limit | 75 Hz square-wave vibration near Haption joint limits |
| Clutch align | — | Rotational spring (K=10 Nm/rad) toward target orientation during clutch |
| Global damping | — | Viscous Kd_lin=0.7, Kd_ang=0.1 |

**F_guide belief blend**: `pi_blend = Σ_k w(k)·pi_k` (convex combination over goal policies); `error_v = pi_blend − v_user`; confidence gain `alpha = smoothstep(1 − H_norm)` where `H_norm` is normalized belief entropy (transparent at uniform belief, full guidance when one goal dominates).

**Passivity architecture**: Observer integrates `power = −(wrench · twist)`; Controller injects dissipative damping `β·v` when energy < 0, saturated at `MAX_PC_FORCE=5N`/`MAX_PC_TORQUE=0.5Nm` (toggle `ENABLE_PASSIVITY_CONTROL`, currently `False`).

## 7. Frame Convention (Haption ↔ TRIAGo)

Haption base frame: X toward user, Y right (operator perspective). TRIAGo `base_footprint`: X forward, Y left. Relationship is a pure **180° rotation about Z**:

```
TRIAGo_vel.x = -Haption_vel.x
TRIAGo_vel.y = -Haption_vel.y
TRIAGo_vel.z = +Haption_vel.z          (same pattern for angular velocity)
```

Force feedback (Haption ← TRIAGo) uses the **same** negation (transpose of a 180° rotation = itself).

## 8. Haption Device Joint Limits (from `calibration_main.cpp`)

| Joint | Min (rad) | Max (rad) |
|---|---|---|
| J1 | -0.804 | +0.782 |
| J2 | -1.650 | -0.065 |
| J3 | +0.728 | +2.498 |
| J4 | -3.024 | +2.820 |
| J5 | -1.282 | +1.047 |
| J6 | -2.054 | +2.095 |

Vibration warning at `LIMIT_OUTER=0.25 rad` from a limit; maximum at `LIMIT_INNER=0.15 rad`.

## 9. Topic Interface

**Virtual Fixture mode (`cfg.BLENDING=False`)**

| Direction | Topic | Publisher | Subscriber |
|---|---|---|---|
| Haption → Robot | `/arm_right/cartesian_reference` | `teleop_triago_clutch.py` | `main_qp_controller.py` |
| Robot → Haption | `virtuose/force_cmd` | `haptic_force_manager_tutorial.py` | `virtuose_server_node` |
| Inference → Force | `/shared_autonomy/goal_names`, `goal_probabilities`, `user_policy` | `main_shared_autonomy.py` | `haptic_force_manager_tutorial.py` |
| Robot state | `/qp_debug/ee_real` | `main_qp_controller.py` | both teleop scripts |
| CBF telemetry | `/collision_constraints`, `/qp_debug/lambda_cbf` | `main_qp_controller.py` | `haptic_force_manager_tutorial.py` |
| Authority handover | `/shared_autonomy/grasp_active` | `main_shared_autonomy.py` | `teleop_triago_clutch.py` |
| Virtual fixture | `/shared_autonomy/active_goal_pose` | `main_shared_autonomy.py` | `haptic_force_manager_tutorial.py` |

**Twist Blending mode (`cfg.BLENDING=True`) — topics that differ**

| Direction | Topic | Publisher | Subscriber |
|---|---|---|---|
| Haption → Robot (pure user intent) | `/arm_right/user_cartesian_reference` | `teleop_triago_clutch.py` | `main_shared_autonomy.py` |
| Robot (blended) → QP | `/arm_right/cartesian_reference` | `main_shared_autonomy.py` (sole publisher) | `main_qp_controller.py` |
| Robot → Haption | `virtuose/force_cmd` | `haptic_force_manager_blending_tutorial.py` | `virtuose_server_node` |
| Authority-share telemetry | `/shared_autonomy/blend_debug` | `main_shared_autonomy.py` | `haptic_force_manager_blending_tutorial.py` |

All other topics (grasp trigger, clutch, device velocity, `/qp_debug/ee_real`) are unchanged between modes.

## 10. Build & Run

```bash
cd ~/exchange/ros2-ws
colcon build --packages-select haption_teleoperation
source install/setup.bash

ros2 run haption_teleoperation virtuose_server_node
ros2 run haption_teleoperation teleop_triago_clutch.py

# pick the force node matching cfg.BLENDING:
ros2 run haption_teleoperation haptic_force_manager_tutorial.py            # BLENDING=False
ros2 run haption_teleoperation haptic_force_manager_blending_tutorial.py   # BLENDING=True

ros2 run haption_teleoperation virtuose_calibration      # joint-limit discovery
ros2 run haption_teleoperation haption_plotter.py        # debug plotting
```

## 11. Coding Conventions

- Tunable parameters are class-level constants per script (no shared config file in this repo — the cross-package flag `cfg.BLENDING` lives in `triago_control`).
- snake_case files/variables, PascalCase classes.
- Frame mapping: always apply the 180° Z-flip explicitly — never assume implicit frames.
- No per-tick console spam — only startup banners, state transitions, warnings.
- Matplotlib plots on the main thread, ROS spin on a daemon thread.

## 12. Git Workflow

- Push directly to `main` (no feature branches / PRs for this repo).
- **After every push**, run for the user on their local machine:

```bash
cd ~/exchange/ros2-ws/src/haption_teleoperation
git checkout -- .
git pull origin main
cd ~/exchange/ros2-ws
colcon build --packages-select haption_teleoperation
source install/setup.bash
```
