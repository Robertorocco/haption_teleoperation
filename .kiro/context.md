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
    ├── teleop_triago_clutch.py                    clutch-indexing teleop (Virtual Fixture, cfg.BLENDING=False)
    ├── teleop_triago_joystick.py                  Joystick Mode teleop (cfg.BLENDING=True)
    ├── haptic_force_manager_tutorial.py           active when cfg.BLENDING=False (Virtual Fixture)
    ├── haptic_force_manager_blending_tutorial.py  active when cfg.BLENDING=True (Joystick Mode centering spring)
    ├── teleop_triago.py / teleop_demo_integrator.py   alternate/demo teleop variants
    └── haption_plotter.py / workspace_debug_visualizer.py   debug visualization
```

## 3. Architecture: Two Teleoperation Modes

Mode is selected by `cfg.BLENDING` (`triago_control/qp_controller/config.py`). Both nodes below read the SAME flag at their own startup — no live toggle, restart both after changing it.

The two modes use **different teleop nodes** (`teleop_triago_clutch.py` for Virtual Fixture, `teleop_triago_joystick.py` for Joystick Mode) and **different force nodes** (`haptic_force_manager_tutorial.py` vs `haptic_force_manager_blending_tutorial.py`).

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

### 3.2 Mode B — Joystick Mode (`cfg.BLENDING = True`)

The **only** haptic force is a centering spring; the handle is a spring-centered joystick whose **displacement from home** is the pure user twist. This isolates the raw user twist and breaks the unstable feedback loop of the old design (which fed a robot-state-derived force back onto the handle whose motion was then re-read as user intent). Assistance is applied at the **reference level**: `main_shared_autonomy.py` blends the user twist with the belief-weighted policy (computed from the true EE pose) and is the sole writer of `/arm_*/cartesian_reference`. See `triago_control`'s context.md §5 for the alignment-based arbitration.

```
v_blend = (1 - alpha) * v_user + alpha * pi_policy        (alpha from twist ALIGNMENT, see triago §5)
```

```
Haption pose ─┐
              │  teleop_triago_joystick.py:
              │    v_user = K · (handle_pose − home_pose)   [deadbanded, 180°-Z mapped]
              │    publishes /arm_*/user_cartesian_reference  (pure user twist)
              │    publishes /joystick/home_pose             (live home, single source of truth)
              ▼
     main_shared_autonomy.py: alpha = compute_alpha(align(v_user, pi_policy));
       v_blend integrated persistently every tick; SOLE publisher of /arm_*/cartesian_reference
              ▼
     main_qp_controller.py (QP CLF-CBF)   [unchanged, topic-agnostic]

     haptic_force_manager_blending_tutorial.py: subscribes /joystick/home_pose;
       renders ONLY the restorative spring toward home → virtuose/force_cmd
```

**Home pose** (Haption base frame): position fixed at `JOYSTICK_NEUTRAL_POSITION_M = [0.5, -0.03, -0.03]`; orientation starts from `JOYSTICK_NEUTRAL_ORIENTATION_XYZW` (measured on the device at the operator's comfortable rest orientation) and is **dynamically re-based** to track the gripper's orientation (so "handle at rest" always means "hold current gripper orientation"). The gripper reference that defines this mapping is **per-arm**: captured the first time each arm becomes active, saved/restored across arm switches (returning to an arm resumes its own home, not neutral), and re-anchored after an autonomous grasp. The gripper's rotation away from its **startup reference** is scaled DOWN by `JOYSTICK_ROT_HOME_SCALE = 1.25` (gripper 90° → handle 72°) when building the home orientation — lower scale = tighter (more synchronized) tracking, kept above 1.0 so the handle stays within the Haption's more restrictive rotational workspace. This scaling applies ONLY to the home pose, never to the commanded twist. `teleop_triago_joystick.py` owns and publishes the live home pose so the spring and the twist zero-point stay identical.

**Deadband**: handle displacement below `JOYSTICK_DEADBAND_LIN = 12 cm` / `JOYSTICK_DEADBAND_ANG = ~17°` yields zero user twist (removed radially, continuous at the boundary). It is intentionally large because the centering spring cannot settle the handle to mm/sub-degree precision — a tighter band would read the residual settle-oscillation as spurious user input. A still handle (zero user twist) makes the arbitration fall back to a gentle autonomous crawl (see triago §5).

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

## 5. Teleop Scripts

Both output the same 13-float `Float64MultiArray` protocol: `[pos(3), rpy(3), vel_lin(3), vel_ang(3), task_dim(1)]` (`task_dim`: `6.0` = full 6D, `5.0` = free rotation about the approach axis), run at 150 Hz, and apply the 180°-Z frame map (§7). Both suspend on `/shared_autonomy/grasp_active` and re-anchor on the falling edge, and follow `/shared_autonomy/active_arm`.

**`teleop_triago_clutch.py`** (Virtual Fixture, `cfg.BLENDING=False`): clutch-indexing (mouse-mode). Anchors at `/qp_debug/ee_real`; integrates the Haption velocity into a pose; clutch pressed → pose frozen; publishes to `/arm_*/cartesian_reference`.

**`teleop_triago_joystick.py`** (Joystick Mode, `cfg.BLENDING=True`): the handle is a spring-centered joystick. Reads the Haption **pose** (`virtuose/pose`) and maps its displacement from the home pose to a pure Cartesian twist (magnitude strictly proportional to distance, past the deadband). No pose integration, no clutch. The **pose slots** of the outgoing message carry the live EE pose (so downstream `current_T_user == current_T_EE`); the **velocity slots** carry the twist. Publishes the pure user twist on `/arm_*/user_cartesian_reference` and the live home pose on `/joystick/home_pose`. Owns the dynamic home orientation (§3.2).

## 6. Force Feedback

### 6.0 Joystick Mode spring (`haptic_force_manager_blending_tutorial.py`, `cfg.BLENDING=True`)

The **only** force rendered in Joystick Mode: a spring-damper pulling the handle back to the (dynamic) home pose (§3.2), in the Haption base frame:

```
F_lin = KP_LIN·(home_pos − handle_pos) − KD_LIN·handle_vel_lin       (KP_LIN=60 N/m, KD_LIN=1.0)
Tau   = KP_ANG·rotvec(home_rot · handle_rot⁻¹) − KD_ANG·handle_vel_ang (KP_ANG=1.5 Nm/rad, KD_ANG=0.15)
```

Clipped to `MAX_FORCE=10N` / `MAX_TORQUE=1Nm`. No `F_guide`/`F_fixture`/`F_sync`/`F_cbf`, no clutch-align, no joint-limit vibration — coupling any robot-state-derived force onto the handle is exactly what destabilized the previous design. The home pose target is subscribed from `/joystick/home_pose` (single source of truth = the joystick teleop), falling back to the config neutral until the first message.

### 6.1 Virtual Fixture superposition (`haptic_force_manager_tutorial.py`, `cfg.BLENDING=False`)

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

**Joystick Mode (`cfg.BLENDING=True`) — topics that differ**

| Direction | Topic | Publisher | Subscriber |
|---|---|---|---|
| Haption → Robot (pure user twist) | `/arm_right/user_cartesian_reference` | `teleop_triago_joystick.py` | `main_shared_autonomy.py` |
| Joystick home pose | `/joystick/home_pose` | `teleop_triago_joystick.py` | `haptic_force_manager_blending_tutorial.py` |
| Robot (blended) → QP | `/arm_right/cartesian_reference` | `main_shared_autonomy.py` (sole publisher) | `main_qp_controller.py` |
| Robot → Haption (centering spring) | `virtuose/force_cmd` | `haptic_force_manager_blending_tutorial.py` | `virtuose_server_node` |
| Authority-share telemetry | `/shared_autonomy/blend_debug` | `main_shared_autonomy.py` | (optional) |

The joystick teleop + blending force manager read `virtuose/pose` (handle Cartesian pose); the force manager also reads `virtuose/velocity` (spring damping). All other topics (grasp trigger, device velocity, `/qp_debug/ee_real`) are unchanged between modes.

## 10. Build & Run

```bash
cd ~/exchange/ros2-ws
colcon build --packages-select haption_teleoperation
source install/setup.bash

ros2 run haption_teleoperation virtuose_server_node

# pick the teleop + force node pair matching cfg.BLENDING:
#   BLENDING=False (Virtual Fixture):
ros2 run haption_teleoperation teleop_triago_clutch.py
ros2 run haption_teleoperation haptic_force_manager_tutorial.py
#   BLENDING=True (Joystick Mode):
ros2 run haption_teleoperation teleop_triago_joystick.py
ros2 run haption_teleoperation haptic_force_manager_blending_tutorial.py

ros2 run haption_teleoperation virtuose_calibration      # joint-limit discovery
ros2 run haption_teleoperation haption_plotter.py        # debug plotting
```

## 11. Coding Conventions

- Tunable parameters are class-level constants per script (no shared config file in this repo — the cross-package flag `cfg.BLENDING` lives in `triago_control`).
- snake_case files/variables, PascalCase classes.
- Frame mapping: always apply the 180° Z-flip explicitly — never assume implicit frames.
- No per-tick console spam — only startup banners, state transitions, warnings.
- Matplotlib plots on the main thread, ROS spin on a daemon thread.

## 12. User Workspace Paths

- **Colcon workspace**: `~/exchange/ros2-ws/`
- **This repo clone location**: `~/exchange/ros2-ws/src/haption_teleoperation`

## 13. Git Workflow

- Push directly to `main` (no feature branches / PRs for this repo).
- **After every push**, ALWAYS provide the user with the exact commands to sync their local machine:

```bash
cd ~/exchange/ros2-ws/src/haption_teleoperation
git checkout -- .
git pull origin main
cd ~/exchange/ros2-ws
colcon build --packages-select haption_teleoperation
source install/setup.bash
```
