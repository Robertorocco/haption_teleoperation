# Project context

Before doing any work in this repo, read `.kiro/context.md` — it holds the current state of the project (architecture, in-progress work, decisions) and is the primary source of truth for Claude sessions here, alongside README.md for user-facing info.

It is Claude's responsibility to keep `.kiro/context.md` accurate and up to date as work happens in this repo: update it when architecture changes, features land, or decisions are made, so a new chat can pick up full context from this file alone.

## Sibling repo: triago_control

`~/exchange/ros2-ws/src/triago_control` runs the QP controller and shared-autonomy stack that this repo's teleop/force-manager scripts drive and receive telemetry from. They run together at runtime and share a live ROS2 topic interface — a change on one side routinely requires a matching change on the other.

**If a task touches the teleop loop, grasp state machine, blending, or force feedback, also read `../triago_control/.kiro/context.md` before making changes** — don't rely on the table below alone for anything beyond a quick topic lookup.

### Runtime interface (verified 2026-07-10)

**This repo → triago_control** (user commands)

| Topic | Type | Published by (this repo) | Consumed by (triago_control) |
|---|---|---|---|
| `/arm_right/cartesian_reference`, `/arm_left/cartesian_reference` | `Float64MultiArray` | `teleop_triago_clutch.py` / `teleop_triago_joystick.py` (only when `cfg.BLENDING`/`ASSIST_BLENDING=False`) | `scripts/qp_arm_teleop/main_qp_controller.py` (`ref_cb_right/left`) |
| `/arm_right/user_cartesian_reference`, `/arm_left/user_cartesian_reference` | `Float64MultiArray` | same scripts, redirected here when blending is on | `scripts/qp_arm_teleop/main_shared_autonomy.py` (`sub_human_reference_right/left`) |
| `virtuose/button_left` | `std_msgs/Bool` | `virtuose_server_node.cpp` (this repo) | `main_shared_autonomy.py` (`sub_trigger` — grasp/switch-arm) |
| `virtuose/button_right` | `std_msgs/Bool` | `virtuose_server_node.cpp` (this repo) | `main_shared_autonomy.py` (`sub_clutch` — suspend blending while indexing) |

Note: triago_control's `main_shared_autonomy.py` also publishes `/arm_*/cartesian_reference` (`pub_blend_right/left`) and becomes the sole writer once `ASSIST_BLENDING=True` — at that point this repo's teleop scripts stop writing that topic and only write `user_cartesian_reference`.

**triago_control → this repo** (state / force-feedback context)

| Topic | Type | Published by (triago_control) | Consumed by (this repo) |
|---|---|---|---|
| `/qp_debug/ee_real` | `Float64MultiArray` | `main_qp_controller.py` (`pub_ee_state`) | `teleop_triago_{clutch,joystick}.py`, `haptic_force_manager_{C,CF,CFB}.py` |
| `/collision_constraints` | `Float64MultiArray` | `main_qp_controller.py` (`pub_shared_col`) | `haptic_force_manager_{CF,CFB,CB}.py` (`cbf_gradient_cb`) |
| `/qp_debug/lambda_cbf` | `Float64MultiArray` | `main_qp_controller.py` (`pub_lambda_cbf`) | `haptic_force_manager_{CF,CFB,CB}.py` (`lambda_cb`) |
| `/shared_autonomy/grasp_active` | `std_msgs/Bool` | `main_shared_autonomy.py` (`pub_grasp_active`) | all teleop + force-manager scripts |
| `/shared_autonomy/active_arm` | `std_msgs/String` | `main_shared_autonomy.py` (`pub_active_arm`) | all teleop + force-manager scripts |
| `/shared_autonomy/goal_names`, `goal_probabilities`, `user_policy`, `active_goal_pose` | `String`/`Float64MultiArray` | `main_shared_autonomy.py` | `haptic_force_manager_{J,C}F/JFB/CFB/CB/JB.py` |
| `/shared_autonomy/blend_debug` | `Float64MultiArray` (19 floats: `alpha, v_user(6), v_policy(6), v_blend(6)`) | `main_shared_autonomy.py` (`pub_blend_debug`) | `haptic_force_manager_{CB,CFB}.py` |

Force feedback to the operator (`virtuose/force_cmd`, `geometry_msgs/Wrench`, → `virtuose_server_node.cpp`, both in this repo) is synthesized locally inside this repo's `haptic_force_manager_*.py` scripts from the triago_control telemetry above — triago_control does not publish a dedicated force/state topic for that purpose.

**Import-level coupling (not a topic):** this repo depends on triago_control's `triago_control.qp_controller.config` module (`cfg.BLENDING`, `cfg.ASSIST_*`, `cfg.validate_condition(...)`) — declared as a `<depend>` in this repo's `package.xml`. If triago_control changes the condition-selector shape in `qp_controller/config.py`, this repo's teleop and force-manager scripts break.

**Key entrypoints in this repo:** `src/virtuose_server_node.cpp` (C++ hardware driver, root of all `virtuose/*` topics), `scripts/teleop_triago_clutch.py` (CLUTCH position-control teleop), `scripts/teleop_triago_joystick.py` (JOYSTICK velocity-control teleop), `scripts/haptic_force_manager_{C,CF,CB,CFB,J,JF,JB,JFB}.py` (per-condition haptic force renderers, selected by `cfg.CONTROL_MODE`/`ASSIST_FEEDBACK`/`ASSIST_BLENDING`).
