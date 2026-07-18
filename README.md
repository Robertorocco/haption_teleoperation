# haption_teleoperation

Operator-side half of the TRIAGo shared-autonomy teleoperation system: it drives a
**Haption Virtuose 6-DOF** haptic device and renders force feedback, while the sibling
package [`triago_control`](https://github.com/Robertorocco/triago_control) runs the
QP-CLF-CBF safety controller and the shared-autonomy inference stack on the robot side.
The two packages run together and communicate over live ROS 2 topics.

## Architecture

```
haption_teleoperation/
├── include/VirtuoseAPI.h            proprietary C header (Haption S.A.)
├── lib/libVirtuoseAPI.so            proprietary device driver
├── src/
│   ├── virtuose_server_node.cpp     150 Hz impedance-mode device server (root of all virtuose/* topics)
│   └── calibration_main.cpp         manual joint-limit discovery tool
└── scripts/
    ├── teleop_triago_clutch.py      CLUTCH teleop: position control with indexing (clutch button)
    ├── teleop_triago_joystick.py    JOYSTICK teleop: spring-centered handle, displacement -> twist
    ├── haptic_force_manager_<CELL>.py   one force renderer per study condition (see below)
    ├── haption_plotter.py           live device telemetry plots
    └── workspace_debug_visualizer.py    workspace/frame mapping debug views
```

**Frame convention:** the Haption base frame and the TRIAGo base frame differ by a pure
180° rotation about Z (negate X and Y, keep Z). The same map carries motion commands
(device → robot) and rendered forces (robot → device).

## The 2×2×2 study matrix

The active experiment condition is selected **once**, in `triago_control`'s
`qp_controller/config.py` (section 1b), by three orthogonal flags:

- `CONTROL_MODE` ∈ {`CLUTCH`, `JOYSTICK`} — position vs velocity control;
- `ASSIST_FEEDBACK` — channel F: assistive guidance forces on the handle;
- `ASSIST_BLENDING` — channel B: reference-level user↔policy blending (robot side).

Every teleop and force-manager node validates the selected cell at startup and refuses
to run on a mismatch. The eight force managers are named `haptic_force_manager_<CELL>`,
where `<CELL>` = `C`/`J` (mode) + `F` if feedback + `B` if blending:

| Cell | Mode | Operator feels |
|---|---|---|
| `C`   | clutch   | EE sync tether + cues (baseline) |
| `CF`  | clutch   | tether + guidance bias |
| `CB`  | clutch   | tether (to the blended reference) |
| `CFB` | clutch   | tether + guidance bias, reference blended |
| `J`   | joystick | homing spring + cues (baseline) |
| `JF`  | joystick | spring + guidance bias |
| `JB`  | joystick | homing spring, reference blended |
| `JFB` | joystick | spring + guidance bias, reference blended |

## Build & run

```bash
cd ~/exchange/ros2-ws
colcon build --packages-select haption_teleoperation
source install/setup.bash

ros2 run haption_teleoperation virtuose_server_node

# pick the teleop + force-manager pair matching the condition in config.py, e.g.:
ros2 run haption_teleoperation teleop_triago_clutch.py
ros2 run haption_teleoperation haptic_force_manager_C.py

ros2 run haption_teleoperation virtuose_calibration     # joint-limit discovery
ros2 run haption_teleoperation haption_plotter.py       # debug plotting
```

The robot side (`main_qp_controller*.py`, `main_shared_autonomy.py`) must run from
`triago_control`; see that repo's README for the full launch sequence.

## Device topics

| Topic | Type | Direction | Content |
|---|---|---|---|
| `virtuose/pose` | `geometry_msgs/Pose` | out | handle position + quaternion (x,y,z,w) |
| `virtuose/velocity` | `geometry_msgs/Twist` | out | handle 6-DOF spatial velocity |
| `virtuose/button_right` | `std_msgs/Bool` | out | clutch button |
| `virtuose/button_left` | `std_msgs/Bool` | out | grasp trigger / double-click arm switch |
| `virtuose/deadman` | `std_msgs/Bool` | out | grip presence sensor (true while held) |
| `virtuose/articular_position` | `Float64MultiArray` | out | 6 device joint positions (rad) |
| `virtuose/force_cmd` | `geometry_msgs/Wrench` | in | 6-DOF wrench applied to the handle |
