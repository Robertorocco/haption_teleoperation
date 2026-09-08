// Virtuose server: 150 Hz impedance-mode driver bridging the Haption device to ROS 2.

// Libraries
#include <iostream>
#include <chrono>
#include <array>
#include <algorithm>
#include <cmath>
#include "VirtuoseAPI.h"

// ROS 2 Libraries
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/wrench.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>

using namespace std;
using namespace std::chrono_literals;

// Connects to the LOCAL SvcHaptic bridge daemon, not the device directly: the port after '#'
// is that daemon's per-unit portToApi, so the IP token here always stays 127.0.0.1.
#define VIRTUOSE_IPADDRESS         ("127.0.0.1#53211")
#define VIRTUOSE_FREQUENCY         (150) // Hz

// Driver-level safety bound on the total wrench actually sent to the device.
#define MAX_FORCE_N                (5.0f)
#define MAX_TORQUE_NM              (0.5f)

// Publishes handle pose/velocity/buttons/joints, applies the commanded wrench each tick.
class VirtuoseServerNode : public rclcpp::Node {
public:
    // Opens the device, creates all virtuose/* topics, starts the 150 Hz loop.
    VirtuoseServerNode() : Node("virtuose_server_node") {

        RCLCPP_INFO(this->get_logger(), "Initializing Virtuose Server Node...");

        int setup_result = SetupVirtuose();
        if (setup_result == -1){
            RCLCPP_FATAL(this->get_logger(), "Virtuose setup failed. Shutting down node.");
            rclcpp::shutdown();
            return;
        }

        damping_lin_ = this->declare_parameter("damping_lin", damping_lin_);
        damping_ang_ = this->declare_parameter("damping_ang", damping_ang_);
        RCLCPP_INFO(this->get_logger(),
            "Local damping: lin=%.4f Ns/m, ang=%.4f Nms/rad (tune with ros2 param set).",
            damping_lin_, damping_ang_);

        slew_lin_ = this->declare_parameter("slew_lin", slew_lin_);
        slew_ang_ = this->declare_parameter("slew_ang", slew_ang_);
        RCLCPP_INFO(this->get_logger(),
            "Command slew limit: lin=%.2f N/s, ang=%.2f Nm/s -> full scale in %.2f s / %.2f s "
            "(0 disables).", slew_lin_, slew_ang_,
            slew_lin_ > 0.0 ? MAX_FORCE_N / slew_lin_ : 0.0,
            slew_ang_ > 0.0 ? MAX_TORQUE_NM / slew_ang_ : 0.0);

        if (debug_mode_) RCLCPP_INFO(this->get_logger(), "Setting up ROS 2 topics...");

        pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose>("virtuose/pose", 10);
        velocity_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("virtuose/velocity", 10);
        button_right_pub_ = this->create_publisher<std_msgs::msg::Bool>("virtuose/button_right", 10);
        button_left_pub_ = this->create_publisher<std_msgs::msg::Bool>("virtuose/button_left", 10);
        // Dead-man: grip presence sensor, true only while the operator physically holds the handle.
        deadman_pub_ = this->create_publisher<std_msgs::msg::Bool>("virtuose/deadman", 10);
        force_sub_ = this->create_subscription<geometry_msgs::msg::Wrench>(
            "virtuose/force_cmd", 10, std::bind(&VirtuoseServerNode::ForceCallback, this, std::placeholders::_1)
        );

        articular_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("virtuose/articular_position", 10);

        // Microsecond period for a precise 150 Hz wall-clock loop.
        auto timer_period = std::chrono::microseconds(1000000 / VIRTUOSE_FREQUENCY);
        timer_ = this->create_wall_timer(
            timer_period, std::bind(&VirtuoseServerNode::TimerCallback, this)
        );

        if (debug_mode_) RCLCPP_INFO(this->get_logger(), "Initialization complete. Entering 150Hz control loop.");
    }

    // Powers off force feedback and closes the device connection cleanly.
    ~VirtuoseServerNode() {
        if (debug_mode_) RCLCPP_INFO(this->get_logger(), "Shutting down, closing device connection...");
        if (VC != NULL) {
            virtSetPowerOn(VC, 0);
            virtClose(VC);
            cout << "Virtuose connection closed cleanly." << "\n";
        }
    }

private:
    // Set to true to re-enable periodic debug prints (per-tick force/hardware-state spam).
    bool debug_mode_ = false;

    VirtContext VC = NULL;

    // Latest wrench command, applied to the handle every tick.
    float current_force[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Damping is computed here, not in a force-manager node, since a damper is only passive
    // when applied in the same tick its velocity was measured. Live-tunable via ros2 param set.
    double damping_lin_ = 0.35;   // Ns/m
    double damping_ang_ = 0.025;  // Nms/rad

    // Slew-rate limit on the commanded wrench: every force manager shares current_force, so a mode
    // switch steps it in one tick; slew_ang floors near 4.7 Nm/s, the vibration cues' own toggle rate.
    double slew_lin_ = 10.0;      // N/s   -> full 5 N in 0.5 s
    double slew_ang_ = 6.0;       // Nm/s  -> above the 4.7 Nm/s cue floor
    // Wrench actually rendered, ramping toward current_force; starts at zero so the first
    // command after startup fades in rather than stepping.
    float applied_force_[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Wrench>::SharedPtr force_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr articular_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr button_right_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr button_left_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr deadman_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Opens the connection and configures impedance mode; returns -1 on failure.
    int SetupVirtuose(){
        VC = virtOpen(VIRTUOSE_IPADDRESS);

        cout << "Connecting to Virtuose..." << "\n";
        if (VC == NULL){
            fprintf(stderr, "Error in virtOpen: %s\n", virtGetErrorMessage(virtGetErrorCode(NULL)));
            return -1;
        }
        cout << "Connection to Virtuose established successfully!" << "\n";
        cout << "time step: " << 1.0f / VIRTUOSE_FREQUENCY << "\n";

        float identity[7] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f};
        // INDEXING_NONE: driver-level indexing off; clutch semantics are implemented in the teleop nodes.
        virtSetIndexingMode(VC, INDEXING_NONE);
        virtSetForceFactor(VC, 1.0f);
        virtSetSpeedFactor(VC, 1.0f);
        virtSetTimeStep(VC, 1.0f / VIRTUOSE_FREQUENCY);
        virtSetBaseFrame(VC, identity);
        virtSetObservationFrame(VC, identity);
        virtSetObservationFrameSpeed(VC, identity);
        // Impedance: force commands in, position/velocity readings out.
        virtSetCommandType(VC, COMMAND_TYPE_IMPEDANCE);
        virtSetPowerOn(VC, 1);

        float null_6 [6] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        virtSetForce(VC, null_6);

        cout << "Waiting 3 seconds for physical motor relays to engage..." << "\n";
        std::this_thread::sleep_for(std::chrono::seconds(3));
        cout << "Motors engaged. Ready!" << "\n";

        return 0;
    }

    // Reads handle pose, physical speed and both button states from the device.
    int VirtuoseStateInterface(float *pose, float *velocity, int *button_right, int *button_left){
        virtGetPosition(VC, pose);
        virtGetPhysicalSpeed(VC, velocity);
        // Device button 1 is physically the left one, 2 the right: read swapped so the topic
        // contract holds, virtuose/button_right = clutch and button_left = grasp trigger.
        virtGetButton(VC, 1, button_left);
        virtGetButton(VC, 2, button_right);

        return 0;
    }


    // Applies the wrench to the handle, logging the Haption error message on failure.
    // Not gated by debug_mode_: a real API failure is a fault, not verbose tracing.
    int VirtuoseCommandInterface(float *force){
        int result = virtSetForce(VC, force);

        if (result == -1) {
            int err_code = virtGetErrorCode(VC);
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Virtuose API Error: %s", virtGetErrorMessage(err_code));
        }
        return result;
    }

    // Moves a 3-vector toward its target by at most max_step, limiting the change vector's
    // magnitude so a transition keeps its direction instead of skewing axis by axis.
    static void SlewLimit(float *applied, const float *target, float max_step) {
        float d[3] = {target[0] - applied[0], target[1] - applied[1], target[2] - applied[2]};
        const float n = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
        if (max_step > 0.0f && n > max_step) {
            const float k = max_step / n;
            d[0] *= k; d[1] *= k; d[2] *= k;
        }
        applied[0] += d[0];
        applied[1] += d[1];
        applied[2] += d[2];
    }

    // Stores the latest commanded wrench; applied to the device by the timer loop.
    void ForceCallback(const geometry_msgs::msg::Wrench::SharedPtr msg) {
        current_force[0] = msg->force.x;
        current_force[1] = msg->force.y;
        current_force[2] = msg->force.z;
        current_force[3] = msg->torque.x;
        current_force[4] = msg->torque.y;
        current_force[5] = msg->torque.z;

        if (debug_mode_) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "[DEBUG FORCE IN] x:%.2f y:%.2f z:%.2f", current_force[0], current_force[1], current_force[2]);
        }
    }

    // 150 Hz loop: read device state, publish all topics, apply the commanded wrench.
    void TimerCallback() {
        float pose[7];
        float velocity[6];
        int button_right = 0;
        int button_left = 0;
        VirtuoseStateInterface(pose, velocity, &button_right, &button_left);

        if (debug_mode_) {
            int power_state = 0;
            virtGetPowerOn(VC, &power_state);

            unsigned int failure_state = 0;
            virtGetFailure(VC, &failure_state);

            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[DEBUG HARDWARE] Actual Motor Power: %d (1=ON, 0=OFF) | Failure State: %d",
                power_state, failure_state);
        }

        geometry_msgs::msg::Pose pose_msg;
        pose_msg.position.x = pose[0];
        pose_msg.position.y = pose[1];
        pose_msg.position.z = pose[2];
        // Haption quaternion order is qx, qy, qz, qw.
        pose_msg.orientation.x = pose[3];
        pose_msg.orientation.y = pose[4];
        pose_msg.orientation.z = pose[5];
        pose_msg.orientation.w = pose[6];
        pose_pub_->publish(pose_msg);

        geometry_msgs::msg::Twist vel_msg;
        vel_msg.linear.x = velocity[0];
        vel_msg.linear.y = velocity[1];
        vel_msg.linear.z = velocity[2];
        vel_msg.angular.x = velocity[3];
        vel_msg.angular.y = velocity[4];
        vel_msg.angular.z = velocity[5];
        velocity_pub_->publish(vel_msg);

        std_msgs::msg::Bool btn_right_msg;
        btn_right_msg.data = (button_right != 0);
        button_right_pub_->publish(btn_right_msg);

        std_msgs::msg::Bool btn_left_msg;
        btn_left_msg.data = (button_left != 0);
        button_left_pub_->publish(btn_left_msg);

        // Published only when the API read succeeds (no message that tick otherwise).
        int dead_man = 0;
        if (virtGetDeadMan(VC, &dead_man) == 0) {
            std_msgs::msg::Bool deadman_msg;
            deadman_msg.data = (dead_man != 0);
            deadman_pub_->publish(deadman_msg);
        }

        float art_pos[6];
        if (virtGetArticularPosition(VC, art_pos) == 0) {
            std_msgs::msg::Float64MultiArray art_msg;
            for(int i=0; i<6; i++) {
                art_msg.data.push_back(art_pos[i]);
            }
            articular_pub_->publish(art_msg);
        }

        // Damping uses the velocity read at the top of THIS tick and is applied before the
        // tick ends, so no transport delay sits between measurement and reaction.
        this->get_parameter("damping_lin", damping_lin_);
        this->get_parameter("damping_ang", damping_ang_);
        this->get_parameter("slew_lin", slew_lin_);
        this->get_parameter("slew_ang", slew_ang_);

        // Limited on the command side only: damping is added afterwards so it keeps reaching
        // the device in the same tick its velocity was measured.
        const float dt = 1.0f / (float)VIRTUOSE_FREQUENCY;
        SlewLimit(&applied_force_[0], &current_force[0], (float)slew_lin_ * dt);
        SlewLimit(&applied_force_[3], &current_force[3], (float)slew_ang_ * dt);

        float total_force[6];
        for (int i = 0; i < 3; i++) {
            total_force[i]     = applied_force_[i]     - (float)damping_lin_ * velocity[i];
            total_force[i + 3] = applied_force_[i + 3] - (float)damping_ang_ * velocity[i + 3];
        }
        for (int i = 0; i < 3; i++) {
            total_force[i]     = std::max(-MAX_FORCE_N,   std::min(MAX_FORCE_N,   total_force[i]));
            total_force[i + 3] = std::max(-MAX_TORQUE_NM, std::min(MAX_TORQUE_NM, total_force[i + 3]));
        }

        VirtuoseCommandInterface(total_force);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VirtuoseServerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
