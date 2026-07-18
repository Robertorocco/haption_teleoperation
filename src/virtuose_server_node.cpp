// Virtuose server: 150 Hz impedance-mode driver bridging the Haption device to ROS 2.

// Libraries
#include <iostream>
#include <chrono>
#include <array>
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

#define VIRTUOSE_IPADDRESS         ("127.0.0.1#53210")
#define VIRTUOSE_FREQUENCY         (150) // Hz

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
    // Set to false to silence periodic debug prints.
    bool debug_mode_ = true;

    VirtContext VC = NULL;

    // Latest wrench command, applied to the handle every tick.
    float current_force[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

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
        // Button 1 = right (clutch), button 2 = left (grasp trigger / arm switch).
        virtGetButton(VC, 1, button_right);
        virtGetButton(VC, 2, button_left);

        return 0;
    }


    // Applies the wrench to the handle, logging the Haption error message on failure.
    int VirtuoseCommandInterface(float *force){
        int result = virtSetForce(VC, force);

        if (debug_mode_) {
            if (result == -1) {
                int err_code = virtGetErrorCode(VC);
                RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "Virtuose API Error: %s", virtGetErrorMessage(err_code));
            }
        }
        return result;
    }

    // Stores the latest commanded wrench; it is applied by the timer loop.
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

        VirtuoseCommandInterface(current_force);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VirtuoseServerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
