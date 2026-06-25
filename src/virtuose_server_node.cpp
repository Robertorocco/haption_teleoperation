//////////////////////////////////////////////////////////////////////////////////
////////////////////////////// VIRTUOSE SERVER ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

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

// Global Variables
#define VIRTUOSE_IPADDRESS         ("127.0.0.1#53210")
#define VIRTUOSE_FREQUENCY         (150) // Hz


class VirtuoseServerNode : public rclcpp::Node {
public:
    VirtuoseServerNode() : Node("virtuose_server_node") {
        
        RCLCPP_INFO(this->get_logger(), "Initializing Virtuose Server Node...");

        // ====================================> Setup Virtuose
        int setup_result = SetupVirtuose();
        if (setup_result == -1){ // Break the code if setup failed
            RCLCPP_FATAL(this->get_logger(), "Virtuose setup failed. Shutting down node.");
            rclcpp::shutdown();
            return;
        }

        // ===============================> ROS 2 Publishers & Subscribers
        if (debug_mode_) RCLCPP_INFO(this->get_logger(), "Setting up ROS 2 topics...");
        
        pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose>("virtuose/pose", 10);
        velocity_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("virtuose/velocity", 10);
        button_right_pub_ = this->create_publisher<std_msgs::msg::Bool>("virtuose/button_right", 10);
        button_left_pub_ = this->create_publisher<std_msgs::msg::Bool>("virtuose/button_left", 10); 
        // Subscribe to a wrench topic (Force: x,y,z | Torque: x,y,z)
        force_sub_ = this->create_subscription<geometry_msgs::msg::Wrench>(
            "virtuose/force_cmd", 10, std::bind(&VirtuoseServerNode::ForceCallback, this, std::placeholders::_1)
        );
        
        //To show joint value
        articular_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("virtuose/articular_position", 10);

        // ====================================> Main Loop Timer
        // Set Frequency to 150 Hz using microseconds for precision
        auto timer_period = std::chrono::microseconds(1000000 / VIRTUOSE_FREQUENCY);
        timer_ = this->create_wall_timer(
            timer_period, std::bind(&VirtuoseServerNode::TimerCallback, this)
        );

        if (debug_mode_) RCLCPP_INFO(this->get_logger(), "Initialization complete. Entering 150Hz control loop.");
    }

    ~VirtuoseServerNode() {
        // Terminate the communication with Virtuose
        if (debug_mode_) RCLCPP_INFO(this->get_logger(), "Shutting down, closing device connection...");
        if (VC != NULL) {
            virtSetPowerOn(VC, 0); // Disable force-feedback
            virtClose(VC); // Close connection
            cout << "Virtuose connection closed cleanly." << "\n";
        }
    }

private:
    // ===============================> DEBUG FLAG
    // Change to 'false' to disable all console spam when everything works perfectly
    bool debug_mode_ = true; 

    VirtContext VC = NULL;
    
    // ===============================> Command Interface Variables
    float current_force[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // ROS 2 Objects
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Wrench>::SharedPtr force_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr articular_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr button_right_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr button_left_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // ===============================> Setup Device
    int SetupVirtuose(){
        // Open Connection    
        VC = virtOpen(VIRTUOSE_IPADDRESS);

        // Check if connection was successful
        cout << "Connecting to Virtuose..." << "\n";
        if (VC == NULL){
            fprintf(stderr, "Error in virtOpen: %s\n", virtGetErrorMessage(virtGetErrorCode(NULL)));
            return -1;
        }
        cout << "Connection to Virtuose established successfully!" << "\n";
        cout << "time step: " << 1.0f / VIRTUOSE_FREQUENCY << "\n";

        // Configure Device
        float identity[7] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f};
        virtSetIndexingMode(VC, INDEXING_NONE); //INDEXING_ALL = authorizes indexing on rotations and translations; INDEXING_TRANS = Indexing only translation; INDEXING_NONE = Only mode if touch dead-man button;   
        virtSetForceFactor(VC, 1.0f);
        virtSetSpeedFactor(VC, 1.0f);
        virtSetTimeStep(VC, 1.0f / VIRTUOSE_FREQUENCY);
        virtSetBaseFrame(VC, identity);
        virtSetObservationFrame(VC, identity);
        virtSetObservationFrameSpeed(VC, identity);
        virtSetCommandType(VC, COMMAND_TYPE_IMPEDANCE); //COMMAND_TYPE_IMPEDANCE  COMMAND_TYPE_VIRTMECH
        virtSetPowerOn(VC, 1);

        // Set initial force as zero  
        float null_6 [6] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};  
        virtSetForce(VC, null_6);

        cout << "Waiting 3 seconds for physical motor relays to engage..." << "\n";
        std::this_thread::sleep_for(std::chrono::seconds(3));
        cout << "Motors engaged. Ready!" << "\n";

        return 0;
    }

    // ===============================> State Interface
    int VirtuoseStateInterface(float *pose, float *velocity, int *button_right, int *button_left){
        // Get Virtuose Pose
        virtGetPosition(VC, pose);
        // Get Virtuose Velocity
        virtGetPhysicalSpeed(VC, velocity);
        // Get Button States: 1 is right button (clutch), 2 is left button (grasp)
        virtGetButton(VC, 1, button_right);
        virtGetButton(VC, 2, button_left);
        
        return 0;
    }


    // ===============================> Command Interface
    int VirtuoseCommandInterface(float *force){
        // Send Force Command to Virtuose
        int result = virtSetForce(VC, force);
        
        if (debug_mode_) {
            // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
            //     "[DEBUG API] virtSetForce return code: %d | Applied Fx: %.2f", 
            //     result, force[0]);
            
            // If there's an error, print the exact Haption error message
            if (result == -1) {
                int err_code = virtGetErrorCode(VC);
                RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "Virtuose API Error: %s", virtGetErrorMessage(err_code));
            }
        }
        return result;
    }

    // Callback to update the internal force variable asynchronously
    void ForceCallback(const geometry_msgs::msg::Wrench::SharedPtr msg) {
        current_force[0] = msg->force.x;
        current_force[1] = msg->force.y;
        current_force[2] = msg->force.z;
        current_force[3] = msg->torque.x;
        current_force[4] = msg->torque.y;
        current_force[5] = msg->torque.z;

        if (debug_mode_) {
            // Throttled to prevent flooding the terminal if GUI sends at high frequency
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "[DEBUG FORCE IN] x:%.2f y:%.2f z:%.2f", current_force[0], current_force[1], current_force[2]);
        }
    }

    // Main Control Loop Executed at VIRTUOSE_FREQUENCY
    void TimerCallback() {
        // ===============> State Interface Variables
        float pose[7];
        float velocity[6];
        int button_right = 0;
        int button_left = 0;
        // State Interface
        VirtuoseStateInterface(pose, velocity, &button_right, &button_left);

        // Debug prints for read data, throttled to 1 print per second
        // if (debug_mode_) {
        //     RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
        //         "[DEBUG STATE] Pose X:%.3f Y:%.3f Z:%.3f | Vel X:%.3f Y:%.3f Z:%.3f",
        //         pose[0], pose[1], pose[2], velocity[0], velocity[1], velocity[2]);
        // }

        // Add this inside TimerCallback() to monitor true hardware state
        if (debug_mode_) {
            int power_state = 0;
            virtGetPowerOn(VC, &power_state);
            
            unsigned int failure_state = 0;
            virtGetFailure(VC, &failure_state);

            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                "[DEBUG HARDWARE] Actual Motor Power: %d (1=ON, 0=OFF) | Failure State: %d",
                power_state, failure_state);
        }

        // =================> Publish Virtuose State Interface
        geometry_msgs::msg::Pose pose_msg;
        pose_msg.position.x = pose[0];
        pose_msg.position.y = pose[1];
        pose_msg.position.z = pose[2];
        pose_msg.orientation.x = pose[3]; // Assuming Haption quaternion order is qx, qy, qz, qw
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
        
        // =================> Publish Button States
        std_msgs::msg::Bool btn_right_msg;
        btn_right_msg.data = (button_right != 0);
        button_right_pub_->publish(btn_right_msg);

        std_msgs::msg::Bool btn_left_msg;
        btn_left_msg.data = (button_left != 0);
        button_left_pub_->publish(btn_left_msg);

        // =================> NEW: Publish Articular (Joint) Positions <=================
        float art_pos[6];
        if (virtGetArticularPosition(VC, art_pos) == 0) {
            std_msgs::msg::Float64MultiArray art_msg;
            for(int i=0; i<6; i++) {
                art_msg.data.push_back(art_pos[i]);
            }
            articular_pub_->publish(art_msg);
        }

        // =================> Testing Virtuose Command Interface
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