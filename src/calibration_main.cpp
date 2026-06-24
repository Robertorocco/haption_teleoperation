#include <iostream>
#include <chrono>
#include <thread>
#include <csignal>
#include <vector>
#include "VirtuoseAPI.h"

using namespace std;

#define VIRTUOSE_IPADDRESS ("127.0.0.1#53210") // Change to your actual IP if needed

// Global flag to catch Ctrl+C
volatile sig_atomic_t keep_running = 1;
void sig_handler(int) { 
    keep_running = 0; 
}

int main()
{
    // Register the signal handler for Ctrl+C
    signal(SIGINT, sig_handler);

    // ====================================> Open Connection
    VirtContext VC;
    cout << "Connecting to Virtuose using " << VIRTUOSE_IPADDRESS << endl;
    VC = virtOpen(VIRTUOSE_IPADDRESS);

    if (VC == NULL) {
        fprintf(stderr, "Error in virtOpen: %s\n", virtGetErrorMessage(virtGetErrorCode(NULL)));
        return -1;
    }

    // ====================================> Configure Virtuose
    float identity[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    virtSetIndexingMode(VC, INDEXING_NONE); 
    virtSetForceFactor(VC, 1.0f);
    virtSetSpeedFactor(VC, 1.0f);
    virtSetTimeStep(VC, 0.003f);
    virtSetBaseFrame(VC, identity);
    virtSetObservationFrame(VC, identity);
    virtSetObservationFrameSpeed(VC, identity);
    virtSetCommandType(VC, COMMAND_TYPE_IMPEDANCE);
    virtSetPowerOn(VC, 1);
    
    // Arrays to hold the 6 joint limits
    float current_pos[6] = {0.0f};
    float min_pos[6] = {1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f};
    float max_pos[6] = {-1000.0f, -1000.0f, -1000.0f, -1000.0f, -1000.0f, -1000.0f};

    cout << "\n=====================================================" << endl;
    cout << " CALIBRATION ACTIVE" << endl;
    cout << " -> Power on the device." << endl;
    cout << " -> Push the handle manually to every mechanical limit." << endl;
    cout << " -> Press Ctrl+C when you are completely finished." << endl;
    cout << "=====================================================\n" << endl;

    int iter = 0;

    // Run until user presses Ctrl+C
    while (keep_running)
    {
        // 1. Send zero force to keep the robot completely transparent
        float null_f[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        virtSetForce(VC, null_f);

        // 2. Read articular (joint) positions
        if (virtGetArticularPosition(VC, current_pos) == 0) {
            // 3. Update min and max limits
            for (int i = 0; i < 6; i++) {
                if (current_pos[i] < min_pos[i]) min_pos[i] = current_pos[i];
                if (current_pos[i] > max_pos[i]) max_pos[i] = current_pos[i];
            }
        }

        // 4. Print live feedback occasionally so you know it's working
        if (iter % 200 == 0) {
            cout << "\rTracking... Joint 1: [" << min_pos[0] << " to " << max_pos[0] << "]   " << flush;
        }
        
        iter++;
        this_thread::sleep_for(chrono::milliseconds(5)); // ~200Hz loop
    }

    // ====================================> Shutdown & Print Results
    cout << "\n\nStopping device..." << endl;
    virtSetPowerOn(VC, 0); 
    virtClose(VC);

    cout << "\n=====================================================" << endl;
    cout << " CALIBRATION COMPLETE! COPY THE VALUES BELOW:" << endl;
    cout << "=====================================================" << endl;
    
    cout << "virtuose_joint_limits:" << endl;
    cout << "  min: [";
    for(int i=0; i<6; i++) cout << min_pos[i] << (i < 5 ? ", " : "");
    cout << "]" << endl;

    cout << "  max: [";
    for(int i=0; i<6; i++) cout << max_pos[i] << (i < 5 ? ", " : "");
    cout << "]" << endl;
    cout << "=====================================================\n" << endl;

    return 0;
}