#include <signal.h>
#include "mujoco_node.h"

//------------------------------------------ main --------------------------------------------------

//**************************
// run event loop
int main(int argc, char **argv)
{
  ros::init(argc, argv, "mujoco_sim");
  ros::NodeHandle nh;
  simulate_loop(nh, true);

  return 0;
}
