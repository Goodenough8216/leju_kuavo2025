#pragma once

#include <string>
#include <vector>

#include <ros/ros.h>
#include <std_msgs/Bool.h>

#include <humanoid_interface/gait/ModeSequenceTemplate.h>
#include "kuavo_msgs/switchGaitByName.h"

namespace ocs2
{
  namespace humanoid
  {

    /** This class implements ModeSequence communication using ROS. */
    class GaitSwitchByNamePublisher
    {
    public:
      GaitSwitchByNamePublisher(ros::NodeHandle nodeHandle, const std::string &gaitFile, const std::string &robotName, bool verbose = false);

    private:
      void switchGaitByNameCallback(const kuavo_msgs::switchGaitByName::ConstPtr &msg);
      std::vector<std::string> gaitList_;
      std::map<std::string, ModeSequenceTemplate> gaitMap_;

      ros::Publisher modeSequenceTemplatePublisher_;
      ros::Subscriber switchGaitByNameSubscriber_;
    };

  } // namespace humanoid
} // end of namespace ocs2
