/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include "ocs2_ros_interfaces/synchronized_module/RosReferenceManager.h"

#include "ocs2_ros_interfaces/common/RosMsgConversions.h"

#include <ros/transport_hints.h>

// MPC messages
#include <ocs2_msgs/mode_schedule.h>
#include <ocs2_msgs/mpc_target_trajectories.h>

namespace ocs2 {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
RosReferenceManager::RosReferenceManager(std::string topicPrefix, std::shared_ptr<ReferenceManagerInterface> referenceManagerPtr)
    : ReferenceManagerDecorator(std::move(referenceManagerPtr)), topicPrefix_(std::move(topicPrefix)) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void RosReferenceManager::subscribe(ros::NodeHandle& nodeHandle) {
  referenceManagerPtr_->setupSubscriptions(topicPrefix_);
  // ModeSchedule
  auto modeScheduleCallback = [this](const ocs2_msgs::mode_schedule::ConstPtr& msg) {
    auto modeSchedule = ros_msg_conversions::readModeScheduleMsg(*msg);
    referenceManagerPtr_->setModeSchedule(std::move(modeSchedule));
  };
  modeScheduleSubscriber_ = nodeHandle.subscribe<ocs2_msgs::mode_schedule>(topicPrefix_ + "_mode_schedule", 1, modeScheduleCallback);

  // TargetTrajectories
  auto targetTrajectoriesCallback = [this](const ocs2_msgs::mpc_target_trajectories::ConstPtr& msg) {
    auto targetTrajectories = ros_msg_conversions::readTargetTrajectoriesMsg(*msg);
    //  for (auto &time : targetTrajectories.timeTrajectory)
    // {
    //   std::cout << "Lower limb Target Traj " << std::endl;
    //   std::cout << "target_time:  " << time << std::endl;

    // }
    
    referenceManagerPtr_->setTargetTrajectories(std::move(targetTrajectories));
  };
  targetTrajectoriesSubscriber_ =
      nodeHandle.subscribe<ocs2_msgs::mpc_target_trajectories>(topicPrefix_ + "_mpc_target", 1, targetTrajectoriesCallback);

  // Arm TargetTrajectories
  auto armTargetTrajectoriesCallback = [this](const ocs2_msgs::mpc_target_trajectories::ConstPtr& msg) {
    // std::cout << "Arm tragetTrj working !!!!!!!!!!!!!!!!!!" << std::endl;
    auto targetTrajectories = ros_msg_conversions::readTargetTrajectoriesMsg(*msg);
    auto pre_targetTrajectories = referenceManagerPtr_->getTargetTrajectories();
    //  for (auto &time : pre_targetTrajectories.timeTrajectory)
    // {
    //   std::cout << "Pre ARM Target Traj" << std::endl;
    //   std::cout << "target_time:  " << time << std::endl;
    //   std::cout << "Pre Arm Joint:   " << pre_targetTrajectories.stateTrajectory[1](12+12+1) << std::endl;
    // }
    
    // for (auto &time : targetTrajectories.timeTrajectory)
    // {
    //   std::cout << "ARM Target Traj" << std::endl;
    //   std::cout << "target_time:  " << time << std::endl;
    //   std::cout << "Arm Joint:    " << targetTrajectories.stateTrajectory[1](12+12+1) << std::endl;
    // }

    for (int i = 0; i < targetTrajectories.stateTrajectory.size(); i++)
    {
      targetTrajectories.stateTrajectory[i].segment(0, 12 + 12) = pre_targetTrajectories.getDesiredState(targetTrajectories.timeTrajectory[i]).segment(0, 12 + 12);
    }

    referenceManagerPtr_->setTargetTrajectories(std::move(targetTrajectories));
  };
  // armTargetTrajectoriesSubscriber_ =
  //     nodeHandle.subscribe<ocs2_msgs::mpc_target_trajectories>(topicPrefix_ + "_mpc_target_arm", 1, armTargetTrajectoriesCallback);

  // Observation
  auto observationCallback = [this](const ocs2_msgs::mpc_observation::ConstPtr &msg){
    auto observation = ros_msg_conversions::readObservationMsg(*msg);

    referenceManagerPtr_->observationStateCallback(observation.state);
  };
  observationSubscriber_ = 
      nodeHandle.subscribe<ocs2_msgs::mpc_observation>(topicPrefix_ + "_mpc_observation", 1, observationCallback);
}

}  // namespace ocs2
