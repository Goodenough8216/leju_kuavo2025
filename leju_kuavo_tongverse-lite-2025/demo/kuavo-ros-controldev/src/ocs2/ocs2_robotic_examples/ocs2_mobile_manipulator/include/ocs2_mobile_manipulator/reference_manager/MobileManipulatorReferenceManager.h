#pragma once

#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>

#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_msgs/mpc_observation.h>
#include "ocs2_mobile_manipulator/ManipulatorModelInfo.h"
#include <ocs2_mpc/SystemObservation.h>

namespace ocs2 {
namespace mobile_manipulator {

class MobileManipulatorReferenceManager : public ReferenceManager {
public:
  MobileManipulatorReferenceManager(const ManipulatorModelInfo& info);
  ~MobileManipulatorReferenceManager() override = default;

  // inline void setChangeQTime(const double& time) { QTime_ = time; }
  // inline void setChangeRTime(const double& time) { RTime_ = time; }
  inline double getChangeQTime() const{ return QTime_; }
  inline double getChangeRTime() const{ return RTime_; }

  // inline void setMatrixQ(const matrix_t& Q) { gait_Q_ = Q; }
  // inline void setMatrixR(const matrix_t& R) { gait_R_ = R; }
  virtual matrix_t getMatrixQ() override{ return Q_; }
  virtual matrix_t getMatrixR() override{ return R_; }

  inline bool getUpdatedR() const override{ return updatedR_; }
  inline bool getUpdatedQ() const override{ return updatedQ_; }
  inline void setUpdatedR(bool flag) override{ updatedR_= flag; }
  inline void setUpdatedQ(bool flag) override{ updatedQ_= flag; }

  virtual std::vector<scalar_t> getSwingPlannerMultipliers() override{ return std::vector<scalar_t>(); }

protected:
  virtual void modifyReferences(scalar_t initTime, scalar_t finalTime, const vector_t& initState, TargetTrajectories& targetTrajectories,
                                ModeSchedule& modeSchedule) override;
private:
  void callbackQMatrix(const std_msgs::Float64MultiArray::ConstPtr& msg);
  void callbackRMatrix(const std_msgs::Float64MultiArray::ConstPtr& msg);
  void callbackObservation(const ocs2_msgs::mpc_observation::ConstPtr &msg);

private:
  ros::NodeHandle nodeHandle_;
  ros::Subscriber subQMatrix_;
  ros::Subscriber subRMatrix_;
  ros::Subscriber subObservation_;

  const ManipulatorModelInfo info_;
  SystemObservation latestObservation_;
  size_t baseStateDim_;
  double QTime_ = 0.0;
  double RTime_ = 0.0;
  matrix_t Q_;
  matrix_t R_;
  bool updatedR_ = false; // if updated R in ocp solver
  bool updatedQ_ = false;
};
} // namespace mobile_manipulator
} // namespace ocs2