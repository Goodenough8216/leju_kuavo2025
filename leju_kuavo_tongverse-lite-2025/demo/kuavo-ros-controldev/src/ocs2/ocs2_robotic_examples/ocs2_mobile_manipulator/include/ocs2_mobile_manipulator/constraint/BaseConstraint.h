#pragma once

#include <memory>

#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <ocs2_core/constraint/StateConstraint.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

namespace ocs2 {
namespace mobile_manipulator {

class BaseConstraint final : public StateConstraint {
 public:
  using vector6_t = Eigen::Matrix<scalar_t, 6, 1>;

  BaseConstraint(const ReferenceManager& referenceManager, size_t baseStateDim);
  ~BaseConstraint() override = default;
  BaseConstraint* clone() const override { return new BaseConstraint(*referenceManagerPtr_, baseStateDim_); }

  size_t getNumConstraints(scalar_t time) const override;
  vector_t getValue(scalar_t time, const vector_t& state, const PreComputation& preComputation) const override;
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time, const vector_t& state,
                                                           const PreComputation& preComputation) const override;
  bool isActive(scalar_t time) const override;

 private:
  BaseConstraint(const BaseConstraint& other) = default;
  vector_t interpolateBasePose(scalar_t time) const;

  /** Cached pointer to the pinocchio end effector kinematics. Is set to nullptr if not used. */
  PinocchioEndEffectorKinematics* pinocchioEEKinPtr_ = nullptr;

  // vector6_t baseDesiredPose_;// xyz ypr
  // vector3_t baseDesiredZyx_;
  const ReferenceManager* referenceManagerPtr_;
  const size_t baseStateDim_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
