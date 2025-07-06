#include <ocs2_mobile_manipulator/MobileManipulatorPreComputation.h>
#include <ocs2_mobile_manipulator/constraint/BaseConstraint.h>

#include <ocs2_core/misc/LinearInterpolation.h>

namespace ocs2 {
namespace mobile_manipulator {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
BaseConstraint::BaseConstraint(const ReferenceManager& referenceManager, size_t baseStateDim)
    : StateConstraint(ConstraintOrder::Linear)
    , referenceManagerPtr_(&referenceManager)
    , baseStateDim_(baseStateDim)
  {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
size_t BaseConstraint::getNumConstraints(scalar_t time) const {
  return baseStateDim_;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t BaseConstraint::getValue(scalar_t time, const vector_t& state, const PreComputation& preComputation) const {
  const auto desiredBasePose = interpolateBasePose(time);

  vector_t constraint(baseStateDim_);
  constraint = state.head(baseStateDim_) - desiredBasePose;
  return constraint;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionLinearApproximation BaseConstraint::getLinearApproximation(scalar_t time, const vector_t& state,
                                                                                const PreComputation& preComputation) const {
  const auto desiredBasePose = interpolateBasePose(time);

  auto approximation = VectorFunctionLinearApproximation(baseStateDim_, state.rows(), 0);

  approximation.f = state.head(baseStateDim_) - desiredBasePose;
  approximation.dfdx = matrix_t::Identity(baseStateDim_, baseStateDim_);

  return approximation;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t BaseConstraint::interpolateBasePose(scalar_t time) const {
  const auto& targetTrajectories = referenceManagerPtr_->getTargetTrajectories();
  const auto& timeTrajectory = targetTrajectories.timeTrajectory;
  const auto& stateTrajectory = targetTrajectories.stateTrajectory;

  vector_t pose = vector_t::Zero(baseStateDim_);
  if(stateTrajectory[0].size() < 14 + baseStateDim_)
  {
    std::cerr << "[BaseConstraint] Error: The state trajectory does not have enough dimensions for the base pose." << std::endl;
    return pose;
  }
  pose = targetTrajectories.getDesiredState(time).tail(baseStateDim_);
  // if (stateTrajectory.size() > 1) {
  //   // Normal interpolation case
  //   int index;
  //   scalar_t alpha;
  //   std::tie(index, alpha) = LinearInterpolation::timeSegment(time, timeTrajectory);

  //   const auto& lhs = stateTrajectory[index];
  //   const auto& rhs = stateTrajectory[index + 1];

  //   pose = alpha * lhs.tail(baseStateDim_) + (1.0 - alpha) * rhs.tail(baseStateDim_);

  // } else {  // stateTrajectory.size() == 1
  //   pose = stateTrajectory.front().tail(baseStateDim_);
  // }
  return pose;
}

  bool BaseConstraint::isActive(scalar_t time) const 
  {
    const auto& stateTrajectory = referenceManagerPtr_->getTargetTrajectories().stateTrajectory;
    if(stateTrajectory[0].size() < 14 + baseStateDim_)
     return false;
    else
      return true;
  }

} // namespace mobile_manipulator
} // namespace ocs2