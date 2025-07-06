#include "ocs2_mobile_manipulator/dynamics/ActuatedXYZYawPitchManipulatorDynamics.h"

namespace ocs2 {
namespace mobile_manipulator {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ActuatedXYZYawPitchManipulatorDynamics::ActuatedXYZYawPitchManipulatorDynamics(ManipulatorModelInfo info, const std::string& modelName,
                                                                         const std::string& modelFolder /*= "/tmp/ocs2"*/,
                                                                         bool recompileLibraries /*= true*/, bool verbose /*= true*/)
    : info_(std::move(info)) {
  this->initialize(info_.stateDim, info_.inputDim, modelName, modelFolder, recompileLibraries, verbose);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ad_vector_t ActuatedXYZYawPitchManipulatorDynamics::systemFlowMap(ad_scalar_t time, const ad_vector_t& state, const ad_vector_t& input,
                                                               const ad_vector_t&) const {
  ad_vector_t dxdt(info_.stateDim);
  const auto theta = state(2);
  const auto v_x = input(0);  // forward velocity in base frame
  const auto v_y = input(1);  // lateral velocity in base frame
  dxdt << cos(theta) * v_x - sin(theta) * v_y, sin(theta) * v_x + cos(theta) * v_y, input.segment<3>(2), input.tail(info_.armDim);
  return dxdt;
}

}  // namespace mobile_manipulator
}  // namespace ocs2
