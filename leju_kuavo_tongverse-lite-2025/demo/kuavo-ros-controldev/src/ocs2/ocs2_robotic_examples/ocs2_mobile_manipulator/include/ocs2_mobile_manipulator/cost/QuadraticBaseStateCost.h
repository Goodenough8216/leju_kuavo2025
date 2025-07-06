#pragma once

#include <ocs2_core/cost/QuadraticStateInputCost.h>

namespace ocs2 {
namespace mobile_manipulator {

class QuadraticBaseStateCost final : public QuadraticStateInputCost {
 public:
  QuadraticBaseStateCost(matrix_t Q, size_t stateDim, size_t inputDim)
      : QuadraticStateInputCost(std::move(Q)
      , matrix_t::Zero(inputDim, inputDim))
      , stateDim_(stateDim)
      , inputDim_(inputDim)
      {}

  ~QuadraticBaseStateCost() override = default;

  QuadraticBaseStateCost(const QuadraticBaseStateCost& rhs) = default;
  QuadraticBaseStateCost* clone() const override { return new QuadraticBaseStateCost(*this); }

  std::pair<vector_t, vector_t> getStateInputDeviation(scalar_t time, const vector_t& state, const vector_t& input,
                                                       const TargetTrajectories& targetTrajectories,
                                                       const PreComputation& preComp) const override {
    vector_t stateDeviation = vector_t::Zero(stateDim_);
    const auto target = targetTrajectories.getDesiredState(time);
    if(target.size() > 14)
      stateDeviation.head(stateDim_) = state.head(stateDim_) - targetTrajectories.getDesiredState(time).head(stateDim_);
    return {stateDeviation, vector_t::Zero(inputDim_)};
  }

 private:
  const size_t stateDim_;
  const size_t inputDim_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
