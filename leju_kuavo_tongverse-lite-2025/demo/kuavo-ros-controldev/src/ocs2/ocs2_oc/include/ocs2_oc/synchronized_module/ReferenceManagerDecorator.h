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

#pragma once

#include "ocs2_oc/synchronized_module/ReferenceManagerInterface.h"

namespace ocs2 {

/**
 * Implements the basic decorator functionality for the ReferenceManager
 * Forwards all methods to the wrapped object.
 */
class ReferenceManagerDecorator : public ReferenceManagerInterface {
 public:
  explicit ReferenceManagerDecorator(std::shared_ptr<ReferenceManagerInterface> referenceManagerPtr)
      : referenceManagerPtr_(std::move(referenceManagerPtr)) {
    if (referenceManagerPtr_ == nullptr) {
      throw std::runtime_error("[ReferenceManagerDecorator] ReferenceManager pointer is nullptr!");
    }
  }

  ~ReferenceManagerDecorator() override = default;

  void preSolverRun(scalar_t initTime, scalar_t finalTime, const vector_t& initState) override {
    referenceManagerPtr_->preSolverRun(initTime, finalTime, initState);
  }

  const ModeSchedule& getModeSchedule() const override { return referenceManagerPtr_->getModeSchedule(); }
  void setModeSchedule(const ModeSchedule& modeSchedule) override { referenceManagerPtr_->setModeSchedule(modeSchedule); }
  void setModeSchedule(ModeSchedule&& modeSchedule) override { referenceManagerPtr_->setModeSchedule(std::move(modeSchedule)); }

  const TargetTrajectories& getTargetTrajectories() const override { return referenceManagerPtr_->getTargetTrajectories(); }
  void setTargetTrajectories(const TargetTrajectories& targetTrajectories) override {
    referenceManagerPtr_->setTargetTrajectories(targetTrajectories);
  }
  void setTargetTrajectories(TargetTrajectories&& targetTrajectories) override {
    referenceManagerPtr_->setTargetTrajectories(std::move(targetTrajectories));
  }
  void updateBuffer() override { referenceManagerPtr_->updateBuffer(); }

  void setSwingHeight(scalar_t toe_height,  scalar_t heel_height) override {
    referenceManagerPtr_->setSwingHeight(toe_height, heel_height);
  }

  double getChangeQTime(void) { return referenceManagerPtr_->getChangeQTime(); }
  double getChangeRTime(void) { return referenceManagerPtr_->getChangeRTime(); }

  matrix_t getMatrixQ(void) { return referenceManagerPtr_->getMatrixQ(); }
  matrix_t getMatrixR(void) { return referenceManagerPtr_->getMatrixR(); }

  inline bool getUpdatedR() const override{ return referenceManagerPtr_->getUpdatedR(); }
  inline bool getUpdatedQ() const override{ return referenceManagerPtr_->getUpdatedQ(); }
  inline void setUpdatedR(bool flag) override{ referenceManagerPtr_->setUpdatedR(flag); }
  inline void setUpdatedQ(bool flag) override{ referenceManagerPtr_->setUpdatedQ(flag); }
  
  inline std::vector<scalar_t> getSwingPlannerMultipliers() override
  {
    return referenceManagerPtr_->getSwingPlannerMultipliers();
  }

  virtual void resetReference(const std::vector<scalar_t> multipliers, const ModeSchedule& modeSchedule) override
  {
      referenceManagerPtr_->resetReference(multipliers, modeSchedule);
  }

 protected:
  std::shared_ptr<ReferenceManagerInterface> referenceManagerPtr_;
};
}  // namespace ocs2
