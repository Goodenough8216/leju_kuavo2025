#pragma once

#include <string>
#include <utility>

#include <ocs2_core/Types.h>

namespace ocs2 {

/** The Kinematics function which maps state-input pair to the end-effector (position, velocity, orientation error) */
template <typename SCALAR_T>
class EndEffectorSpatialKinematics {
 public:
  using vector3_t = Eigen::Matrix<SCALAR_T, 3, 1>;
  using matrix3x_t = Eigen::Matrix<SCALAR_T, 3, Eigen::Dynamic>;
  using vector_t = Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1>;
  using quaternion_t = Eigen::Quaternion<SCALAR_T>;

  EndEffectorSpatialKinematics() = default;
  virtual ~EndEffectorSpatialKinematics() = default;
  virtual EndEffectorSpatialKinematics* clone() const = 0;
  EndEffectorSpatialKinematics& operator=(const EndEffectorSpatialKinematics&) = delete;

  /** Get end-effector IDs (names) */
  virtual const std::vector<std::string>& getIds() const = 0;

  /**
   * Get end-effector position vectors in world frame
   *
   * @param [in] state vector
   * @return array of position vectors
   */
  virtual std::vector<vector3_t> getPosition(const vector_t& state) const = 0;

  /**
   * Get end-effector velocity vectors in world frame
   *
   * @param [in] state: state vector
   * @param [in] input: input vector
   * @return array of velocity vectors
   */
  virtual std::vector<vector3_t> getVelocity(const vector_t& state, const vector_t& input) const = 0;

  /**
   * Get orientation error in world frame
   *
   * @note: To calculate the error use quaternionDistance() from ocs2_robotic_tools/common/RotationTransforms.h
   *
   * @param [in] state vector
   * @param [in] referenceOrientation: reference quaternion
   * @return array of orientation errors
   */
  virtual std::vector<vector3_t> getOrientationError(const vector_t& state,
                                                     const std::vector<quaternion_t>& referenceOrientations) const = 0;
  
  virtual std::vector<vector3_t> getAngularVelocity(const vector_t& state, const vector_t& input) const = 0;

  /**
   * Get end-effector position linear approximation in world frame
   *
   * @param [in] state: state vector
   * @return array of position function linear approximations
   */
  virtual std::vector<VectorFunctionLinearApproximation> getPositionLinearApproximation(const vector_t& state) const = 0;

  /**
   * Get end-effector velocity linear approximation in world frame
   *
   * @param [in] state: state vector
   * @param [in] input: input vector
   * @return array of velocity function linear approximations
   */
  virtual std::vector<VectorFunctionLinearApproximation> getVelocityLinearApproximation(const vector_t& state,
                                                                                        const vector_t& input) const = 0;

  /**
   * Get end-effector orintation error linear approximation in world frame
   *
   * @note: To calculate the error and Jacobian use quaternionDistance() and quaternionDistanceJacobian() from
   *        ocs2_robotic_tools/common/RotationTransforms.h
   *
   * @param [in] state: state vector
   * @param [in] referenceOrientation: reference quaternion
   * @return array of orientation error linear approximations
   */
  virtual std::vector<VectorFunctionLinearApproximation> getOrientationErrorLinearApproximation(
      const vector_t& state, const std::vector<quaternion_t>& referenceOrientations) const = 0;
  
  virtual std::vector<VectorFunctionLinearApproximation> getAngularVelocityLinearApproximation(const vector_t& state,
                                                                                               const vector_t& input) const = 0;

 protected:
  EndEffectorSpatialKinematics(const EndEffectorSpatialKinematics&) = default;
};

}  // namespace ocs2
