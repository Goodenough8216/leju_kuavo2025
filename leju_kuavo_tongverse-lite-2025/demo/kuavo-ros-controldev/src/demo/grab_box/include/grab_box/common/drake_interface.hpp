#include <iostream>
#include <memory>
#include <drake/math/jacobian.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/geometry/scene_graph.h>


#pragma once

namespace GrabBox
{
  enum class HandSide
  {
    LEFT,
    RIGHT
  };

  class DrakeInterface
  {
  public:
    DrakeInterface(){}
    DrakeInterface(drake::multibody::MultibodyPlant<double> *plant, drake::systems::Context<double>* plant_context, bool print_plant_info=false)
    : plant_(plant)
    , plant_context_(plant_context)
    {
      nq_ = plant_->num_positions();
      nv_ = plant_->num_velocities();
      q0_ = plant_->GetPositions(*plant_context_);
      std::cout << "nq: " << nq_ << " nv: " << nv_ << std::endl;
      if(print_plant_info)
      {
        for (drake::multibody::FrameIndex i(0); i < plant_->num_frames(); ++i) {
            const auto& frame = plant_->get_frame(i);
            std::cout << "Link name[" << i << "]: " << frame.name() << std::endl;
        }
        // 假设 nv_ 已经被设置为 plant_->num_velocities()
        for (drake::multibody::JointIndex j(0); j < plant_->num_joints(); ++j) {
          const auto& joint = plant_->get_joint(j);
          std::cout << "Joint name[" << j << "]: " << joint.name() << std::endl;

          // 获取该关节的自由度数量
          int num_dofs = joint.num_velocities();
          for (int k = 0; k < num_dofs; ++k) {
              // 打印该关节对应的速度索引
              auto velocity_index = joint.velocity_start() + k;
              std::cout << "  Corresponding velocity index: " << velocity_index << std::endl;
          }
        }
      }
    }

    Eigen::MatrixXd getHandJacobian(Eigen::VectorXd q, HandSide side)
    {
      int arm_num = q.size();
      Eigen::VectorXd q_tmp = q0_;
      q_tmp.tail(arm_num) = q;
      std::string hand_link_name = (side == HandSide::LEFT) ? "l_hand_roll" : "r_hand_roll";
      int idx = (side == HandSide::LEFT) ? 0 : 1;
      auto J = getJacobian(q_tmp, hand_link_name);//3*nv_
      return J.middleCols(nv_ - arm_num + idx * (arm_num/2), arm_num/2);
    }

  private:
    Eigen::MatrixXd getJacobian(const Eigen::VectorXd &q, const std::string &link_name)
    {
      Eigen::MatrixXd J(3, nv_);
      const auto& frame = plant_->GetFrameByName(link_name);
      const auto &world_frame = plant_->world_frame();
      plant_->SetPositions(plant_context_, q);
      plant_->CalcJacobianTranslationalVelocity(*plant_context_,
                                            drake::multibody::JacobianWrtVariable::kV,
                                            frame,
                                            Eigen::Vector3d(0, 0, -0.17),//TODO: 外部传入
                                            world_frame,
                                            world_frame,
                                            &J);
      return std::move(J);
    }

  private:
    drake::systems::Context<double> *plant_context_;
    drake::multibody::MultibodyPlant<double> *plant_;
    int nq_, nv_;
    Eigen::VectorXd q0_;
  };
}  // namespace GrabBox