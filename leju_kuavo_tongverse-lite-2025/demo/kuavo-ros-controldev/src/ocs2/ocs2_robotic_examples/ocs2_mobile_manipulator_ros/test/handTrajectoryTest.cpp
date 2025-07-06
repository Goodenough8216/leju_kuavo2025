#include <ros/ros.h>
#include <ocs2_msgs/mpc_target_trajectories.h>
#include <ocs2_ros_interfaces/common/RosMsgConversions.h>
#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_mpc/SystemObservation.h>

/**
 * @brief single hand control
 *  */ 
ocs2::TargetTrajectories goalPoseToTargetTrajectories(const Eigen::Vector3d& position_l, const Eigen::Vector3d& position_r, 
                                              const Eigen::Quaterniond& orientation_l, const Eigen::Quaterniond& orientation_r,
                                              const ocs2::vector_t& torso_ref,
                                              const ocs2::SystemObservation& observation) {
  const size_t baseDim = torso_ref.size();
  // time trajectory
  const ocs2::scalar_array_t timeTrajectory{observation.time};
  // state trajectory: 3 + 4 for desired position vector and orientation quaternion
  const ocs2::vector_t target_l = (ocs2::vector_t(7) << position_l, orientation_l.coeffs()).finished();
  const ocs2::vector_t target_r = (ocs2::vector_t(7) << position_r, orientation_r.coeffs()).finished();
  const ocs2::vector_t target = (ocs2::vector_t(14+baseDim) << torso_ref, target_l, target_r).finished();
  const ocs2::vector_array_t stateTrajectory{target};
  // input trajectory
  const ocs2::vector_array_t inputTrajectory{ocs2::vector_t::Zero(observation.input.size())};

  return {timeTrajectory, stateTrajectory, inputTrajectory};
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "hand_trajectory_test");
  ros::NodeHandle n;

  ocs2::SystemObservation currentObservation;
  bool updated = false;

  ros::Publisher pub = n.advertise<ocs2_msgs::mpc_target_trajectories>("/mobile_manipulator_mpc_target", 10);
  ros::Subscriber sub = n.subscribe<ocs2_msgs::mpc_observation>("/mobile_manipulator_mpc_observation", 10, 
    [&](const ocs2_msgs::mpc_observation::ConstPtr &msg)
    {
      // current time, state, input, and subsystem
      currentObservation = ocs2::ros_msg_conversions::readObservationMsg(*msg);
      std::cout << "Received observation time: " << currentObservation.time << std::endl;
      std::cout << "Received observation state: " << currentObservation.state.transpose() << std::endl;
      if(!updated)
       updated = true;
    }
  );
  ros::Rate loop_rate(10);
  Eigen::Quaterniond q(0.6967934370040894, -5.443499517809869e-08, -0.7172719240188599, 1.3007548815835435e-08);
  std::cout << "q coeeficients: " << q.coeffs().transpose() << std::endl;
  Eigen::Vector3d p(0.1, 0.0, 0.3);
  std::vector<Eigen::Vector3d> ps;
  int N_x = 30, N_y = 20, N_z = 10;
  for(int i = 0; i < N_x; i++)
  {
    p.x() += 0.01;
    ps.push_back(p);
  }
  for(int i = 0; i < N_y; i++)
  {
    p.y() -= 0.01;
    ps.push_back(p);
  }
  for(int i = 0; i < N_z; i++)
  {
    p.z() += 0.01;
    ps.push_back(p);
  }
  for(int i = 0; i < N_x; i++)
  {
    p.x() -= 0.01;
    ps.push_back(p);
  }
  for (size_t i = 0; i < (N_x+N_y+N_z+N_x); i++)
  {
    ps.push_back(ps[(N_x+N_y+N_z+N_x)-i-1]);
  }
  

  while(!updated)
  {
    ros::spinOnce();
    loop_rate.sleep();
  }
  // TODO: 适配不同base
  ocs2::vector_t torso_ref = (ocs2::vector_t(3) << 0, 0, 0).finished();
  for(int i = 0; i < ps.size(); i++)
  {
    ros::spinOnce();
    loop_rate.sleep();
    auto ps_l = ps[i];
    auto ps_r = ps[i];
    ps_l.y() += 0.1;
    ps_r.y() -= 0.1;
    auto targetTrajectories = goalPoseToTargetTrajectories(ps_l, ps_r, q, q, torso_ref, currentObservation);
    const auto mpcTargetTrajectoriesMsg = ocs2::ros_msg_conversions::createTargetTrajectoriesMsg(targetTrajectories);
    pub.publish(mpcTargetTrajectoriesMsg);
  }
  return 0;
}