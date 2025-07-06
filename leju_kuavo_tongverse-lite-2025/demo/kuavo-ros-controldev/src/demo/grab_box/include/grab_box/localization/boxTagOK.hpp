#include <ros/ros.h>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_srvs/SetBool.h>
#include <std_msgs/Int32.h>
#include <kuavo_msgs/setTagId.h>

namespace GrabBox
{
  class BoxTagOK : public BT::ConditionNode
  {
  public:
    BoxTagOK(const std::string &name, const BT::NodeConfiguration &config) 
      : BT::ConditionNode(name, config)
    {
      ros::NodeHandle nh = ros::NodeHandle("~");
      tag_pose_world_.setZero(7);
      boxWorldPoseSuber = nh.subscribe("/tag_world_pose", 10, &BoxTagOK::tagWorldPoseCallBack, this);
      set_tag_id_client_ = nh.serviceClient<kuavo_msgs::setTagId>("/set_target_tag_id");
      timeout_ = getParamsFromBlackboard<double>(config, "box_tag_ok.timeout");
      ROS_INFO_STREAM("[BoxTagOK] timeout: " << timeout_);
    }

    static BT::PortsList providedPorts()
    {
      return {
              BT::InputPort<int>("tag_id"), 
              BT::InputPort<bool>("time_check"), 
              BT::OutputPort<Eigen::Vector3d>("box_pos"), 
              BT::OutputPort<Eigen::Vector4d>("box_quat")
      };
    }

    BT::NodeStatus tick() override final
    {

      int tag_id = -1;
      if (!getInput("tag_id", tag_id))
      {
        ROS_ERROR("[BoxTagOK] Missing input port: tag_id");
        return BT::NodeStatus::FAILURE;
      }
      bool time_check = false;
      if (!getInput("time_check", time_check))
      {
        ROS_WARN("[BoxTagOK] Missing input port: time_check, set time_check to false!!!");
      }
      if(time_check && (ros::Time::now() - recieved_time_).toSec() > timeout_)
      {
        ROS_ERROR_STREAM("[BoxTagOK] The diff-time(" << (ros::Time::now() - recieved_time_).toSec() 
          << " s) is larger than timeout(" << timeout_ << " s)! tag_id: " << tag_id << " sleep 0.2s and retry...");
        ros::Duration(0.2).sleep(); // 等待0.2秒，避免频繁打印日志
        return BT::NodeStatus::FAILURE;
      }

      // 如果当前的 Tag ID 不匹配，则通过服务设置新的 Tag ID
      if (tag_id != current_tag_id_)
      {
        if (!setTagId(tag_id))
        {
          ROS_ERROR("[BoxTagOK] Failed to set target tag ID via service");
          return BT::NodeStatus::FAILURE;
        }
        current_tag_id_ = tag_id; // 更新当前 Tag ID
        data_received_ = false; // 标记数据未接收，等待下一帧数据
      }

      if (!data_received_) {
        // Data not yet received, stay in this tick until data is available
        return BT::NodeStatus::RUNNING;
      }

      Eigen::Vector3d box_pos(tag_pose_world_(0), tag_pose_world_(1), tag_pose_world_(2));
      Eigen::Vector4d box_quat(tag_pose_world_(3), tag_pose_world_(4), tag_pose_world_(5), tag_pose_world_(6));

      // Output box position and orientation to the behavior tree's blackboard
      setOutput("box_pos", box_pos);
      setOutput("box_quat", box_quat);

      // std::cout << "tag_pose_world_ : " << tag_pose_world_.transpose() << std::endl;

      // // Publish tag_pose_world_ to the blackboard 
      // data_received_ = false;

      return BT::NodeStatus::SUCCESS;
    }

    void tagWorldPoseCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
      recieved_time_ = msg->header.stamp;

      tag_pose_world_(0) = msg->pose.position.x;
      tag_pose_world_(1) = msg->pose.position.y;
      tag_pose_world_(2) = msg->pose.position.z;
      tag_pose_world_(3) = msg->pose.orientation.x;
      tag_pose_world_(4) = msg->pose.orientation.y;
      tag_pose_world_(5) = msg->pose.orientation.z;
      tag_pose_world_(6) = msg->pose.orientation.w;
      config().blackboard->set<Eigen::VectorXd>("tag_pose_world", tag_pose_world_);
      data_received_ = true;
    }

  private:

    bool setTagId(int tag_id)
    {
      kuavo_msgs::setTagId srv;
      srv.request.tag_id = tag_id;
      if (!set_tag_id_client_.call(srv))
      {
        ROS_ERROR("[BoxTagOK] Failed to call set_target_tag_id service");
        return false;
      }
      if (!srv.response.success)
      {
        ROS_ERROR_STREAM("[BoxTagOK] Service response: " << srv.response.message);
        return false;
      }
      ROS_INFO_STREAM("[BoxTagOK] Target tag ID set to " << tag_id << ": " << srv.response.message<< "   (tag_id: " << tag_id << ")");
      return true;
    }


    ros::Subscriber boxWorldPoseSuber;
    ros::ServiceClient set_tag_id_client_;
    Eigen::VectorXd tag_pose_world_;
    bool data_received_ = false;
    int current_tag_id_ = -1;
    double timeout_ = 1.0;
    ros::Time recieved_time_ = ros::Time::now();
  };
} // namespace GrabBox
