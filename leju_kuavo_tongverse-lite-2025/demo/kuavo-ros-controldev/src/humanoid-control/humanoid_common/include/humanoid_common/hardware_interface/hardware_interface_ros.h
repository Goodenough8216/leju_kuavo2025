#pragma once
#include <vector>
#include <ros/ros.h>
#include <ros/service.h>
#include <kuavo_msgs/jointMoveTo.h>
#include <kuavo_msgs/setHwIntialState.h>
#include "humanoid_interface_drake/common/sensor_data.h"
#include "kuavo_msgs/setMotorEncoderRoundService.h"

namespace ocs2
{
  namespace humanoid
  {
    class KuavoHardwareInterface
    {
    private:
      ros::NodeHandlePtr nh_ptr_;
      uint32_t num_joint_;
      bool sensor_subcribe_{false};
      bool is_initialized_{false};
      ros::Publisher joint_cmd_pub_;
      ros::Subscriber sensors_data_sub_;
      SensorDataBuffer sensor_data_buffer_;

    public:
      KuavoHardwareInterface(const ros::NodeHandlePtr &nh_ptr, size_t num_joint, bool sensor_subcriber = false) : nh_ptr_(nh_ptr),
                                                                                                                    num_joint_(num_joint),
                                                                                                                    sensor_subcribe_(sensor_subcriber),
                                                                                                                    sensor_data_buffer_(20){};
      ~KuavoHardwareInterface();
      void msg2SensorData(const kuavo_msgs::sensorsData::ConstPtr &msg, SensorData_t &sensor_data_motor);
      void sensorDataCallback(const kuavo_msgs::sensorsData::ConstPtr &msg);
      int8_t init(SensorData_t &init_sensor_data);
      void getSensorData(SensorData_t &sensor_data_motor);
      void writeCommand(Eigen::VectorXd cmd_r, std::vector<uint8_t> control_modes);
      void jointMoveTo(std::vector<double> goal_pos, double speed, double dt, bool is_intial_state = false);
      void calibrateMotor(int motor_id, int direction, bool save_offset = false);

    };

    void KuavoHardwareInterface::sensorDataCallback(const kuavo_msgs::sensorsData::ConstPtr &msg)
    {
      // if (msg->joint_data.joint_q.size() != num_joint_)
      // {
      //   std::cerr << "sensorDataCallback Error: joint data size not match!"
      //             << "msg->joint_data.joint_q.size():" << msg->joint_data.joint_q.size() << " num_joint_:" << num_joint_ << std::endl;
      //   return;
      // }
      // TODO: only control legs
      SensorData_t sensor_data_motor;
      msg2SensorData(msg, sensor_data_motor);
      sensor_data_buffer_.addData(sensor_data_motor);
    }

    void KuavoHardwareInterface::msg2SensorData(const kuavo_msgs::sensorsData::ConstPtr &msg, SensorData_t &sensor_data_motor)
    {
      sensor_data_motor.time = msg->sensor_time.toSec();
      sensor_data_motor.resizeJoint(num_joint_);
      for (uint32_t i = 0; i < num_joint_; i++)
      {
        sensor_data_motor.joint_q[i] = msg->joint_data.joint_q[i];
        sensor_data_motor.joint_v[i] = msg->joint_data.joint_v[i];
        sensor_data_motor.joint_vd[i] = msg->joint_data.joint_vd[i];
        sensor_data_motor.joint_current[i] = msg->joint_data.joint_current[i];
      }
      sensor_data_motor.quat = {msg->imu_data.quat.w, msg->imu_data.quat.x, msg->imu_data.quat.y, msg->imu_data.quat.z};
      sensor_data_motor.acc = {msg->imu_data.acc.x, msg->imu_data.acc.y, msg->imu_data.acc.z};
      sensor_data_motor.gyro = {msg->imu_data.gyro.x, msg->imu_data.gyro.y, msg->imu_data.gyro.z};
      sensor_data_motor.free_acc = {msg->imu_data.free_acc.x, msg->imu_data.free_acc.y, msg->imu_data.free_acc.z};
    }
    void KuavoHardwareInterface::getSensorData(SensorData_t &sensor_data_motor)
    {
      if (!is_initialized_)
      {
        std::cerr << "getSensorData Error: not initialized!" << std::endl;
        return;
      }
      if (!sensor_subcribe_)
      {
        std::cerr << "getSensorData Error: sensor_subcriber is set to false, you can not get sensor data from this interface!" << std::endl;
        return;
      }
      sensor_data_motor = sensor_data_buffer_.getData(ros::Time::now().toSec());
    }
    int8_t KuavoHardwareInterface::init(SensorData_t &init_sensor_data)
    {
      joint_cmd_pub_ = nh_ptr_->advertise<kuavo_msgs::jointCmd>("joint_cmd", 10);
      sensors_data_sub_ = nh_ptr_->subscribe("/sensors_data_raw", 10, &KuavoHardwareInterface::sensorDataCallback, this);
      std::cout << "waiting for message on topic /sensors_data_raw ...\n";
      while (ros::ok() && !sensor_data_buffer_.isReady())
      {
        ros::Duration(0.001).sleep();
      }
      if (!sensor_data_buffer_.isReady())
      {
        std::cout << "wait for message on topic /sensors_data_raw failed!\n";
        std::raise(SIGINT);
        return -1;
      }
      is_initialized_ = true;
      init_sensor_data = sensor_data_buffer_.getData(ros::Time::now().toSec());
      std::cout << "initial sensor_data_buffer_ is ready!\n";
      if (!sensor_subcribe_)
      {
        std::cout << "[KuavoHardwareInterface]: sensor_subcribe_ is false, no need to subscribe /sensors_data_raw\n";
        sensors_data_sub_.shutdown();
      }
      return 0;
    }
    void KuavoHardwareInterface::calibrateMotor(int motor_id, int direction, bool save_offset)
    {
      ros::ServiceClient client;
      bool res = false;
      bool service_available = ros::service::waitForService("hardware/modify_motor_encoder_offset", ros::Duration(5.0));
      if (!service_available)
      {
        ROS_ERROR("Failed to connect to modify_motor_encoder_offset service");
        return;
      }
      client = nh_ptr_->serviceClient<kuavo_msgs::setMotorEncoderRoundService>("hardware/modify_motor_encoder_offset");
      kuavo_msgs::setMotorEncoderRoundService srv;
      srv.request.motor_id = motor_id;
      srv.request.direction = direction;
      srv.request.save_offset = save_offset;
      res = client.call(srv);
      if (!res)
      {
        ROS_ERROR("Failed to call modify_motor_encoder_offset service");
        ROS_ERROR("%s", srv.response.message.c_str());
      }
      
    }
    void KuavoHardwareInterface::jointMoveTo(std::vector<double> goal_pos, double speed, double dt, bool is_intial_state)
    {
      ros::ServiceClient client;
      bool res = false;
      if (is_intial_state)
      {
        bool service_available = ros::service::waitForService("hardware/set_intial_state", ros::Duration(5.0));
        if (!service_available)
        {
          ROS_ERROR("Failed to connect to set_intial_state service");
          return;
        }
        client = nh_ptr_->serviceClient<kuavo_msgs::setHwIntialState>("hardware/set_intial_state");
        kuavo_msgs::setHwIntialState srv;
        srv.request.q_intial = goal_pos;
        srv.request.v_intial = std::vector<double>(num_joint_, 0);
        res = client.call(srv);
        if (!res)
        {
          ROS_ERROR("Failed to call jointMoveTo/set_intial_state service");
          ROS_ERROR("%s", srv.response.message.c_str());
        }
      }
      else
      {
        bool service_available = ros::service::waitForService("hardware/joint_move_to", ros::Duration(5.0));
        if (!service_available)
        {
          ROS_ERROR("Failed to connect to joint_move_to service");
          return;
        }
        client = nh_ptr_->serviceClient<kuavo_msgs::jointMoveTo>("hardware/joint_move_to");
        kuavo_msgs::jointMoveTo srv;
        srv.request.goal_position = goal_pos;
        srv.request.dt = dt;
        srv.request.speed = speed;
        res = client.call(srv);
        if (!res)
        {
          ROS_ERROR("Failed to call jointMoveTo service");
          ROS_ERROR("%s", srv.response.message.c_str());
        }
      }
    }
    void KuavoHardwareInterface::writeCommand(Eigen::VectorXd cmd_r, std::vector<uint8_t> control_modes)
    {
      if (cmd_r.size() != num_joint_ * 5)
      {
        std::cerr << "cmd_r size is not correct" << std::endl;
        return;
      }

      kuavo_msgs::jointCmd ros_joint_cmd;
      ros_joint_cmd.header.stamp = ros::Time::now();
      ros_joint_cmd.control_modes.resize(num_joint_);
      ros_joint_cmd.joint_q.resize(num_joint_);
      ros_joint_cmd.joint_v.resize(num_joint_);
      ros_joint_cmd.tau.resize(num_joint_);
      ros_joint_cmd.tau_max.resize(num_joint_);
      ros_joint_cmd.tau_ratio.resize(num_joint_);
      for (uint32_t i = 0; i < num_joint_; i++)
      {
        ros_joint_cmd.joint_q[i] = cmd_r[num_joint_ * 0 + i];

        ros_joint_cmd.joint_v[i] = cmd_r[num_joint_ * 1 + i];

        ros_joint_cmd.tau[i] = cmd_r[num_joint_ * 2 + i];

        ros_joint_cmd.tau_max[i] = cmd_r[num_joint_ * 3 + i];

        ros_joint_cmd.tau_ratio[i] = cmd_r[num_joint_ * 4 + i];

        ros_joint_cmd.control_modes[i] = control_modes[i];
      }

      joint_cmd_pub_.publish(ros_joint_cmd);
    }
  }; // namespace humanoid
};   // namespace ocs2
