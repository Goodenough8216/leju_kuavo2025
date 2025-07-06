#include "gazebo-sim/joint_cmd_publisher.h"

JointCmdPublisher::JointCmdPublisher(ros::NodeHandle &nh) : nh_(nh)
{
    std::cout << "[JointCmdPublisher] Initializing..." << std::endl;
    set_model_config_client_ = nh_.serviceClient<gazebo_msgs::SetModelConfiguration>("/gazebo/set_model_configuration");
    set_model_state_client_ = nh_.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");

    
}

void JointCmdPublisher::start()
{
    if (ros::param::has("robot_description"))
    {
        std::string robot_description;
        ros::param::get("robot_description", robot_description);
        robot_name_ = extractRobotNameFromDescription(robot_description);
        ROS_INFO("Robot name extracted from /robot_description: %s", robot_name_.c_str());
    }
    else
    {
        ROS_ERROR("Parameter 'robot_description' not found.");
    }

    waitForGazeboModelToLoad();

    joint_cmd_subscriber_ = nh_.subscribe("/joint_cmd", 10, &JointCmdPublisher::jointCmdCallback, this);
    joints_effort_publisher_ = nh_.advertise<std_msgs::Float64MultiArray>("/limbs_effort_controller/command", 10);
    joint_effort_msg_.data.resize(0);
}

void JointCmdPublisher::waitForGazeboModelToLoad()
{
    std::cout << "[JointCmdPublisher] Waiting for Gazebo model to load..." << std::endl;
    ros::ServiceClient model_states_client = nh_.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
    model_states_client.waitForExistence();
    ROS_INFO("Gazebo model states service is available.");

    while (ros::ok())
    {
        gazebo_msgs::GetModelState srv;
        srv.request.model_name = robot_name_;

        if (model_states_client.call(srv))
        {
            if (srv.response.success)
            {
                ROS_INFO("Model '%s' is loaded in Gazebo.", robot_name_.c_str());
                break;
            }
            else
            {
                ROS_WARN("Model '%s' not found yet. Waiting...", robot_name_.c_str());
            }
        }
        else
        {
            ROS_ERROR("Failed to call service /gazebo/get_model_state.");
        }
        ros::spinOnce();
        usleep(1000000);
    }
    std::cout << "[JointCmdPublisher] Gazebo model is loaded." << std::endl;
    waitForParams();
    setGazeboInitialState();
    waitingForSimStart();
}

bool JointCmdPublisher::handleSimStart(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res)
{
    if (req.data)
    {
        ROS_INFO("Received sim_start request: true");
    }
    else
    {
        ROS_INFO("Received sim_start request: false");
    }
    res.success = true;
    res.message = "Received sim_start request";
    sim_start_received_ = req.data;
    return true;
}

void JointCmdPublisher::waitingForSimStart()
{
    ros::ServiceServer service = nh_.advertiseService("sim_start", &JointCmdPublisher::handleSimStart, this);
    std::cout << "[JointCmdPublisher] Waiting for sim_start service callback..." << std::endl;
    ros::spinOnce();
    while (ros::ok())
    {
        if (sim_start_received_)
        {
            ROS_INFO("Received sim_start request, simulating ...");
            break;
        }
        // setGazeboInitialState();
        ros::Duration(0.05).sleep();
        ros::spinOnce();
    }
}

std::string JointCmdPublisher::extractRobotNameFromDescription(const std::string &robot_description)
{
    size_t start_pos = robot_description.find("<robot name=\"");
    size_t end_pos = robot_description.find("\">", start_pos);
    if (start_pos != std::string::npos && end_pos != std::string::npos)
    {
        return robot_description.substr(start_pos + 13, end_pos - (start_pos + 13));
    }
    return "";
}

void JointCmdPublisher::jointCmdCallback(const kuavo_msgs::jointCmd::ConstPtr &joint_cmd_msg)
{
    joint_effort_msg_.data.clear();
    for (int i = 0; i < active_joint_count_; i++)
        joint_effort_msg_.data.push_back(joint_cmd_msg->tau[i]);
    joints_effort_publisher_.publish(joint_effort_msg_);
}

void JointCmdPublisher::waitForParams()
{
    std::cout << "[JointCmdPublisher] waiting For Params ..." << std::endl;
    int i = 0;
    while (ros::ok())
    {
        if (ros::param::has("robot_init_state_param") && ros::param::has("limbs_effort_controller/joints"))
        {
            std::cout << "[JointCmdPublisher] Gazebo initial state parameter found, proceeding with joint commands..." << std::endl;
            ros::param::get("robot_init_state_param", robot_init_state_param_);
            std::cout << "robot_init_state_param_ " << std::endl;
            if (robot_init_state_param_.size() > 0)
            {
                for (int i = 0; i < robot_init_state_param_.size(); i++)
                {
                    std::cout << "robot_init_state_param_[" << i << "] = " << robot_init_state_param_[i] << std::endl;
                }
                break;
            }
            else
            {
                if (i++ % 1000 == 0)
                    ROS_WARN("Gazebo initial state not set, retrying...");
                usleep(100000);
            }
        }
        else
        {
            ROS_WARN("Gazebo waiting for initial state parameter /robot_init_state_param ...");
            usleep(100000);
        }
        ros::spinOnce();
    }
    std::cout << "[JointCmdPublisher] Initialization complete." << std::endl;
    ros::ServiceClient model_states_client = nh_.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
    model_states_client.waitForExistence();
    ROS_INFO("Gazebo model states service is available.");
    ROS_INFO("Waiting for service /gazebo/set_model_configuration...");
    set_model_config_client_.waitForExistence();
    ROS_INFO("Service /gazebo/set_model_configuration is available.");
    if (ros::param::has("limbs_effort_controller/joints"))
    {
        ros::param::get("limbs_effort_controller/joints", joint_names_);
        ROS_INFO("Successfully fetched joint names. %lu joints found.", joint_names_.size());
    }
    else
    {
        ROS_ERROR("Parameter 'limbs_effort_controller/joints' not found.");
    }
    active_joint_count_ = joint_names_.size();
}

void JointCmdPublisher::setGazeboInitialState()
{
    gazebo_msgs::SetModelConfiguration set_model_config_srv;
    set_model_config_srv.request.model_name = robot_name_;
    set_model_config_srv.request.urdf_param_name = "robot_description";
    set_model_config_srv.request.joint_names = joint_names_;
    std::vector<double> robot_init_positions(joint_names_.size(), 0.0);

    for (int i = 7; i < robot_init_state_param_.size(); i++)
    {
        robot_init_positions[i - 7] = robot_init_state_param_[i];
    }
    set_model_config_srv.request.joint_positions = robot_init_positions;
    for (int i = 0; i < set_model_config_srv.request.joint_positions.size(); i++)
    {
        std::cout << "joint_positions[" << i << "] " << set_model_config_srv.request.joint_names[i] << "= " << set_model_config_srv.request.joint_positions[i] << std::endl;
    }

    if (set_model_config_client_.call(set_model_config_srv))
    {
        if (set_model_config_srv.response.success)
            ROS_INFO("Successfully set Gazebo initial joint positions.%s", set_model_config_srv.response.status_message.c_str());
        else
            ROS_ERROR("Failed to set Gazebo initial joint positions.%s", set_model_config_srv.response.status_message.c_str());
    }
    else
    {
        ROS_ERROR("Failed to set Gazebo initial joint positions.");
    }
    usleep(100000);

    geometry_msgs::Quaternion orientation;
    orientation.w = robot_init_state_param_[3];
    orientation.x = robot_init_state_param_[4];
    orientation.y = robot_init_state_param_[5];
    orientation.z = robot_init_state_param_[6];
    geometry_msgs::Point position;
    position.x = robot_init_state_param_[0];
    position.y = robot_init_state_param_[1];
    position.z = robot_init_state_param_[2];

    setTorsoPositionAndOrientation(robot_name_, position, orientation);
}

void JointCmdPublisher::setTorsoPositionAndOrientation(const std::string &model_name, const geometry_msgs::Point &position, const geometry_msgs::Quaternion &orientation)
{
    ROS_INFO("Waiting for service /gazebo/set_model_state...");
    set_model_state_client_.waitForExistence();
    ROS_INFO("Service /gazebo/set_model_state is available.");

    gazebo_msgs::SetModelState set_model_state_srv;
    set_model_state_srv.request.model_state.model_name = model_name;
    set_model_state_srv.request.model_state.pose.position = position;
    set_model_state_srv.request.model_state.pose.orientation = orientation;
    set_model_state_srv.request.model_state.twist.linear.x = 0.0;
    set_model_state_srv.request.model_state.twist.linear.y = 0.0;
    set_model_state_srv.request.model_state.twist.linear.z = 0.0;
    set_model_state_srv.request.model_state.twist.angular.x = 0.0;
    set_model_state_srv.request.model_state.twist.angular.y = 0.0;
    set_model_state_srv.request.model_state.twist.angular.z = 0.0;
    set_model_state_srv.request.model_state.reference_frame = "world";

    if (set_model_state_client_.call(set_model_state_srv))
    {
        if (set_model_state_srv.response.success)
        {
            ROS_INFO("Successfully set torso position and orientation.");
        }
        else
        {
            ROS_ERROR("Failed to set torso position and orientation. Response: %s", set_model_state_srv.response.status_message.c_str());
        }
    }
    else
    {
        ROS_ERROR("Failed to call service /gazebo/set_model_state.");
    }
}
