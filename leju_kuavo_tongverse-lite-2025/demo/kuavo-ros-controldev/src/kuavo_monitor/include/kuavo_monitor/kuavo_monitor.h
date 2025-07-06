#include <iostream>
#include <ros/ros.h>
#include <std_msgs/Float64.h>

#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <string>

#include <ocs2_core/misc/LoadData.h>

#include "kuavo_monitor/data_analyzer.hpp"
#include "kuavo_monitor/mpc_policy_publisher.h"

namespace HighlyDynamic
{
    class KuavoMonitor
    {
    public:
        KuavoMonitor(ros::NodeHandle &nh, const std::string &taskfile, double wbc_frequency_des);
        void run();

    private:
        DataAnalyzer mpc_frequency_analyzer_;
        DataAnalyzer mpc_time_cost_analyzer_;
        DataAnalyzer wbc_frequency_analyzer_;
        DataAnalyzer wbc_time_cost_analyzer_;
        ros::NodeHandle nh_;
        ros::Subscriber mpc_frequency_sub_;
        ros::Subscriber mpc_time_cost_sub_;
        ros::Subscriber wbc_frequency_sub_;
        ros::Subscriber wbc_time_cost_sub_;

        // MpcPolicyPublisher mpc_policy_pub_;

        // for analyzing
        double warning_threshold_, error_threshold_;
        int total_num_, recent_num_;
        double mpc_frequency_des_, wbc_frequency_des_;
        double mpc_frequency_bias_percent_, wbc_frequency_bias_percent_;
        bool print_warning_;

        void mpcFrequencyCallback(const std_msgs::Float64 &msg);
        void mpcTimecostCallback(const std_msgs::Float64 &msg);
        void wbcFrequencyCallback(const std_msgs::Float64 &msg);
        void wbcTimecostCallback(const std_msgs::Float64 &msg);

        void loadParams(const std::string &filename, bool verbose);

        bool doesNodeExist(const std::string& node_name);
    };

} // namespace HighlyDynamic