#include <ros/ros.h>
#include <kuavo_msgs/Metadata.h>
#include <sensor_msgs/CameraInfo.h>
#include <string>
#include <sstream>

class MetadataSimPublisher {
private:
    ros::NodeHandle nh_;
    std::string camera_name_;
    
    // Publishers
    ros::Publisher color_pub_;
    ros::Publisher depth_pub_;
    ros::Publisher infra1_pub_;
    ros::Publisher infra2_pub_;
    
    // Subscribers for camera_info
    ros::Subscriber color_info_sub_;
    ros::Subscriber depth_info_sub_;
    ros::Subscriber infra1_info_sub_;
    ros::Subscriber infra2_info_sub_;
    
    // Store latest timestamps
    ros::Time color_timestamp_;
    ros::Time depth_timestamp_;
    ros::Time infra1_timestamp_;
    ros::Time infra2_timestamp_;
    
    ros::Timer timer_;
    uint64_t frame_number_;

    // Camera info callbacks
    void colorInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
        color_timestamp_ = msg->header.stamp;
    }

    void depthInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
        depth_timestamp_ = msg->header.stamp;
    }

    void infra1InfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
        infra1_timestamp_ = msg->header.stamp;
    }

    void infra2InfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
        infra2_timestamp_ = msg->header.stamp;
    }

    // 创建模拟的JSON metadata
    std::string createMetadataJson(const std::string& stream_type, uint64_t frame_number, const ros::Time& timestamp) {
        std::stringstream ss;
        uint64_t current_time = timestamp.toNSec();
        uint32_t hw_timestamp = frame_number * 16667;
        
        if (stream_type == "Color") {
            ss << "{"
               << "\"frame_number\":" << frame_number << ","
               << "\"clock_domain\":\"global_time\","
               << "\"frame_timestamp\":" << current_time / 1e3 << "," // 转换为微秒
               << "\"frame_counter\":" << frame_number << ","
               << "\"hw_timestamp\":" << hw_timestamp << ","
               << "\"auto_exposure\":1,"
               << "\"time_of_arrival\":" << (current_time / 1e6) << "," // 转换为毫秒
               << "\"backend_timestamp\":0,"
               << "\"actual_fps\":60,"
               << "\"brightness\":0,"
               << "\"contrast\":50,"
               << "\"saturation\":64,"
               << "\"sharpness\":50,"
               << "\"auto_white_balance_temperature\":1,"
               << "\"backlight_compensation\":0,"
               << "\"hue\":0,"
               << "\"gamma\":300,"
               << "\"manual_white_balance\":4600,"
               << "\"power_line_frequency\":3,"
               << "\"low_light_compensation\":0,"
               << "\"raw_frame_size\":614400"
               << "}";
        } else {
            bool is_infrared = (stream_type.find("Infrared") != std::string::npos);
            int emitter_mode = (frame_number % 2 ? 0 : 1);
            if (is_infrared) {
                emitter_mode = (frame_number % 2 ? 1 : 0);
            }

            ss << "{"
               << "\"frame_number\":" << frame_number << ","
               << "\"clock_domain\":\"global_time\","
               << "\"frame_timestamp\":" << current_time / 1e3 << ","
               << "\"frame_counter\":" << frame_number << ","
               << "\"hw_timestamp\":" << hw_timestamp << ","
               << "\"sensor_timestamp\":" << (hw_timestamp - 16000) << ","
               << "\"actual_exposure\":31979,"
               << "\"gain_level\":16,"
               << "\"auto_exposure\":1,"
               << "\"time_of_arrival\":" << (current_time / 1e6) << ","
               << "\"backend_timestamp\":0,"
               << "\"actual_fps\":60,"
               << "\"frame_laser_power\":" << (emitter_mode ? 150 : 0) << ","
               << "\"frame_laser_power_mode\":" << emitter_mode << ","
               << "\"exposure_priority\":1,"
               << "\"exposure_roi_left\":0,"
               << "\"exposure_roi_right\":847,"
               << "\"exposure_roi_top\":0,"
               << "\"exposure_roi_bottom\":479,"
               << "\"frame_emitter_mode\":" << emitter_mode << ","
               << "\"raw_frame_size\":814080,"
               << "\"gpio_input_data\":0,"
               << "\"sequence_name\":15,"
               << "\"sequence_id\":0,"
               << "\"sequence_size\":2"
               << "}";
        }
        return ss.str();
    }

    void publishMetadata(const ros::TimerEvent&) {
        frame_number_++;

        kuavo_msgs::Metadata msg;

        // 发布color metadata
        if (color_timestamp_ != ros::Time(0)) {
            msg.header.stamp = color_timestamp_;
            msg.header.frame_id = "camera_color_optical_frame";
            msg.json_data = createMetadataJson("Color", frame_number_, color_timestamp_);
            color_pub_.publish(msg);
        }

        // 发布depth metadata
        if (depth_timestamp_ != ros::Time(0)) {
            msg.header.stamp = depth_timestamp_;
            msg.header.frame_id = "camera_depth_optical_frame";
            msg.json_data = createMetadataJson("Depth", frame_number_, depth_timestamp_);
            depth_pub_.publish(msg);
        }

        // 发布infra1 metadata
        if (infra1_timestamp_ != ros::Time(0)) {
            msg.header.stamp = infra1_timestamp_;
            msg.header.frame_id = "camera_infra1_optical_frame";
            msg.json_data = createMetadataJson("Infrared 1", frame_number_, infra1_timestamp_);
            infra1_pub_.publish(msg);
        }

        // 发布infra2 metadata
        if (infra2_timestamp_ != ros::Time(0)) {
            msg.header.stamp = infra2_timestamp_;
            msg.header.frame_id = "camera_infra2_optical_frame";
            msg.json_data = createMetadataJson("Infrared 2", frame_number_, infra2_timestamp_);
            infra2_pub_.publish(msg);
        }

        ROS_DEBUG_THROTTLE(1.0, "Publishing metadata for frame %lu", frame_number_);
    }

public:
    MetadataSimPublisher() : frame_number_(0) {
        ros::NodeHandle private_nh("~");
        private_nh.param<std::string>("camera_name", camera_name_, "camera");

        // 创建发布者
        color_pub_ = nh_.advertise<kuavo_msgs::Metadata>(
            camera_name_ + "/color/metadata", 1);
        depth_pub_ = nh_.advertise<kuavo_msgs::Metadata>(
            camera_name_ + "/depth/metadata", 1);
        infra1_pub_ = nh_.advertise<kuavo_msgs::Metadata>(
            camera_name_ + "/infra1/metadata", 1);
        infra2_pub_ = nh_.advertise<kuavo_msgs::Metadata>(
            camera_name_ + "/infra2/metadata", 1);

        // 创建订阅者
        color_info_sub_ = nh_.subscribe(camera_name_ + "/color/camera_info", 1, 
                                      &MetadataSimPublisher::colorInfoCallback, this);
        depth_info_sub_ = nh_.subscribe(camera_name_ + "/depth/camera_info", 1, 
                                      &MetadataSimPublisher::depthInfoCallback, this);
        infra1_info_sub_ = nh_.subscribe(camera_name_ + "/infra1/camera_info", 1, 
                                       &MetadataSimPublisher::infra1InfoCallback, this);
        infra2_info_sub_ = nh_.subscribe(camera_name_ + "/infra2/camera_info", 1, 
                                       &MetadataSimPublisher::infra2InfoCallback, this);

        // 60Hz发布频率
        timer_ = nh_.createTimer(ros::Duration(1.0/60.0), 
                               &MetadataSimPublisher::publishMetadata, this);

        ROS_INFO("Metadata simulator started for camera: %s", camera_name_.c_str());
        ROS_INFO("Synchronizing with camera_info timestamps");
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "camera_metadata_sim_pub");
    MetadataSimPublisher metadata_sim;
    ros::spin();
    return 0;
}