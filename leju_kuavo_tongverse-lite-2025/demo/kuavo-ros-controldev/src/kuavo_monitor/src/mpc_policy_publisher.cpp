#include "kuavo_monitor/mpc_policy_publisher.h"

namespace HighlyDynamic
{

    void MpcPolicyPublisher::init()
    {
        policy_vec_.resize(POLICY_MAX_NUM);
        for (int i = 0; i < POLICY_MAX_NUM; ++i)
        {
            policy_vec_[i] = MpcPolicy(); // 显式初始化每个MpcPolicy对象
        }
        // policy subscriber
        sub_ = nh_.subscribe<ocs2_msgs::mpc_flattened_controller>(
            "humanoid_mpc_policy",                                        // topic name
            1,                                                            // queue length
            boost::bind(&MpcPolicyPublisher::mpcPolicyCallback, this, _1) // callback
        );
        mpc_policy_publishers_.reserve(POLICY_MAX_NUM);

        for (int i = 0; i < POLICY_MAX_NUM; i++)
        {
            std::string topic_name = "monitor/mpc_policy_" + std::to_string(i);
            auto pub = nh_.advertise<ocs2_msgs::mpc_observation>(topic_name, 1);
            mpc_policy_publishers_.push_back(pub);
        }
    }

    void MpcPolicyPublisher::mpcPolicyCallback(const ocs2_msgs::mpc_flattened_controller::ConstPtr &msg)
    {
        const size_t N = msg->timeTrajectory.size();
        std::vector<int> stateDim(N);
        std::vector<int> inputDim(N);
        MpcPolicy mpc_policy;
        // TODO: add modeTrajectory
        for (size_t i = 0; i < N; i++)
        {
            stateDim[i] = msg->stateTrajectory[i].value.size();
            inputDim[i] = msg->inputTrajectory[i].value.size();
            if (i == 0)
                mpc_policy.timeDiffTrajectory.push(0);
            else
                mpc_policy.timeDiffTrajectory.push(msg->timeTrajectory[i] - msg->timeTrajectory[i - 1]);

            mpc_policy.stateTrajectory.push(
                Eigen::Map<const Eigen::VectorXf>(msg->stateTrajectory[i].value.data(), stateDim[i]).cast<double>());
            mpc_policy.inputTrajectory.push(
                Eigen::Map<const Eigen::VectorXf>(msg->inputTrajectory[i].value.data(), inputDim[i]).cast<double>());
        }
        if (policy_queue_.size() >= POLICY_MAX_NUM)
            policy_queue_.pop();
        policy_queue_.push(mpc_policy);
        recieved_first_msg_ = true;
        // std::cout << "Received message" << std::endl;
    }

    void MpcPolicyPublisher::publishSinglePolicy(const MpcObservation &observation, int id)
    {
        auto observationToMsg = [](const MpcObservation &ob) -> ocs2_msgs::mpc_observation
        {
            ocs2_msgs::mpc_observation msg;
            msg.time = ob.time_diff;
            msg.mode = ob.mode;
            const int N = ob.state.rows();
            msg.state.value.reserve(N);
            for (int i = 0; i < N; i++)
            {
                msg.state.value.push_back(ob.state(i));
                msg.input.value.push_back(ob.input(i));
            }
            return msg;
        };
        mpc_policy_publishers_[id].publish(observationToMsg(observation));
    }

    void MpcPolicyPublisher::run()
    {
        std::thread spin_thread([&]()
                                {
        while (ros::ok())
        {
            ros::spinOnce();
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 避免高频率spinOnce调用
        } });

        for (int i = 0; i < POLICY_MAX_NUM; i++)
        {
            boost::asio::post(pool_, boost::bind(&MpcPolicyPublisher::executePubTask, this, i));
        }
        pool_.join();
        spin_thread.join();
    }

    void MpcPolicyPublisher::executePubTask(int id)
    {
        while (!recieved_first_msg_)
        {
            std::cout << "Waiting for first message..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        while (ros::ok())
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                // 优先填充vector前序的元素
                for (int j = 0; j < POLICY_MAX_NUM; j++)
                {
                    if (policy_vec_[j].empty() && !policy_queue_.empty())
                        policy_vec_[j] = policy_queue_.pop();
                }
            }
            MpcObservation mpc_observation;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!policy_vec_[id].empty())
                {
                    mpc_observation = policy_vec_[id].front();
                    policy_vec_[id].pop();
                }
            }
            if (mpc_observation.state.size() > 0)
            {
                int dt_microseconds = mpc_observation.time_diff * 1e6;

                // std::cout << "dt: " << 0.001 * dt_microseconds << "ms" << std::endl;
                std::this_thread::sleep_for(std::chrono::microseconds(dt_microseconds));
                publishSinglePolicy(mpc_observation, id);
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::microseconds(1000));
            }
        }
    }
} // namespace HighlyDynamic