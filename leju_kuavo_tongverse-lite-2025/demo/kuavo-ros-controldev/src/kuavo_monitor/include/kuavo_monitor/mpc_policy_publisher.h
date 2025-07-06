#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>

#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <functional>

#include <Eigen/Core>
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>

#include <ocs2_msgs/mpc_observation.h>
#include <ocs2_msgs/mpc_flattened_controller.h>

namespace
{
    constexpr int POLICY_MAX_NUM = 50; // Little Law, 100Hz (MPC) * 0.5s (Horizon)
}

namespace HighlyDynamic
{
    // ****************************************************************************************
    // ****************************************************************************************
    // ****************************************************************************************
    template <typename T>
    class ThreadSafeQueue
    {
    private:
        std::queue<T> queue_;
        mutable std::mutex mutex_;
        std::condition_variable cond_;

    public:
        ThreadSafeQueue() {}

        void push(T value)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(value));
            cond_.notify_one();
        }

        T pop()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.wait(lock, [this]
                       { return !queue_.empty(); });
            T value = std::move(queue_.front());
            queue_.pop();
            return value;
        }

        bool empty() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }

        inline int size() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }
    };
    // ****************************************************************************************
    // ****************************************************************************************
    // ****************************************************************************************
    struct MpcObservation
    {
        double time_diff;
        Eigen::VectorXd state;
        Eigen::VectorXd input;
        int8_t mode;
        MpcObservation() {}
        MpcObservation(double time_diff_in, const Eigen::VectorXd &state_in, const Eigen::VectorXd &input_in, int8_t mode_in)
            : time_diff(time_diff_in), state(state_in), input(input_in), mode(mode_in)
        {
        }
    };
    // ****************************************************************************************
    // ****************************************************************************************
    // ****************************************************************************************
    struct MpcPolicy
    {
        std::queue<double> timeDiffTrajectory;
        std::queue<Eigen::VectorXd> stateTrajectory;
        std::queue<Eigen::VectorXd> inputTrajectory;
        std::queue<int8_t> modeTrajectory;
        void pop()
        {
            timeDiffTrajectory.pop();
            stateTrajectory.pop();
            inputTrajectory.pop();
            modeTrajectory.pop();
        }

        MpcObservation front() const
        {
            return MpcObservation(timeDiffTrajectory.front(), stateTrajectory.front(), inputTrajectory.front(), modeTrajectory.front());
        }

        bool empty() const
        {
            if (this == nullptr)
            {
                std::cout << "this is nullptr" << std::endl;
                return true;
            }
            assert(this != nullptr);
            return stateTrajectory.empty();
        }
    };

    // ****************************************************************************************
    // ****************************************************************************************
    // ****************************************************************************************
    class MpcPolicyPublisher
    {
    public:
        MpcPolicyPublisher(ros::NodeHandle &nh) : nh_(nh), pool_(POLICY_MAX_NUM)
        {
            init();
        }
        void run();

    private:
        void init();
        void publishSinglePolicy(const MpcObservation &observation, int id);
        void mpcPolicyCallback(const ocs2_msgs::mpc_flattened_controller::ConstPtr &msg);
        void executePubTask(int id);

    private:
        mutable std::mutex mutex_;
        ThreadSafeQueue<MpcPolicy> policy_queue_;
        std::vector<MpcPolicy> policy_vec_; // TODO: 使用指针数组代替，减少数据拷贝消耗及内存占用
        ros::NodeHandle nh_;
        boost::asio::thread_pool pool_;
        std::vector<ros::Publisher> mpc_policy_publishers_;
        ros::Subscriber sub_;
        bool recieved_first_msg_ = false;
    };

} // namespace HighlyDynamic