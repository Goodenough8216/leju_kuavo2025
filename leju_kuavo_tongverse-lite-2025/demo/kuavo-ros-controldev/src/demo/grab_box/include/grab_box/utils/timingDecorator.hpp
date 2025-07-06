#include <chrono>
#include <iostream>
#include "CSVLogger.hpp" 
#include <behaviortree_cpp_v3/decorator_node.h>

class TimingDecorator : public BT::DecoratorNode {
public:
    TimingDecorator(const std::string& name, const BT::NodeConfiguration& config)
        : BT::DecoratorNode(name, config) {}

    
    BT::NodeStatus tick() override {
        BT::NodeStatus status;

        if (!timing_active_) {
            // 子节点刚开始运行，记录开始时间
            start_time_ = std::chrono::high_resolution_clock::now();
            timing_active_ = true;
        }

        // 执行子节点的 tick
        status = child()->executeTick();

        if (status != BT::NodeStatus::RUNNING && timing_active_) {
            // 子节点已完成，记录结束时间并计算耗时
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end_time - start_time_;

            // 记录到 CSV
            CSVLogger::getInstance().log(this->name(), duration.count());

            // 重置状态
            timing_active_ = false;
        }
        return status;
    }


    static BT::PortsList providedPorts() {
        return { };
    }

    private:
    bool timing_active_ = false;
    std::chrono::high_resolution_clock::time_point start_time_;
};
