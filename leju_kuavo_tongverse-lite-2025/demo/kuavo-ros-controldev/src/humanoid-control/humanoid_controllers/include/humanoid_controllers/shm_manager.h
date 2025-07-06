#pragma once

#include "shm_data_structure.h"
#include <sys/shm.h>
#include <semaphore.h>
#include <fcntl.h>
#include <memory>

namespace gazebo_shm
{

class ShmManager {
public:
    ShmManager();
    ~ShmManager();

    bool initializeSensorsShm();
    bool initializeCommandShm();
    void cleanup();

    // 传感器数据操作
    bool writeSensorsData(const SensorsData& data);
    bool readSensorsData(SensorsData& data);

    // 关节命令操作
    bool writeJointCommand(const JointCommand& cmd);
    bool readJointCommand(JointCommand& cmd);

    // 同步读写接口
    void setCmdTimeIncrement(double increment);
    bool readSensorsDataSync(gazebo_shm::SensorsData& data, double timeout_ms = 1000.0);
    bool readJointCommandSync(gazebo_shm::JointCommand& cmd, double timeout_ms = 1000.0);
    bool writeJointCommandNext(gazebo_shm::JointCommand& cmd);
    void resetTimestamps();

private:
    bool attachShm(int key, void** shm_ptr, int size);
    void detachShm(void* shm_ptr);
    
    SensorsData* sensors_shm_ptr_;
    JointCommand* cmd_shm_ptr_;
    int sensors_shm_id_;
    int cmd_shm_id_;
    double last_sensor_timestamp_{0.0};
    double last_cmd_timestamp_{0.0};
    double cmd_time_increment_{0.002};

    sem_t* sensors_sem_;
    sem_t* cmd_sem_;
    
    static constexpr const char* SENSORS_SEM_NAME = "/sensors_sem";
    static constexpr const char* CMD_SEM_NAME = "/cmd_sem";
};

} 
