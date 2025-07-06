#include <fstream>
#include <mutex>
#include <string>

class CSVLogger {
public:
    // 获取单例实例
    static CSVLogger& getInstance() {
        static CSVLogger instance;
        return instance;
    }

    // 删除拷贝构造函数和赋值操作符
    CSVLogger(const CSVLogger&) = delete;
    CSVLogger& operator=(const CSVLogger&) = delete;

    // 初始化日志文件，写入CSV头
    void initialize(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            file_.open(filename, std::ios::out | std::ios::trunc);
            if (file_.is_open()) {
                file_ << "NodeName,ExecutionTime_ms\n";
                initialized_ = true;
            }
        }
    }

    // 记录一条日志
    void log(const std::string& nodeName, double execTime) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_ << "\"" << nodeName << "\"," << execTime << "\n";
        }
    }

    // 关闭日志文件
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.close();
            initialized_ = false;
        }
    }

private:
    CSVLogger() : initialized_(false) {}
    ~CSVLogger() {
        close();
    }

    std::ofstream file_;
    std::mutex mutex_;
    bool initialized_;
};
