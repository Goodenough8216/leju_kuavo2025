/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#pragma once

#include <pthread.h>
#include <iostream>
#include <thread>
#include <sched.h>
#include <unistd.h>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
namespace ocs2 {

/**
 * Sets the priority of the input thread.
 *
 * @param priority: The priority of the thread from 0 (lowest) to 99 (highest)
 * @param thread: A reference to the tread.
 */
inline void setThreadPriority(int priority, pthread_t thread) {
  sched_param sched{};
  sched.sched_priority = priority;

  if (priority != 0) {
    if (pthread_setschedparam(thread, SCHED_FIFO, &sched) != 0) {
      std::cerr << "WARNING: Failed to set threads priority (one possible reason could be "
                   "that the user and the group permissions are not set properly.)"
                << std::endl;
    }
  }
}


inline void setThreadAffinity(cpu_set_t cpuset, pthread_t thread) {
    // 设置线程的 CPU 亲和性
    if (pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset) != 0) {
        std::cerr << "WARNING: Failed to set thread's CPU affinity (one possible reason could be "
                     "that the user and the group permissions are not set properly.)"
                  << std::endl;
    }
}

inline cpu_set_t getCurrentCpuset() {
    cpu_set_t current_cpuset;
    CPU_ZERO(&current_cpuset);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &current_cpuset) == -1) {
        perror("Error getting current CPU affinity");
    }
    return current_cpuset;
}

// 设置为所有可用的 CPU 并直接应用
inline void setAllCPUs() {
    cpu_set_t all_cpus_set;
    CPU_ZERO(&all_cpus_set);
    int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    for (int i = 0; i < num_cpus; i++) {
        CPU_SET(i, &all_cpus_set);
    }
    
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &all_cpus_set) != 0) {
        perror("Error setting thread CPU affinity to all CPUs");
    }
}

inline void setThreadAffinity(cpu_set_t cpuset, std::thread& thread) {
  setThreadAffinity(cpuset, thread.native_handle());
}

inline void setThreadAffinity(cpu_set_t cpuset) {
  setThreadAffinity(cpuset, pthread_self());
}

// 解析 isolcpus 的 CPU ID
std::vector<int> parseIsolCpus(const std::string& cmdline) {
    std::vector<int> isolated_cpus;
    size_t pos = cmdline.find("isolcpus=");
    
    if (pos != std::string::npos) {
        // 找到 isolcpus 参数，从 "isolcpus=" 后面开始解析
        pos += 9; // 跳过 "isolcpus="
        size_t end_pos = cmdline.find(' ', pos);
        std::string cpus = cmdline.substr(pos, end_pos - pos);

        // 支持的格式包括单个 CPU（如 1）、范围（如 1-3）或多个值（如 1,3,5-7）
        std::stringstream ss(cpus);
        std::string token;
        while (std::getline(ss, token, ',')) {
            size_t dash_pos = token.find('-');
            if (dash_pos != std::string::npos) {
                // 处理范围，如 1-3
                int start = std::stoi(token.substr(0, dash_pos));
                int end = std::stoi(token.substr(dash_pos + 1));
                for (int i = start; i <= end; ++i) {
                    isolated_cpus.push_back(i);
                }
            } else {
                // 处理单个 CPU ID
                isolated_cpus.push_back(std::stoi(token));
            }
        }
    }
    return isolated_cpus;
}

std::vector<int> getIsolatedCpus() {
    std::ifstream cmdline_file("/proc/cmdline");
    if (!cmdline_file.is_open()) {
        std::cerr << "Failed to open /proc/cmdline" << std::endl;
        return {};
    }

    std::string cmdline;
    std::getline(cmdline_file, cmdline);
    
    return parseIsolCpus(cmdline);
}
void setThreadRealtime(pthread_t thread) {
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);

    int ret = pthread_setschedparam(thread, SCHED_FIFO, &param);
    if (ret != 0) {
        std::cerr << "Failed to set real-time priority: " << ret << std::endl;
    }
}

/**
 * Sets the priority of the input thread.
 *
 * @param priority: The priority of the thread from 0 (lowest) to 99 (highest)
 * @param thread: A reference to the tread.
 */
inline void setThreadPriority(int priority, std::thread& thread) {
  setThreadPriority(priority, thread.native_handle());
}

/**
 * Sets the priority of the thread this function is called from.
 *
 * @param priority: The priority of the thread from 0 (lowest) to 99 (highest)
 */
inline void setThisThreadPriority(int priority) {
  setThreadPriority(priority, pthread_self());
}

}  // namespace ocs2
