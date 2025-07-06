#include <behaviortree_cpp_v3/bt_factory.h>
#include <Eigen/Core>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

#pragma once

namespace GrabBox
{
  typedef Eigen::Vector4d torsoPose;
  typedef std::vector<torsoPose> torsoPoseTraj;
  std::vector<std::string> splitString(const std::string& str, char delimiter)
  {
    std::vector<std::string> tokens;
    std::istringstream tokenStream(str);
    std::string token;

    while (std::getline(tokenStream, token, delimiter))
        tokens.push_back(token);
    return tokens;
  }

  template <typename T>
  T getParamsFromBlackboard(const BT::NodeConfiguration& config, const std::string& key)
  {
    auto node = config.blackboard->get<YAML::Node>("config");
    std::vector<std::string> keys = splitString(key, '.');
    if(keys.size() == 1)
      if(node[key])
        return node[key].as<T>();
      else
        throw std::runtime_error("Key[" + key + "] not found in blackboard config");
    else if(keys.size() == 2)
      if(node[keys[0]][keys[1]])
        return node[keys[0]][keys[1]].as<T>();
      else
        throw std::runtime_error("key[" + keys[0] + "." + keys[1]+ "] not found in blackboard config");
    else
      throw std::runtime_error("Invalid key format. Key should be in the form of key1.key2 or key1");
  }
}
// Template specialization to converts a string to torsoPose.
namespace BT
{
  using namespace GrabBox;
  template <> inline Eigen::Vector3d convertFromString(StringView str)
  {
    // We expect real numbers separated by semicolons
    auto parts = splitString(str, ',');
    if (parts.size() != 3)
    {
      throw RuntimeError("invalid input)");
    }
    else
    {
      Eigen::Vector3d output;
      for(int i=0;i<3;i++)
        output(i) = convertFromString<double>(parts[i]);
      return output;
    }
  }

  template <> inline torsoPose convertFromString(StringView str)
  {
    // We expect real numbers separated by semicolons
    auto parts = splitString(str, ',');
    if (parts.size() != 4)
    {
      throw RuntimeError("invalid input)");
    }
    else
    {
      torsoPose output;
      for(int i=0;i<4;i++)
        output(i) = convertFromString<double>(parts[i]);
      return output;
    }
  }

  // Template specialization to converts a string to torsoPoseVec.
  template <> inline torsoPoseTraj convertFromString(StringView str)
  {
    auto poseLists = splitString(str, ';');
    torsoPoseTraj result;
    result.reserve(poseLists.size());
    for (auto& pose : poseLists)
    {
      auto parts = splitString(pose, ',');
      if(parts.size() != 4)
      {
        throw RuntimeError("invalid input)");
      }
      else
      {
        torsoPose pose_tmp;
        for(int i=0;i<4;i++)
          pose_tmp(i) = convertFromString<double>(parts[i]);
        result.push_back(pose_tmp);
      }
    }
    return result;
  }
} // end namespace BT