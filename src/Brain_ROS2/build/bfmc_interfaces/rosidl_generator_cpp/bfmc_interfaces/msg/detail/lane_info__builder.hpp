// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from bfmc_interfaces:msg/LaneInfo.idl
// generated code does not contain a copyright notice

#ifndef BFMC_INTERFACES__MSG__DETAIL__LANE_INFO__BUILDER_HPP_
#define BFMC_INTERFACES__MSG__DETAIL__LANE_INFO__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "bfmc_interfaces/msg/detail/lane_info__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace bfmc_interfaces
{

namespace msg
{

namespace builder
{

class Init_LaneInfo_right_coeffs
{
public:
  explicit Init_LaneInfo_right_coeffs(::bfmc_interfaces::msg::LaneInfo & msg)
  : msg_(msg)
  {}
  ::bfmc_interfaces::msg::LaneInfo right_coeffs(::bfmc_interfaces::msg::LaneInfo::_right_coeffs_type arg)
  {
    msg_.right_coeffs = std::move(arg);
    return std::move(msg_);
  }

private:
  ::bfmc_interfaces::msg::LaneInfo msg_;
};

class Init_LaneInfo_right_found
{
public:
  explicit Init_LaneInfo_right_found(::bfmc_interfaces::msg::LaneInfo & msg)
  : msg_(msg)
  {}
  Init_LaneInfo_right_coeffs right_found(::bfmc_interfaces::msg::LaneInfo::_right_found_type arg)
  {
    msg_.right_found = std::move(arg);
    return Init_LaneInfo_right_coeffs(msg_);
  }

private:
  ::bfmc_interfaces::msg::LaneInfo msg_;
};

class Init_LaneInfo_left_coeffs
{
public:
  explicit Init_LaneInfo_left_coeffs(::bfmc_interfaces::msg::LaneInfo & msg)
  : msg_(msg)
  {}
  Init_LaneInfo_right_found left_coeffs(::bfmc_interfaces::msg::LaneInfo::_left_coeffs_type arg)
  {
    msg_.left_coeffs = std::move(arg);
    return Init_LaneInfo_right_found(msg_);
  }

private:
  ::bfmc_interfaces::msg::LaneInfo msg_;
};

class Init_LaneInfo_left_found
{
public:
  explicit Init_LaneInfo_left_found(::bfmc_interfaces::msg::LaneInfo & msg)
  : msg_(msg)
  {}
  Init_LaneInfo_left_coeffs left_found(::bfmc_interfaces::msg::LaneInfo::_left_found_type arg)
  {
    msg_.left_found = std::move(arg);
    return Init_LaneInfo_left_coeffs(msg_);
  }

private:
  ::bfmc_interfaces::msg::LaneInfo msg_;
};

class Init_LaneInfo_detected
{
public:
  explicit Init_LaneInfo_detected(::bfmc_interfaces::msg::LaneInfo & msg)
  : msg_(msg)
  {}
  Init_LaneInfo_left_found detected(::bfmc_interfaces::msg::LaneInfo::_detected_type arg)
  {
    msg_.detected = std::move(arg);
    return Init_LaneInfo_left_found(msg_);
  }

private:
  ::bfmc_interfaces::msg::LaneInfo msg_;
};

class Init_LaneInfo_offset
{
public:
  Init_LaneInfo_offset()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_LaneInfo_detected offset(::bfmc_interfaces::msg::LaneInfo::_offset_type arg)
  {
    msg_.offset = std::move(arg);
    return Init_LaneInfo_detected(msg_);
  }

private:
  ::bfmc_interfaces::msg::LaneInfo msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::bfmc_interfaces::msg::LaneInfo>()
{
  return bfmc_interfaces::msg::builder::Init_LaneInfo_offset();
}

}  // namespace bfmc_interfaces

#endif  // BFMC_INTERFACES__MSG__DETAIL__LANE_INFO__BUILDER_HPP_
