// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from bfmc_interfaces:msg/Control.idl
// generated code does not contain a copyright notice

#ifndef BFMC_INTERFACES__MSG__DETAIL__CONTROL__BUILDER_HPP_
#define BFMC_INTERFACES__MSG__DETAIL__CONTROL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "bfmc_interfaces/msg/detail/control__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace bfmc_interfaces
{

namespace msg
{

namespace builder
{

class Init_Control_id
{
public:
  explicit Init_Control_id(::bfmc_interfaces::msg::Control & msg)
  : msg_(msg)
  {}
  ::bfmc_interfaces::msg::Control id(::bfmc_interfaces::msg::Control::_id_type arg)
  {
    msg_.id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::bfmc_interfaces::msg::Control msg_;
};

class Init_Control_steering_angle
{
public:
  explicit Init_Control_steering_angle(::bfmc_interfaces::msg::Control & msg)
  : msg_(msg)
  {}
  Init_Control_id steering_angle(::bfmc_interfaces::msg::Control::_steering_angle_type arg)
  {
    msg_.steering_angle = std::move(arg);
    return Init_Control_id(msg_);
  }

private:
  ::bfmc_interfaces::msg::Control msg_;
};

class Init_Control_velocity
{
public:
  Init_Control_velocity()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Control_steering_angle velocity(::bfmc_interfaces::msg::Control::_velocity_type arg)
  {
    msg_.velocity = std::move(arg);
    return Init_Control_steering_angle(msg_);
  }

private:
  ::bfmc_interfaces::msg::Control msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::bfmc_interfaces::msg::Control>()
{
  return bfmc_interfaces::msg::builder::Init_Control_velocity();
}

}  // namespace bfmc_interfaces

#endif  // BFMC_INTERFACES__MSG__DETAIL__CONTROL__BUILDER_HPP_
