// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from bfmc_interfaces:msg/SerialStatus.idl
// generated code does not contain a copyright notice

#ifndef BFMC_INTERFACES__MSG__DETAIL__SERIAL_STATUS__BUILDER_HPP_
#define BFMC_INTERFACES__MSG__DETAIL__SERIAL_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "bfmc_interfaces/msg/detail/serial_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace bfmc_interfaces
{

namespace msg
{

namespace builder
{

class Init_SerialStatus_port_name
{
public:
  explicit Init_SerialStatus_port_name(::bfmc_interfaces::msg::SerialStatus & msg)
  : msg_(msg)
  {}
  ::bfmc_interfaces::msg::SerialStatus port_name(::bfmc_interfaces::msg::SerialStatus::_port_name_type arg)
  {
    msg_.port_name = std::move(arg);
    return std::move(msg_);
  }

private:
  ::bfmc_interfaces::msg::SerialStatus msg_;
};

class Init_SerialStatus_connected
{
public:
  Init_SerialStatus_connected()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SerialStatus_port_name connected(::bfmc_interfaces::msg::SerialStatus::_connected_type arg)
  {
    msg_.connected = std::move(arg);
    return Init_SerialStatus_port_name(msg_);
  }

private:
  ::bfmc_interfaces::msg::SerialStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::bfmc_interfaces::msg::SerialStatus>()
{
  return bfmc_interfaces::msg::builder::Init_SerialStatus_connected();
}

}  // namespace bfmc_interfaces

#endif  // BFMC_INTERFACES__MSG__DETAIL__SERIAL_STATUS__BUILDER_HPP_
