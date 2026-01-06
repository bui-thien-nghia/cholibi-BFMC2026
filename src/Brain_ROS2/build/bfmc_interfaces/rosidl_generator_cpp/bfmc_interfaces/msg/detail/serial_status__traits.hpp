// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from bfmc_interfaces:msg/SerialStatus.idl
// generated code does not contain a copyright notice

#ifndef BFMC_INTERFACES__MSG__DETAIL__SERIAL_STATUS__TRAITS_HPP_
#define BFMC_INTERFACES__MSG__DETAIL__SERIAL_STATUS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "bfmc_interfaces/msg/detail/serial_status__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace bfmc_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const SerialStatus & msg,
  std::ostream & out)
{
  out << "{";
  // member: connected
  {
    out << "connected: ";
    rosidl_generator_traits::value_to_yaml(msg.connected, out);
    out << ", ";
  }

  // member: port_name
  {
    out << "port_name: ";
    rosidl_generator_traits::value_to_yaml(msg.port_name, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SerialStatus & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: connected
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "connected: ";
    rosidl_generator_traits::value_to_yaml(msg.connected, out);
    out << "\n";
  }

  // member: port_name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "port_name: ";
    rosidl_generator_traits::value_to_yaml(msg.port_name, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SerialStatus & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace bfmc_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use bfmc_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const bfmc_interfaces::msg::SerialStatus & msg,
  std::ostream & out, size_t indentation = 0)
{
  bfmc_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use bfmc_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const bfmc_interfaces::msg::SerialStatus & msg)
{
  return bfmc_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<bfmc_interfaces::msg::SerialStatus>()
{
  return "bfmc_interfaces::msg::SerialStatus";
}

template<>
inline const char * name<bfmc_interfaces::msg::SerialStatus>()
{
  return "bfmc_interfaces/msg/SerialStatus";
}

template<>
struct has_fixed_size<bfmc_interfaces::msg::SerialStatus>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<bfmc_interfaces::msg::SerialStatus>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<bfmc_interfaces::msg::SerialStatus>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // BFMC_INTERFACES__MSG__DETAIL__SERIAL_STATUS__TRAITS_HPP_
