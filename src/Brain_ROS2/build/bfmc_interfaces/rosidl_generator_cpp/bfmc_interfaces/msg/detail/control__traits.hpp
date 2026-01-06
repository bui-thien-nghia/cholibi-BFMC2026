// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from bfmc_interfaces:msg/Control.idl
// generated code does not contain a copyright notice

#ifndef BFMC_INTERFACES__MSG__DETAIL__CONTROL__TRAITS_HPP_
#define BFMC_INTERFACES__MSG__DETAIL__CONTROL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "bfmc_interfaces/msg/detail/control__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace bfmc_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const Control & msg,
  std::ostream & out)
{
  out << "{";
  // member: velocity
  {
    out << "velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.velocity, out);
    out << ", ";
  }

  // member: steering_angle
  {
    out << "steering_angle: ";
    rosidl_generator_traits::value_to_yaml(msg.steering_angle, out);
    out << ", ";
  }

  // member: id
  {
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Control & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.velocity, out);
    out << "\n";
  }

  // member: steering_angle
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "steering_angle: ";
    rosidl_generator_traits::value_to_yaml(msg.steering_angle, out);
    out << "\n";
  }

  // member: id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Control & msg, bool use_flow_style = false)
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
  const bfmc_interfaces::msg::Control & msg,
  std::ostream & out, size_t indentation = 0)
{
  bfmc_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use bfmc_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const bfmc_interfaces::msg::Control & msg)
{
  return bfmc_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<bfmc_interfaces::msg::Control>()
{
  return "bfmc_interfaces::msg::Control";
}

template<>
inline const char * name<bfmc_interfaces::msg::Control>()
{
  return "bfmc_interfaces/msg/Control";
}

template<>
struct has_fixed_size<bfmc_interfaces::msg::Control>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<bfmc_interfaces::msg::Control>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<bfmc_interfaces::msg::Control>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // BFMC_INTERFACES__MSG__DETAIL__CONTROL__TRAITS_HPP_
