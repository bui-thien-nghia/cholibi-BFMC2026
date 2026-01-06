// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from bfmc_interfaces:msg/LaneInfo.idl
// generated code does not contain a copyright notice

#ifndef BFMC_INTERFACES__MSG__DETAIL__LANE_INFO__TRAITS_HPP_
#define BFMC_INTERFACES__MSG__DETAIL__LANE_INFO__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "bfmc_interfaces/msg/detail/lane_info__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace bfmc_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const LaneInfo & msg,
  std::ostream & out)
{
  out << "{";
  // member: offset
  {
    out << "offset: ";
    rosidl_generator_traits::value_to_yaml(msg.offset, out);
    out << ", ";
  }

  // member: detected
  {
    out << "detected: ";
    rosidl_generator_traits::value_to_yaml(msg.detected, out);
    out << ", ";
  }

  // member: left_found
  {
    out << "left_found: ";
    rosidl_generator_traits::value_to_yaml(msg.left_found, out);
    out << ", ";
  }

  // member: left_coeffs
  {
    if (msg.left_coeffs.size() == 0) {
      out << "left_coeffs: []";
    } else {
      out << "left_coeffs: [";
      size_t pending_items = msg.left_coeffs.size();
      for (auto item : msg.left_coeffs) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: right_found
  {
    out << "right_found: ";
    rosidl_generator_traits::value_to_yaml(msg.right_found, out);
    out << ", ";
  }

  // member: right_coeffs
  {
    if (msg.right_coeffs.size() == 0) {
      out << "right_coeffs: []";
    } else {
      out << "right_coeffs: [";
      size_t pending_items = msg.right_coeffs.size();
      for (auto item : msg.right_coeffs) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const LaneInfo & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: offset
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "offset: ";
    rosidl_generator_traits::value_to_yaml(msg.offset, out);
    out << "\n";
  }

  // member: detected
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "detected: ";
    rosidl_generator_traits::value_to_yaml(msg.detected, out);
    out << "\n";
  }

  // member: left_found
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "left_found: ";
    rosidl_generator_traits::value_to_yaml(msg.left_found, out);
    out << "\n";
  }

  // member: left_coeffs
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.left_coeffs.size() == 0) {
      out << "left_coeffs: []\n";
    } else {
      out << "left_coeffs:\n";
      for (auto item : msg.left_coeffs) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: right_found
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "right_found: ";
    rosidl_generator_traits::value_to_yaml(msg.right_found, out);
    out << "\n";
  }

  // member: right_coeffs
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.right_coeffs.size() == 0) {
      out << "right_coeffs: []\n";
    } else {
      out << "right_coeffs:\n";
      for (auto item : msg.right_coeffs) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const LaneInfo & msg, bool use_flow_style = false)
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
  const bfmc_interfaces::msg::LaneInfo & msg,
  std::ostream & out, size_t indentation = 0)
{
  bfmc_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use bfmc_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const bfmc_interfaces::msg::LaneInfo & msg)
{
  return bfmc_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<bfmc_interfaces::msg::LaneInfo>()
{
  return "bfmc_interfaces::msg::LaneInfo";
}

template<>
inline const char * name<bfmc_interfaces::msg::LaneInfo>()
{
  return "bfmc_interfaces/msg/LaneInfo";
}

template<>
struct has_fixed_size<bfmc_interfaces::msg::LaneInfo>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<bfmc_interfaces::msg::LaneInfo>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<bfmc_interfaces::msg::LaneInfo>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // BFMC_INTERFACES__MSG__DETAIL__LANE_INFO__TRAITS_HPP_
