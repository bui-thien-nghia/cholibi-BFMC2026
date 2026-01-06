// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from bfmc_interfaces:msg/LaneInfo.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "bfmc_interfaces/msg/detail/lane_info__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace bfmc_interfaces
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void LaneInfo_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) bfmc_interfaces::msg::LaneInfo(_init);
}

void LaneInfo_fini_function(void * message_memory)
{
  auto typed_message = static_cast<bfmc_interfaces::msg::LaneInfo *>(message_memory);
  typed_message->~LaneInfo();
}

size_t size_function__LaneInfo__left_coeffs(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__LaneInfo__left_coeffs(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__LaneInfo__left_coeffs(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__LaneInfo__left_coeffs(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__LaneInfo__left_coeffs(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__LaneInfo__left_coeffs(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__LaneInfo__left_coeffs(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__LaneInfo__left_coeffs(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__LaneInfo__right_coeffs(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__LaneInfo__right_coeffs(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__LaneInfo__right_coeffs(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__LaneInfo__right_coeffs(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__LaneInfo__right_coeffs(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__LaneInfo__right_coeffs(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__LaneInfo__right_coeffs(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__LaneInfo__right_coeffs(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember LaneInfo_message_member_array[6] = {
  {
    "offset",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces::msg::LaneInfo, offset),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "detected",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces::msg::LaneInfo, detected),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "left_found",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces::msg::LaneInfo, left_found),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "left_coeffs",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces::msg::LaneInfo, left_coeffs),  // bytes offset in struct
    nullptr,  // default value
    size_function__LaneInfo__left_coeffs,  // size() function pointer
    get_const_function__LaneInfo__left_coeffs,  // get_const(index) function pointer
    get_function__LaneInfo__left_coeffs,  // get(index) function pointer
    fetch_function__LaneInfo__left_coeffs,  // fetch(index, &value) function pointer
    assign_function__LaneInfo__left_coeffs,  // assign(index, value) function pointer
    resize_function__LaneInfo__left_coeffs  // resize(index) function pointer
  },
  {
    "right_found",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces::msg::LaneInfo, right_found),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "right_coeffs",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces::msg::LaneInfo, right_coeffs),  // bytes offset in struct
    nullptr,  // default value
    size_function__LaneInfo__right_coeffs,  // size() function pointer
    get_const_function__LaneInfo__right_coeffs,  // get_const(index) function pointer
    get_function__LaneInfo__right_coeffs,  // get(index) function pointer
    fetch_function__LaneInfo__right_coeffs,  // fetch(index, &value) function pointer
    assign_function__LaneInfo__right_coeffs,  // assign(index, value) function pointer
    resize_function__LaneInfo__right_coeffs  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers LaneInfo_message_members = {
  "bfmc_interfaces::msg",  // message namespace
  "LaneInfo",  // message name
  6,  // number of fields
  sizeof(bfmc_interfaces::msg::LaneInfo),
  LaneInfo_message_member_array,  // message members
  LaneInfo_init_function,  // function to initialize message memory (memory has to be allocated)
  LaneInfo_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t LaneInfo_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &LaneInfo_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace bfmc_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<bfmc_interfaces::msg::LaneInfo>()
{
  return &::bfmc_interfaces::msg::rosidl_typesupport_introspection_cpp::LaneInfo_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, bfmc_interfaces, msg, LaneInfo)() {
  return &::bfmc_interfaces::msg::rosidl_typesupport_introspection_cpp::LaneInfo_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
