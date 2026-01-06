// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from bfmc_interfaces:msg/SerialStatus.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "bfmc_interfaces/msg/detail/serial_status__struct.hpp"
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

void SerialStatus_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) bfmc_interfaces::msg::SerialStatus(_init);
}

void SerialStatus_fini_function(void * message_memory)
{
  auto typed_message = static_cast<bfmc_interfaces::msg::SerialStatus *>(message_memory);
  typed_message->~SerialStatus();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember SerialStatus_message_member_array[2] = {
  {
    "connected",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces::msg::SerialStatus, connected),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "port_name",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces::msg::SerialStatus, port_name),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers SerialStatus_message_members = {
  "bfmc_interfaces::msg",  // message namespace
  "SerialStatus",  // message name
  2,  // number of fields
  sizeof(bfmc_interfaces::msg::SerialStatus),
  SerialStatus_message_member_array,  // message members
  SerialStatus_init_function,  // function to initialize message memory (memory has to be allocated)
  SerialStatus_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t SerialStatus_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &SerialStatus_message_members,
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
get_message_type_support_handle<bfmc_interfaces::msg::SerialStatus>()
{
  return &::bfmc_interfaces::msg::rosidl_typesupport_introspection_cpp::SerialStatus_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, bfmc_interfaces, msg, SerialStatus)() {
  return &::bfmc_interfaces::msg::rosidl_typesupport_introspection_cpp::SerialStatus_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
