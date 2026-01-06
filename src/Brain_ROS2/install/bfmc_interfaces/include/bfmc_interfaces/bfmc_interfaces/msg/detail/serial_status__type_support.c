// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from bfmc_interfaces:msg/SerialStatus.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "bfmc_interfaces/msg/detail/serial_status__rosidl_typesupport_introspection_c.h"
#include "bfmc_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "bfmc_interfaces/msg/detail/serial_status__functions.h"
#include "bfmc_interfaces/msg/detail/serial_status__struct.h"


// Include directives for member types
// Member `port_name`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void bfmc_interfaces__msg__SerialStatus__rosidl_typesupport_introspection_c__SerialStatus_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  bfmc_interfaces__msg__SerialStatus__init(message_memory);
}

void bfmc_interfaces__msg__SerialStatus__rosidl_typesupport_introspection_c__SerialStatus_fini_function(void * message_memory)
{
  bfmc_interfaces__msg__SerialStatus__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember bfmc_interfaces__msg__SerialStatus__rosidl_typesupport_introspection_c__SerialStatus_message_member_array[2] = {
  {
    "connected",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces__msg__SerialStatus, connected),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "port_name",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces__msg__SerialStatus, port_name),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers bfmc_interfaces__msg__SerialStatus__rosidl_typesupport_introspection_c__SerialStatus_message_members = {
  "bfmc_interfaces__msg",  // message namespace
  "SerialStatus",  // message name
  2,  // number of fields
  sizeof(bfmc_interfaces__msg__SerialStatus),
  bfmc_interfaces__msg__SerialStatus__rosidl_typesupport_introspection_c__SerialStatus_message_member_array,  // message members
  bfmc_interfaces__msg__SerialStatus__rosidl_typesupport_introspection_c__SerialStatus_init_function,  // function to initialize message memory (memory has to be allocated)
  bfmc_interfaces__msg__SerialStatus__rosidl_typesupport_introspection_c__SerialStatus_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t bfmc_interfaces__msg__SerialStatus__rosidl_typesupport_introspection_c__SerialStatus_message_type_support_handle = {
  0,
  &bfmc_interfaces__msg__SerialStatus__rosidl_typesupport_introspection_c__SerialStatus_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_bfmc_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, bfmc_interfaces, msg, SerialStatus)() {
  if (!bfmc_interfaces__msg__SerialStatus__rosidl_typesupport_introspection_c__SerialStatus_message_type_support_handle.typesupport_identifier) {
    bfmc_interfaces__msg__SerialStatus__rosidl_typesupport_introspection_c__SerialStatus_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &bfmc_interfaces__msg__SerialStatus__rosidl_typesupport_introspection_c__SerialStatus_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
