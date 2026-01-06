// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from bfmc_interfaces:msg/LaneInfo.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "bfmc_interfaces/msg/detail/lane_info__rosidl_typesupport_introspection_c.h"
#include "bfmc_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "bfmc_interfaces/msg/detail/lane_info__functions.h"
#include "bfmc_interfaces/msg/detail/lane_info__struct.h"


// Include directives for member types
// Member `left_coeffs`
// Member `right_coeffs`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__LaneInfo_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  bfmc_interfaces__msg__LaneInfo__init(message_memory);
}

void bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__LaneInfo_fini_function(void * message_memory)
{
  bfmc_interfaces__msg__LaneInfo__fini(message_memory);
}

size_t bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__size_function__LaneInfo__left_coeffs(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__get_const_function__LaneInfo__left_coeffs(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__get_function__LaneInfo__left_coeffs(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__fetch_function__LaneInfo__left_coeffs(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__get_const_function__LaneInfo__left_coeffs(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__assign_function__LaneInfo__left_coeffs(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__get_function__LaneInfo__left_coeffs(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__resize_function__LaneInfo__left_coeffs(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__size_function__LaneInfo__right_coeffs(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__get_const_function__LaneInfo__right_coeffs(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__get_function__LaneInfo__right_coeffs(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__fetch_function__LaneInfo__right_coeffs(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__get_const_function__LaneInfo__right_coeffs(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__assign_function__LaneInfo__right_coeffs(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__get_function__LaneInfo__right_coeffs(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__resize_function__LaneInfo__right_coeffs(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__LaneInfo_message_member_array[6] = {
  {
    "offset",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces__msg__LaneInfo, offset),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "detected",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces__msg__LaneInfo, detected),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "left_found",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces__msg__LaneInfo, left_found),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "left_coeffs",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces__msg__LaneInfo, left_coeffs),  // bytes offset in struct
    NULL,  // default value
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__size_function__LaneInfo__left_coeffs,  // size() function pointer
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__get_const_function__LaneInfo__left_coeffs,  // get_const(index) function pointer
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__get_function__LaneInfo__left_coeffs,  // get(index) function pointer
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__fetch_function__LaneInfo__left_coeffs,  // fetch(index, &value) function pointer
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__assign_function__LaneInfo__left_coeffs,  // assign(index, value) function pointer
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__resize_function__LaneInfo__left_coeffs  // resize(index) function pointer
  },
  {
    "right_found",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces__msg__LaneInfo, right_found),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "right_coeffs",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(bfmc_interfaces__msg__LaneInfo, right_coeffs),  // bytes offset in struct
    NULL,  // default value
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__size_function__LaneInfo__right_coeffs,  // size() function pointer
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__get_const_function__LaneInfo__right_coeffs,  // get_const(index) function pointer
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__get_function__LaneInfo__right_coeffs,  // get(index) function pointer
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__fetch_function__LaneInfo__right_coeffs,  // fetch(index, &value) function pointer
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__assign_function__LaneInfo__right_coeffs,  // assign(index, value) function pointer
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__resize_function__LaneInfo__right_coeffs  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__LaneInfo_message_members = {
  "bfmc_interfaces__msg",  // message namespace
  "LaneInfo",  // message name
  6,  // number of fields
  sizeof(bfmc_interfaces__msg__LaneInfo),
  bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__LaneInfo_message_member_array,  // message members
  bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__LaneInfo_init_function,  // function to initialize message memory (memory has to be allocated)
  bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__LaneInfo_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__LaneInfo_message_type_support_handle = {
  0,
  &bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__LaneInfo_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_bfmc_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, bfmc_interfaces, msg, LaneInfo)() {
  if (!bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__LaneInfo_message_type_support_handle.typesupport_identifier) {
    bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__LaneInfo_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &bfmc_interfaces__msg__LaneInfo__rosidl_typesupport_introspection_c__LaneInfo_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
