// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from bfmc_interfaces:msg/SerialStatus.idl
// generated code does not contain a copyright notice

#ifndef BFMC_INTERFACES__MSG__DETAIL__SERIAL_STATUS__STRUCT_H_
#define BFMC_INTERFACES__MSG__DETAIL__SERIAL_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'port_name'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/SerialStatus in the package bfmc_interfaces.
typedef struct bfmc_interfaces__msg__SerialStatus
{
  bool connected;
  rosidl_runtime_c__String port_name;
} bfmc_interfaces__msg__SerialStatus;

// Struct for a sequence of bfmc_interfaces__msg__SerialStatus.
typedef struct bfmc_interfaces__msg__SerialStatus__Sequence
{
  bfmc_interfaces__msg__SerialStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} bfmc_interfaces__msg__SerialStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // BFMC_INTERFACES__MSG__DETAIL__SERIAL_STATUS__STRUCT_H_
