// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from bfmc_interfaces:msg/LaneInfo.idl
// generated code does not contain a copyright notice

#ifndef BFMC_INTERFACES__MSG__DETAIL__LANE_INFO__STRUCT_H_
#define BFMC_INTERFACES__MSG__DETAIL__LANE_INFO__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'left_coeffs'
// Member 'right_coeffs'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/LaneInfo in the package bfmc_interfaces.
typedef struct bfmc_interfaces__msg__LaneInfo
{
  int32_t offset;
  bool detected;
  bool left_found;
  rosidl_runtime_c__double__Sequence left_coeffs;
  bool right_found;
  rosidl_runtime_c__double__Sequence right_coeffs;
} bfmc_interfaces__msg__LaneInfo;

// Struct for a sequence of bfmc_interfaces__msg__LaneInfo.
typedef struct bfmc_interfaces__msg__LaneInfo__Sequence
{
  bfmc_interfaces__msg__LaneInfo * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} bfmc_interfaces__msg__LaneInfo__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // BFMC_INTERFACES__MSG__DETAIL__LANE_INFO__STRUCT_H_
