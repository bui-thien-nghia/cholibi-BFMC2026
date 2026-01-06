// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from bfmc_interfaces:msg/Control.idl
// generated code does not contain a copyright notice

#ifndef BFMC_INTERFACES__MSG__DETAIL__CONTROL__STRUCT_H_
#define BFMC_INTERFACES__MSG__DETAIL__CONTROL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Control in the package bfmc_interfaces.
typedef struct bfmc_interfaces__msg__Control
{
  float velocity;
  float steering_angle;
  int32_t id;
} bfmc_interfaces__msg__Control;

// Struct for a sequence of bfmc_interfaces__msg__Control.
typedef struct bfmc_interfaces__msg__Control__Sequence
{
  bfmc_interfaces__msg__Control * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} bfmc_interfaces__msg__Control__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // BFMC_INTERFACES__MSG__DETAIL__CONTROL__STRUCT_H_
