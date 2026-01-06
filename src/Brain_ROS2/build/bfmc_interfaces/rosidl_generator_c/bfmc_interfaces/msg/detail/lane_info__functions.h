// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from bfmc_interfaces:msg/LaneInfo.idl
// generated code does not contain a copyright notice

#ifndef BFMC_INTERFACES__MSG__DETAIL__LANE_INFO__FUNCTIONS_H_
#define BFMC_INTERFACES__MSG__DETAIL__LANE_INFO__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "bfmc_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "bfmc_interfaces/msg/detail/lane_info__struct.h"

/// Initialize msg/LaneInfo message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * bfmc_interfaces__msg__LaneInfo
 * )) before or use
 * bfmc_interfaces__msg__LaneInfo__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
bool
bfmc_interfaces__msg__LaneInfo__init(bfmc_interfaces__msg__LaneInfo * msg);

/// Finalize msg/LaneInfo message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
void
bfmc_interfaces__msg__LaneInfo__fini(bfmc_interfaces__msg__LaneInfo * msg);

/// Create msg/LaneInfo message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * bfmc_interfaces__msg__LaneInfo__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
bfmc_interfaces__msg__LaneInfo *
bfmc_interfaces__msg__LaneInfo__create();

/// Destroy msg/LaneInfo message.
/**
 * It calls
 * bfmc_interfaces__msg__LaneInfo__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
void
bfmc_interfaces__msg__LaneInfo__destroy(bfmc_interfaces__msg__LaneInfo * msg);

/// Check for msg/LaneInfo message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
bool
bfmc_interfaces__msg__LaneInfo__are_equal(const bfmc_interfaces__msg__LaneInfo * lhs, const bfmc_interfaces__msg__LaneInfo * rhs);

/// Copy a msg/LaneInfo message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
bool
bfmc_interfaces__msg__LaneInfo__copy(
  const bfmc_interfaces__msg__LaneInfo * input,
  bfmc_interfaces__msg__LaneInfo * output);

/// Initialize array of msg/LaneInfo messages.
/**
 * It allocates the memory for the number of elements and calls
 * bfmc_interfaces__msg__LaneInfo__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
bool
bfmc_interfaces__msg__LaneInfo__Sequence__init(bfmc_interfaces__msg__LaneInfo__Sequence * array, size_t size);

/// Finalize array of msg/LaneInfo messages.
/**
 * It calls
 * bfmc_interfaces__msg__LaneInfo__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
void
bfmc_interfaces__msg__LaneInfo__Sequence__fini(bfmc_interfaces__msg__LaneInfo__Sequence * array);

/// Create array of msg/LaneInfo messages.
/**
 * It allocates the memory for the array and calls
 * bfmc_interfaces__msg__LaneInfo__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
bfmc_interfaces__msg__LaneInfo__Sequence *
bfmc_interfaces__msg__LaneInfo__Sequence__create(size_t size);

/// Destroy array of msg/LaneInfo messages.
/**
 * It calls
 * bfmc_interfaces__msg__LaneInfo__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
void
bfmc_interfaces__msg__LaneInfo__Sequence__destroy(bfmc_interfaces__msg__LaneInfo__Sequence * array);

/// Check for msg/LaneInfo message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
bool
bfmc_interfaces__msg__LaneInfo__Sequence__are_equal(const bfmc_interfaces__msg__LaneInfo__Sequence * lhs, const bfmc_interfaces__msg__LaneInfo__Sequence * rhs);

/// Copy an array of msg/LaneInfo messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
bool
bfmc_interfaces__msg__LaneInfo__Sequence__copy(
  const bfmc_interfaces__msg__LaneInfo__Sequence * input,
  bfmc_interfaces__msg__LaneInfo__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // BFMC_INTERFACES__MSG__DETAIL__LANE_INFO__FUNCTIONS_H_
