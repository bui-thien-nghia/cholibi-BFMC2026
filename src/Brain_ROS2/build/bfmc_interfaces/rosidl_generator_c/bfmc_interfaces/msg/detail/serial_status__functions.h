// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from bfmc_interfaces:msg/SerialStatus.idl
// generated code does not contain a copyright notice

#ifndef BFMC_INTERFACES__MSG__DETAIL__SERIAL_STATUS__FUNCTIONS_H_
#define BFMC_INTERFACES__MSG__DETAIL__SERIAL_STATUS__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "bfmc_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "bfmc_interfaces/msg/detail/serial_status__struct.h"

/// Initialize msg/SerialStatus message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * bfmc_interfaces__msg__SerialStatus
 * )) before or use
 * bfmc_interfaces__msg__SerialStatus__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
bool
bfmc_interfaces__msg__SerialStatus__init(bfmc_interfaces__msg__SerialStatus * msg);

/// Finalize msg/SerialStatus message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
void
bfmc_interfaces__msg__SerialStatus__fini(bfmc_interfaces__msg__SerialStatus * msg);

/// Create msg/SerialStatus message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * bfmc_interfaces__msg__SerialStatus__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
bfmc_interfaces__msg__SerialStatus *
bfmc_interfaces__msg__SerialStatus__create();

/// Destroy msg/SerialStatus message.
/**
 * It calls
 * bfmc_interfaces__msg__SerialStatus__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
void
bfmc_interfaces__msg__SerialStatus__destroy(bfmc_interfaces__msg__SerialStatus * msg);

/// Check for msg/SerialStatus message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
bool
bfmc_interfaces__msg__SerialStatus__are_equal(const bfmc_interfaces__msg__SerialStatus * lhs, const bfmc_interfaces__msg__SerialStatus * rhs);

/// Copy a msg/SerialStatus message.
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
bfmc_interfaces__msg__SerialStatus__copy(
  const bfmc_interfaces__msg__SerialStatus * input,
  bfmc_interfaces__msg__SerialStatus * output);

/// Initialize array of msg/SerialStatus messages.
/**
 * It allocates the memory for the number of elements and calls
 * bfmc_interfaces__msg__SerialStatus__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
bool
bfmc_interfaces__msg__SerialStatus__Sequence__init(bfmc_interfaces__msg__SerialStatus__Sequence * array, size_t size);

/// Finalize array of msg/SerialStatus messages.
/**
 * It calls
 * bfmc_interfaces__msg__SerialStatus__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
void
bfmc_interfaces__msg__SerialStatus__Sequence__fini(bfmc_interfaces__msg__SerialStatus__Sequence * array);

/// Create array of msg/SerialStatus messages.
/**
 * It allocates the memory for the array and calls
 * bfmc_interfaces__msg__SerialStatus__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
bfmc_interfaces__msg__SerialStatus__Sequence *
bfmc_interfaces__msg__SerialStatus__Sequence__create(size_t size);

/// Destroy array of msg/SerialStatus messages.
/**
 * It calls
 * bfmc_interfaces__msg__SerialStatus__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
void
bfmc_interfaces__msg__SerialStatus__Sequence__destroy(bfmc_interfaces__msg__SerialStatus__Sequence * array);

/// Check for msg/SerialStatus message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_bfmc_interfaces
bool
bfmc_interfaces__msg__SerialStatus__Sequence__are_equal(const bfmc_interfaces__msg__SerialStatus__Sequence * lhs, const bfmc_interfaces__msg__SerialStatus__Sequence * rhs);

/// Copy an array of msg/SerialStatus messages.
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
bfmc_interfaces__msg__SerialStatus__Sequence__copy(
  const bfmc_interfaces__msg__SerialStatus__Sequence * input,
  bfmc_interfaces__msg__SerialStatus__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // BFMC_INTERFACES__MSG__DETAIL__SERIAL_STATUS__FUNCTIONS_H_
