// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from bfmc_interfaces:msg/SerialStatus.idl
// generated code does not contain a copyright notice
#include "bfmc_interfaces/msg/detail/serial_status__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `port_name`
#include "rosidl_runtime_c/string_functions.h"

bool
bfmc_interfaces__msg__SerialStatus__init(bfmc_interfaces__msg__SerialStatus * msg)
{
  if (!msg) {
    return false;
  }
  // connected
  // port_name
  if (!rosidl_runtime_c__String__init(&msg->port_name)) {
    bfmc_interfaces__msg__SerialStatus__fini(msg);
    return false;
  }
  return true;
}

void
bfmc_interfaces__msg__SerialStatus__fini(bfmc_interfaces__msg__SerialStatus * msg)
{
  if (!msg) {
    return;
  }
  // connected
  // port_name
  rosidl_runtime_c__String__fini(&msg->port_name);
}

bool
bfmc_interfaces__msg__SerialStatus__are_equal(const bfmc_interfaces__msg__SerialStatus * lhs, const bfmc_interfaces__msg__SerialStatus * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // connected
  if (lhs->connected != rhs->connected) {
    return false;
  }
  // port_name
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->port_name), &(rhs->port_name)))
  {
    return false;
  }
  return true;
}

bool
bfmc_interfaces__msg__SerialStatus__copy(
  const bfmc_interfaces__msg__SerialStatus * input,
  bfmc_interfaces__msg__SerialStatus * output)
{
  if (!input || !output) {
    return false;
  }
  // connected
  output->connected = input->connected;
  // port_name
  if (!rosidl_runtime_c__String__copy(
      &(input->port_name), &(output->port_name)))
  {
    return false;
  }
  return true;
}

bfmc_interfaces__msg__SerialStatus *
bfmc_interfaces__msg__SerialStatus__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  bfmc_interfaces__msg__SerialStatus * msg = (bfmc_interfaces__msg__SerialStatus *)allocator.allocate(sizeof(bfmc_interfaces__msg__SerialStatus), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(bfmc_interfaces__msg__SerialStatus));
  bool success = bfmc_interfaces__msg__SerialStatus__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
bfmc_interfaces__msg__SerialStatus__destroy(bfmc_interfaces__msg__SerialStatus * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    bfmc_interfaces__msg__SerialStatus__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
bfmc_interfaces__msg__SerialStatus__Sequence__init(bfmc_interfaces__msg__SerialStatus__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  bfmc_interfaces__msg__SerialStatus * data = NULL;

  if (size) {
    data = (bfmc_interfaces__msg__SerialStatus *)allocator.zero_allocate(size, sizeof(bfmc_interfaces__msg__SerialStatus), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = bfmc_interfaces__msg__SerialStatus__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        bfmc_interfaces__msg__SerialStatus__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
bfmc_interfaces__msg__SerialStatus__Sequence__fini(bfmc_interfaces__msg__SerialStatus__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      bfmc_interfaces__msg__SerialStatus__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

bfmc_interfaces__msg__SerialStatus__Sequence *
bfmc_interfaces__msg__SerialStatus__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  bfmc_interfaces__msg__SerialStatus__Sequence * array = (bfmc_interfaces__msg__SerialStatus__Sequence *)allocator.allocate(sizeof(bfmc_interfaces__msg__SerialStatus__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = bfmc_interfaces__msg__SerialStatus__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
bfmc_interfaces__msg__SerialStatus__Sequence__destroy(bfmc_interfaces__msg__SerialStatus__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    bfmc_interfaces__msg__SerialStatus__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
bfmc_interfaces__msg__SerialStatus__Sequence__are_equal(const bfmc_interfaces__msg__SerialStatus__Sequence * lhs, const bfmc_interfaces__msg__SerialStatus__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!bfmc_interfaces__msg__SerialStatus__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
bfmc_interfaces__msg__SerialStatus__Sequence__copy(
  const bfmc_interfaces__msg__SerialStatus__Sequence * input,
  bfmc_interfaces__msg__SerialStatus__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(bfmc_interfaces__msg__SerialStatus);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    bfmc_interfaces__msg__SerialStatus * data =
      (bfmc_interfaces__msg__SerialStatus *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!bfmc_interfaces__msg__SerialStatus__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          bfmc_interfaces__msg__SerialStatus__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!bfmc_interfaces__msg__SerialStatus__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
