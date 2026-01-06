// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from bfmc_interfaces:msg/Control.idl
// generated code does not contain a copyright notice

#ifndef BFMC_INTERFACES__MSG__DETAIL__CONTROL__STRUCT_HPP_
#define BFMC_INTERFACES__MSG__DETAIL__CONTROL__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__bfmc_interfaces__msg__Control __attribute__((deprecated))
#else
# define DEPRECATED__bfmc_interfaces__msg__Control __declspec(deprecated)
#endif

namespace bfmc_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Control_
{
  using Type = Control_<ContainerAllocator>;

  explicit Control_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->velocity = 0.0f;
      this->steering_angle = 0.0f;
      this->id = 0l;
    }
  }

  explicit Control_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->velocity = 0.0f;
      this->steering_angle = 0.0f;
      this->id = 0l;
    }
  }

  // field types and members
  using _velocity_type =
    float;
  _velocity_type velocity;
  using _steering_angle_type =
    float;
  _steering_angle_type steering_angle;
  using _id_type =
    int32_t;
  _id_type id;

  // setters for named parameter idiom
  Type & set__velocity(
    const float & _arg)
  {
    this->velocity = _arg;
    return *this;
  }
  Type & set__steering_angle(
    const float & _arg)
  {
    this->steering_angle = _arg;
    return *this;
  }
  Type & set__id(
    const int32_t & _arg)
  {
    this->id = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    bfmc_interfaces::msg::Control_<ContainerAllocator> *;
  using ConstRawPtr =
    const bfmc_interfaces::msg::Control_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<bfmc_interfaces::msg::Control_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<bfmc_interfaces::msg::Control_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      bfmc_interfaces::msg::Control_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<bfmc_interfaces::msg::Control_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      bfmc_interfaces::msg::Control_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<bfmc_interfaces::msg::Control_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<bfmc_interfaces::msg::Control_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<bfmc_interfaces::msg::Control_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__bfmc_interfaces__msg__Control
    std::shared_ptr<bfmc_interfaces::msg::Control_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__bfmc_interfaces__msg__Control
    std::shared_ptr<bfmc_interfaces::msg::Control_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Control_ & other) const
  {
    if (this->velocity != other.velocity) {
      return false;
    }
    if (this->steering_angle != other.steering_angle) {
      return false;
    }
    if (this->id != other.id) {
      return false;
    }
    return true;
  }
  bool operator!=(const Control_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Control_

// alias to use template instance with default allocator
using Control =
  bfmc_interfaces::msg::Control_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace bfmc_interfaces

#endif  // BFMC_INTERFACES__MSG__DETAIL__CONTROL__STRUCT_HPP_
