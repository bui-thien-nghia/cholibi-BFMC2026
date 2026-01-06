// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from bfmc_interfaces:msg/SerialStatus.idl
// generated code does not contain a copyright notice

#ifndef BFMC_INTERFACES__MSG__DETAIL__SERIAL_STATUS__STRUCT_HPP_
#define BFMC_INTERFACES__MSG__DETAIL__SERIAL_STATUS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__bfmc_interfaces__msg__SerialStatus __attribute__((deprecated))
#else
# define DEPRECATED__bfmc_interfaces__msg__SerialStatus __declspec(deprecated)
#endif

namespace bfmc_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SerialStatus_
{
  using Type = SerialStatus_<ContainerAllocator>;

  explicit SerialStatus_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->connected = false;
      this->port_name = "";
    }
  }

  explicit SerialStatus_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : port_name(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->connected = false;
      this->port_name = "";
    }
  }

  // field types and members
  using _connected_type =
    bool;
  _connected_type connected;
  using _port_name_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _port_name_type port_name;

  // setters for named parameter idiom
  Type & set__connected(
    const bool & _arg)
  {
    this->connected = _arg;
    return *this;
  }
  Type & set__port_name(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->port_name = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    bfmc_interfaces::msg::SerialStatus_<ContainerAllocator> *;
  using ConstRawPtr =
    const bfmc_interfaces::msg::SerialStatus_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<bfmc_interfaces::msg::SerialStatus_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<bfmc_interfaces::msg::SerialStatus_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      bfmc_interfaces::msg::SerialStatus_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<bfmc_interfaces::msg::SerialStatus_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      bfmc_interfaces::msg::SerialStatus_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<bfmc_interfaces::msg::SerialStatus_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<bfmc_interfaces::msg::SerialStatus_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<bfmc_interfaces::msg::SerialStatus_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__bfmc_interfaces__msg__SerialStatus
    std::shared_ptr<bfmc_interfaces::msg::SerialStatus_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__bfmc_interfaces__msg__SerialStatus
    std::shared_ptr<bfmc_interfaces::msg::SerialStatus_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SerialStatus_ & other) const
  {
    if (this->connected != other.connected) {
      return false;
    }
    if (this->port_name != other.port_name) {
      return false;
    }
    return true;
  }
  bool operator!=(const SerialStatus_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SerialStatus_

// alias to use template instance with default allocator
using SerialStatus =
  bfmc_interfaces::msg::SerialStatus_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace bfmc_interfaces

#endif  // BFMC_INTERFACES__MSG__DETAIL__SERIAL_STATUS__STRUCT_HPP_
